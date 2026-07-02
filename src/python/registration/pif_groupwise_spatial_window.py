# PIFReg Spatial Window - spatial sliding window + StackFlow3D + mean-blended flows
#
# Scheme A: scan the stack with 128x128 (configurable) windows, run single-level
# StackFlow3D per window (no pyramid), mean-blend per-band 2D flows, warp once.

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..voxelmorph.config import compact_unet_features
from .pif_groupwise_stackflow import (
    StackInput,
    _as_band_list,
    _bands_to_tensor,
    _resolve_device,
    _tensor_to_bands,
)
from .pif_groupwise_stackflow3d import (
    METHOD_FULL_NAME as STACKFLOW3D_FULL_NAME,
    _train_stackflow3d_at_level,
    _upsample_flow_volume,
    _warp_stack_band_flows,
)

METHOD_NAME = 'PIFReg-SpatialWindow'
METHOD_FULL_NAME = 'PIFReg Spatial Window StackFlow3D (mean-blended flows)'

DEFAULT_SPATIAL_WINDOW = 128
DEFAULT_SPATIAL_STRIDE = 64
DEFAULT_MAX_EPOCHS = 500
DEFAULT_PATIENCE = 60


def _iter_axis_starts(length: int, window: int, stride: int) -> List[int]:
    """Window start positions along one axis; always cover the far edge."""
    window = int(window)
    stride = max(1, int(stride))
    if length <= window:
        return [0]
    last = length - window
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_spatial_windows(
    height: int,
    width: int,
    window_h: int,
    window_w: int,
    stride_y: int,
    stride_x: int,
) -> List[Tuple[int, int, int, int]]:
    """Return (y0, x0, ph, pw) tuples; ph/pw may be smaller than window at edges."""
    wh = min(int(window_h), height)
    ww = min(int(window_w), width)
    y_starts = _iter_axis_starts(height, wh, stride_y)
    x_starts = _iter_axis_starts(width, ww, stride_x)
    windows = []
    for y0 in y_starts:
        for x0 in x_starts:
            ph = min(wh, height - y0)
            pw = min(ww, width - x0)
            windows.append((y0, x0, ph, pw))
    return windows


def _crop_stack(
    bands: List[np.ndarray],
    y0: int,
    x0: int,
    ph: int,
    pw: int,
) -> List[np.ndarray]:
    return [b[y0 : y0 + ph, x0 : x0 + pw].copy() for b in bands]


def _flow_volume_to_numpy(flow_vol: torch.Tensor) -> np.ndarray:
    """(1, N, 2, h, w) -> (N, 2, h, w)"""
    return flow_vol.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _blend_flow_patch(
    flow_sum: np.ndarray,
    weight_sum: np.ndarray,
    flow_patch: np.ndarray,
    y0: int,
    x0: int,
) -> None:
    """Accumulate flow_patch into flow_sum with uniform weights (in-place)."""
    ph, pw = flow_patch.shape[-2], flow_patch.shape[-1]
    flow_sum[:, :, y0 : y0 + ph, x0 : x0 + pw] += flow_patch
    weight_sum[y0 : y0 + ph, x0 : x0 + pw] += 1.0


def _mean_blend_flows(flow_sum: np.ndarray, weight_sum: np.ndarray) -> np.ndarray:
    w = np.maximum(weight_sum, 1e-8)
    return (flow_sum / w[np.newaxis, np.newaxis, :, :]).astype(np.float32)


def register_pifreg_groupwise_spatial_window(
    img_list: StackInput,
    device: str = 'cuda',
    anchor_band_idx: int = -1,
    spatial_window: int = DEFAULT_SPATIAL_WINDOW,
    spatial_stride: int = DEFAULT_SPATIAL_STRIDE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    lr: float = 2e-4,
    lamda: float = 0.005,
    ncc_weight: float = 1.0,
    int_steps: int = 3,
    int_downsize: int = 2,
    nb_unet_features=None,
    early_stop: bool = True,
    min_delta: float = 1e-5,
    lr_schedule: str = 'cosine',
    lr_min: float = 1e-6,
    fast_mode: bool = True,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any], np.ndarray]:
    """
    Spatial sliding-window groupwise registration (StackFlow3D, no pyramid, mean flow blend).

    Args:
        spatial_window: square window size, default 128
        spatial_stride: slide step, default 64 (~50% overlap for 128 windows)
        max_epochs / patience: per-window StackFlow3D optimization limits

    Returns:
        registered, info, flow_stack - same convention as stackflow3d; flow_stack is (N-1,2,H,W)
    """
    device = _resolve_device(device)
    bands = _as_band_list(img_list)
    n = len(bands)
    if n <= 1:
        return bands, {'mode': 'spatial_window', 'num_bands': n}, np.zeros(
            (0, 2, 0, 0), dtype=np.float32
        )

    if fast_mode:
        nb_unet_features = nb_unet_features or compact_unet_features()
        lr = 2e-4
        lamda = 0.005

    h, w = bands[0].shape
    anchor_band_idx = int(anchor_band_idx) % n
    win = int(spatial_window)
    stride = max(1, int(spatial_stride))

    windows = iter_spatial_windows(h, w, win, win, stride, stride)
    num_windows = len(windows)

    if verbose:
        y_starts = _iter_axis_starts(h, min(win, h), stride)
        x_starts = _iter_axis_starts(w, min(win, w), stride)
        print(
            f'{METHOD_NAME}: {n} bands, {h}x{w}, anchor={anchor_band_idx}, '
            f'window={win}x{win}, stride={stride}, '
            f'grid={len(y_starts)}x{len(x_starts)}={num_windows} windows, '
            f'max_epochs/window={max_epochs}, patience={patience}',
            flush=True,
        )

    flow_sum = np.zeros((n, 2, h, w), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)

    for wi, (y0, x0, ph, pw) in enumerate(windows):
        window_label = f'Window {wi + 1}/{num_windows}'
        if verbose:
            print(
                f'  {window_label}: y=[{y0}:{y0 + ph}), x=[{x0}:{x0 + pw})',
                flush=True,
            )

        patch_bands = _crop_stack(bands, y0, x0, ph, pw)
        _, flow_vol = _train_stackflow3d_at_level(
            patch_bands,
            device,
            anchor_band_idx,
            max_epochs,
            patience,
            lr,
            lamda,
            ncc_weight,
            int_steps,
            int_downsize,
            nb_unet_features,
            early_stop,
            min_delta,
            lr_schedule,
            lr_min,
            verbose=verbose,
            log_every=10,
            log_prefix=f'  [{window_label}] ',
        )

        if ph != flow_vol.shape[-2] or pw != flow_vol.shape[-1]:
            flow_vol = _upsample_flow_volume(flow_vol, ph, pw)

        flow_patch = _flow_volume_to_numpy(flow_vol)
        _blend_flow_patch(flow_sum, weight_sum, flow_patch, y0, x0)

    flow_full = _mean_blend_flows(flow_sum, weight_sum)
    flow_full[anchor_band_idx] = 0.0

    flow_vol_full = torch.tensor(
        flow_full[np.newaxis, ...], dtype=torch.float32, device=device
    )
    stack_orig = _bands_to_tensor(bands, device)
    warped_t = _warp_stack_band_flows(stack_orig, flow_vol_full, anchor_band_idx, (h, w))
    registered = _tensor_to_bands(warped_t)

    moving_idx = [i for i in range(n) if i != anchor_band_idx]
    flow_stack_np = flow_full[moving_idx].astype(np.float32)

    info = {
        'mode': 'spatial_window',
        'method': METHOD_FULL_NAME,
        'backbone': STACKFLOW3D_FULL_NAME,
        'num_bands': n,
        'num_flow_fields': n - 1,
        'anchor_band_idx': anchor_band_idx,
        'moving_band_indices': moving_idx,
        'flow_stack_shape': list(flow_stack_np.shape),
        'spatial_window': win,
        'spatial_stride': stride,
        'num_spatial_windows': num_windows,
        'window_grid_y': len(_iter_axis_starts(h, min(win, h), stride)),
        'window_grid_x': len(_iter_axis_starts(w, min(win, w), stride)),
        'flow_blend': 'mean',
        'pyramid_levels': [],
        'max_epochs_per_window': int(max_epochs),
        'patience_per_window': int(patience),
        'loss': 'sequential_pairwise_ncc_mean + per_flow_grad (per window)',
        'ncc_weight': ncc_weight,
        'fast_mode': fast_mode,
    }
    return registered, info, flow_stack_np
