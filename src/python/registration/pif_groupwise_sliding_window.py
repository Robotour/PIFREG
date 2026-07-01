# PIFReg Sliding-Window v2 — 无锚点均衡配准 + 顺序窗口 warp + 可切换扫描顺序
#
# 1. 每窗口 W 个波段各有一个 2D flow（3D U-Net，无锚点）
# 2. 窗口完成后立即 warp，下一窗口基于已配准结果（不做 flow 平均）
# 3. schedule:
#    - pyramid_then_windows: 每层金字塔扫完所有窗口（先空间后光谱）
#    - window_then_pyramid:  每个窗口内走完 128→256→512（先光谱后空间）

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from ..losses.registration_losses import Grad, NCC
from ..voxelmorph.config import compact_unet_features
from ..voxelmorph.layers import ResizeTransform, SpatialTransformer, VecInt

from .pif_groupwise_joint import spectral_variance_loss
from .pif_groupwise_stackflow import (
    DEFAULT_PYRAMID_SIZES,
    StackInput,
    _as_band_list,
    _bands_to_tensor,
    _build_pyramid_levels,
    _compose_flows,
    _create_lr_scheduler,
    _downsample_band,
    _downsample_stack,
    _resolve_device,
    _tensor_to_bands,
    _upsample_flow,
    sequential_pairwise_ncc_loss,
)
from .pif_groupwise_stackflow3d import SpectralStackUnet3d, _upsample_flow_volume

METHOD_NAME = 'PIFReg-SlidingWindow'
METHOD_FULL_NAME = 'PIFReg Sliding-Window Balanced Registration (3D U-Net)'

DEFAULT_WINDOW_SIZE = 5
DEFAULT_WINDOW_STRIDE = 1
DEFAULT_EPOCHS_PER_WINDOW = (300, 500, 800)
DEFAULT_PATIENCE_PER_WINDOW = (50, 70, 90)
DEFAULT_LAMDA_SPEC = 0.02
DEFAULT_LAMDA_GAUGE = 0.05
DEFAULT_LAMDA_VAR = 0.1

SCHEDULE_PYRAMID_THEN_WINDOWS = 'pyramid_then_windows'
SCHEDULE_WINDOW_THEN_PYRAMID = 'window_then_pyramid'


def spectral_flow_smoothness_loss(flow_vol: torch.Tensor) -> torch.Tensor:
    """flow_vol: (1, W, 2, h, w)"""
    w = flow_vol.shape[1]
    if w <= 1:
        return flow_vol.sum() * 0.0
    total = sum(
        ((flow_vol[:, i] - flow_vol[:, i + 1]) ** 2).mean()
        for i in range(w - 1)
    )
    return total / (w - 1)


def gauge_mean_flow_loss(preint_vol: torch.Tensor) -> torch.Tensor:
    """无锚点时约束平均位移≈0，避免整体漂移。"""
    mean_flow = preint_vol.mean(dim=1)
    return (mean_flow ** 2).mean()


def _volume_grad_loss(preint_vol, grad_fn) -> torch.Tensor:
    w = preint_vol.shape[1]
    if w == 0:
        return preint_vol.sum() * 0.0
    total = sum(grad_fn(None, preint_vol[:, i, :, :, :]) for i in range(w))
    return total / w


def _level_train_params(level_side: int, base_int_steps=3, base_int_downsize=2):
    if level_side <= 32:
        return min(base_int_steps, 2), 1
    return base_int_steps, base_int_downsize


def _warp_stack_all_flows(stack_t: torch.Tensor, flow_vol: torch.Tensor, shape_hw) -> torch.Tensor:
    """stack (1,W,H,W), flow_vol (1,W,2,h,w) — 每个 band 独立 warp。"""
    device = stack_t.device
    transformer = SpatialTransformer(shape_hw).to(device)
    n = stack_t.shape[1]
    warped = []
    for i in range(n):
        flow_i = flow_vol[:, i, :, :, :]
        warped.append(transformer(stack_t[:, i : i + 1], flow_i))
    return torch.cat(warped, dim=1)


def _compose_flow_volumes(base_vol: torch.Tensor, delta_vol: torch.Tensor, shape_hw, device) -> torch.Tensor:
    """(1, W, 2, H, W) 逐 band compose。"""
    w = base_vol.shape[1]
    composed = []
    for i in range(w):
        composed.append(
            _compose_flows(base_vol[:, i, :, :, :], delta_vol[:, i, :, :, :], shape_hw, device).unsqueeze(1)
        )
    return torch.cat(composed, dim=1)


def _warp_band_numpy(band: np.ndarray, flow: torch.Tensor, device) -> np.ndarray:
    h, w = band.shape
    transformer = SpatialTransformer((h, w)).to(device)
    img_t = torch.tensor(band, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    flow_t = flow.unsqueeze(0) if flow.dim() == 3 else flow
    out = transformer(img_t, flow_t)
    return out.squeeze().detach().cpu().numpy().astype(np.float32)


def _compose_flow_numpy(base: np.ndarray, delta: np.ndarray, device) -> np.ndarray:
    h, w = base.shape[1], base.shape[2]
    b_t = torch.tensor(base, dtype=torch.float32, device=device).unsqueeze(0)
    d_t = torch.tensor(delta, dtype=torch.float32, device=device).unsqueeze(0)
    out = _compose_flows(b_t, d_t, (h, w), device)
    return out.squeeze(0).detach().cpu().numpy().astype(np.float32)


def flow_volume_to_numpy(flow_vol: torch.Tensor) -> np.ndarray:
    return flow_vol.squeeze(0).detach().cpu().numpy().astype(np.float32)


def warp_bands_with_flow_volume(
    bands: StackInput,
    flow_volume: np.ndarray,
    device: str = 'cpu',
) -> List[np.ndarray]:
    """flow_volume: (N, 2, H, W)，逐 band 施加位移。"""
    device = _resolve_device(device)
    band_list = _as_band_list(bands)
    h, w = band_list[0].shape
    flow_t = torch.tensor(flow_volume, dtype=torch.float32, device=device).unsqueeze(0)
    stack_t = _bands_to_tensor(band_list, device)
    warped_t = _warp_stack_all_flows(stack_t, flow_t, (h, w))
    return _tensor_to_bands(warped_t)


class WindowBalancedFlowNet3d(nn.Module):
    """
    窗口 3D U-Net：输入 (1,1,W,H,W)，输出 W 个 per-band 2D flow，无锚点。
    """

    def __init__(
        self,
        num_bands,
        spatial_shape,
        nb_unet_features=None,
        int_steps=3,
        int_downsize=2,
    ):
        super().__init__()
        self.num_bands = int(num_bands)
        h, w = spatial_shape

        self.unet3d = SpectralStackUnet3d(in_channels=1, nb_features=nb_unet_features)
        self.flow = nn.Conv3d(self.unet3d.final_nf, 2, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        min_side = min(h, w)
        eff_steps = int_steps
        if int_downsize > 1 and min_side >= 32:
            self.resize = ResizeTransform(int_downsize, 2)
            self.fullsize = ResizeTransform(1 / int_downsize, 2)
            down_shape = [int(h / int_downsize), int(w / int_downsize)]
        else:
            self.resize = None
            self.fullsize = None
            down_shape = [h, w]
            if min_side < 32:
                eff_steps = min(int_steps, 2)

        self.integrate = VecInt(down_shape, eff_steps) if eff_steps > 0 else None
        self.int_downsize = int_downsize if self.resize is not None else 1

    def _stack_to_volume(self, stack):
        return stack.unsqueeze(1)

    def _raw_to_band_flows(self, raw):
        return raw.permute(0, 2, 1, 3, 4).contiguous()

    def _process_band_flows(self, flow_vol):
        b, n, _, h, w = flow_vol.shape
        out_slices = []
        preint_slices = []
        for i in range(n):
            pos = flow_vol[:, i, :, :, :]
            if self.resize is not None:
                pos = self.resize(pos)
            pre = pos
            if self.integrate is not None:
                pos = self.integrate(pos)
            if self.fullsize is not None:
                pos = self.fullsize(pos)
            out_slices.append(pos)
            preint_slices.append(pre)
        return torch.stack(out_slices, dim=1), torch.stack(preint_slices, dim=1)

    def predict_flow_volume(self, stack, registration=False):
        vol = self._stack_to_volume(stack)
        feat = self.unet3d(vol)
        raw = self.flow(feat)
        band_raw = self._raw_to_band_flows(raw)
        pos, preint = self._process_band_flows(band_raw)
        if registration:
            return pos, preint
        sh, sw = pos.shape[-2], pos.shape[-1]
        warped = _warp_stack_all_flows(stack, pos, (sh, sw))
        return warped, preint


def _train_window_at_level(
    window_bands: List[np.ndarray],
    device,
    max_epochs,
    patience,
    lr,
    lamda,
    lamda_spec,
    lamda_gauge,
    lamda_var,
    ncc_weight,
    int_steps,
    int_downsize,
    nb_unet_features,
    early_stop,
    min_delta,
    lr_schedule,
    lr_min,
    verbose,
):
    h, w = window_bands[0].shape
    n_win = len(window_bands)
    stack_t = _bands_to_tensor(window_bands, device)
    int_steps, int_downsize = _level_train_params(min(h, w), int_steps, int_downsize)

    model = WindowBalancedFlowNet3d(
        num_bands=n_win,
        spatial_shape=(h, w),
        nb_unet_features=nb_unet_features,
        int_steps=int_steps,
        int_downsize=int_downsize,
    ).to(device)

    grad_fn = Grad('l2', loss_mult=model.int_downsize).loss
    ncc_fn = NCC().loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler, schedule_type = _create_lr_scheduler(optimizer, lr_schedule, max_epochs, lr_min=lr_min)

    best_loss = float('inf')
    best_flow_vol = None
    best_state = None
    stale_epochs = 0
    log_every = max(max_epochs // 10, 1)

    for epoch in range(max_epochs):
        model.train()
        warped, preint = model.predict_flow_volume(stack_t, registration=False)
        loss = (
            ncc_weight * sequential_pairwise_ncc_loss(warped, ncc_fn)
            + lamda * _volume_grad_loss(preint, grad_fn)
            + lamda_spec * spectral_flow_smoothness_loss(preint)
            + lamda_gauge * gauge_mean_flow_loss(preint)
            + lamda_var * spectral_variance_loss(warped)
        )
        current_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if schedule_type == 'plateau':
            scheduler.step(current_loss)
        elif scheduler is not None:
            scheduler.step()

        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            stale_epochs = 0
            model.eval()
            with torch.no_grad():
                best_flow_vol, _ = model.predict_flow_volume(stack_t, registration=True)
                best_state = copy.deepcopy(model.state_dict())
            model.train()
        else:
            stale_epochs += 1

        if verbose and (epoch % log_every == 0 or epoch == max_epochs - 1):
            lr_now = optimizer.param_groups[0]['lr']
            print(
                f'      epoch {epoch + 1}/{max_epochs}: loss={current_loss:.4f} '
                f'best={best_loss:.4f} lr={lr_now:.2e}'
            )

        if early_stop and stale_epochs >= patience:
            if verbose:
                print(f'      early stop @ epoch {epoch + 1} (best={best_loss:.4f})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        flow_vol = best_flow_vol
    else:
        model.eval()
        with torch.no_grad():
            flow_vol, _ = model.predict_flow_volume(stack_t, registration=True)

    return flow_vol


def _iter_window_starts(num_bands: int, window_size: int, window_stride: int) -> List[int]:
    if num_bands <= window_size:
        return [0]
    last_start = num_bands - window_size
    return list(range(0, last_start + 1, window_stride))


def _apply_window_flows_to_working(
    working: List[np.ndarray],
    start: int,
    flow_vol: torch.Tensor,
    flow_global: List[np.ndarray],
    device,
    full_h: int,
    full_w: int,
):
    """将窗口 W 个 flow 立即作用于 working，并累积到 flow_global。"""
    w = flow_vol.shape[1]
    for k in range(w):
        gi = start + k
        flow_level = flow_vol[:, k, :, :, :]
        flow_full = _upsample_flow(flow_level, full_h, full_w)
        flow_np = flow_full.squeeze(0).detach().cpu().numpy().astype(np.float32)
        working[gi] = _warp_band_numpy(working[gi], flow_full.squeeze(0), device)
        if np.allclose(flow_global[gi], 0.0):
            flow_global[gi] = flow_np.copy()
        else:
            flow_global[gi] = _compose_flow_numpy(flow_global[gi], flow_np, device)


def _run_pyramid_then_windows(
    working: List[np.ndarray],
    flow_global: List[np.ndarray],
    device,
    n: int,
    h: int,
    w: int,
    window_size: int,
    window_stride: int,
    window_starts: List[int],
    levels: List[Tuple[int, int]],
    ep_levels: List[int],
    pat_levels: List[int],
    lr,
    lamda,
    lamda_spec,
    lamda_gauge,
    lamda_var,
    ncc_weight,
    int_steps,
    int_downsize,
    nb_unet_features,
    early_stop,
    min_delta,
    lr_schedule,
    lr_min,
    verbose,
):
    """先空间：每层金字塔内顺序滑动窗口，每窗口结束立即 warp。"""
    for li, (sh, sw) in enumerate(levels):
        ep = ep_levels[li]
        pat = pat_levels[li]
        if verbose:
            print(
                f'Level {li + 1}/{len(levels)}: {sh}x{sw}, '
                f'{len(window_starts)} windows (sequential warp), '
                f'max_epochs/window={ep}, patience={pat}'
            )

        for wi, start in enumerate(window_starts):
            end = start + window_size
            window_bands = [
                _downsample_band(working[start + k], sw, sh) for k in range(window_size)
            ]
            if verbose:
                print(f'  Window {wi + 1}/{len(window_starts)}: bands [{start}:{end})')

            flow_vol = _train_window_at_level(
                window_bands, device, ep, pat, lr, lamda, lamda_spec, lamda_gauge, lamda_var,
                ncc_weight, int_steps, int_downsize, nb_unet_features,
                early_stop, min_delta, lr_schedule, lr_min, verbose,
            )
            _apply_window_flows_to_working(
                working, start, flow_vol, flow_global, device, h, w,
            )


def _run_window_then_pyramid(
    working: List[np.ndarray],
    flow_global: List[np.ndarray],
    device,
    n: int,
    h: int,
    w: int,
    window_size: int,
    window_stride: int,
    window_starts: List[int],
    levels: List[Tuple[int, int]],
    ep_levels: List[int],
    pat_levels: List[int],
    lr,
    lamda,
    lamda_spec,
    lamda_gauge,
    lamda_var,
    ncc_weight,
    int_steps,
    int_downsize,
    nb_unet_features,
    early_stop,
    min_delta,
    lr_schedule,
    lr_min,
    verbose,
):
    """先光谱：每个窗口位置内走完金字塔，再滑动到下一窗口。"""
    for wi, start in enumerate(window_starts):
        end = start + window_size
        if verbose:
            print(
                f'Window {wi + 1}/{len(window_starts)}: bands [{start}:{end}), '
                f'pyramid {[f"{a}x{b}" for a, b in levels]}'
            )

        window_flow_cum: Optional[torch.Tensor] = None

        for li, (sh, sw) in enumerate(levels):
            ep = ep_levels[li]
            pat = pat_levels[li]
            window_bands = [
                _downsample_band(working[start + k], sw, sh) for k in range(window_size)
            ]

            if window_flow_cum is not None:
                flow_on_level = _upsample_flow_volume(window_flow_cum, sh, sw)
                stack_t = _bands_to_tensor(window_bands, device)
                stack_t = _warp_stack_all_flows(stack_t, flow_on_level, (sh, sw))
                window_bands = _tensor_to_bands(stack_t)

            if verbose:
                print(
                    f'  Level {li + 1}/{len(levels)}: {sh}x{sw}, '
                    f'max_epochs={ep}, patience={pat}'
                )

            flow_delta = _train_window_at_level(
                window_bands, device, ep, pat, lr, lamda, lamda_spec, lamda_gauge, lamda_var,
                ncc_weight, int_steps, int_downsize, nb_unet_features,
                early_stop, min_delta, lr_schedule, lr_min, verbose,
            )

            if window_flow_cum is None:
                window_flow_cum = flow_delta
            else:
                flow_prev = _upsample_flow_volume(window_flow_cum, sh, sw)
                window_flow_cum = _compose_flow_volumes(
                    flow_prev, flow_delta, (sh, sw), device,
                )

        if window_flow_cum is not None:
            flow_full = _upsample_flow_volume(window_flow_cum, h, w)
            _apply_window_flows_to_working(
                working, start, flow_full, flow_global, device, h, w,
            )


def register_pifreg_groupwise_sliding_window(
    img_list: StackInput,
    device: str = 'cuda',
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
    schedule: str = SCHEDULE_PYRAMID_THEN_WINDOWS,
    pyramid_sizes: Tuple[int, ...] = DEFAULT_PYRAMID_SIZES,
    epochs_per_window: Optional[Sequence[int]] = None,
    patience_per_window: Optional[Sequence[int]] = None,
    lr: float = 2e-4,
    lamda: float = 0.005,
    lamda_spec: float = DEFAULT_LAMDA_SPEC,
    lamda_gauge: float = DEFAULT_LAMDA_GAUGE,
    lamda_var: float = DEFAULT_LAMDA_VAR,
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
    **kwargs,
) -> Tuple[List[np.ndarray], Dict[str, Any], np.ndarray]:
    """
    滑动窗口均衡配准（无锚点，每 band 一个 flow）。

    schedule:
        pyramid_then_windows — 每层扫完所有窗口（默认）
        window_then_pyramid  — 每窗口走完整个金字塔
    """
    if kwargs:
        ignored = ', '.join(sorted(kwargs))
        if verbose:
            print(f'Note: ignoring deprecated/unused kwargs: {ignored}')

    device = _resolve_device(device)
    bands = _as_band_list(img_list)
    n = len(bands)
    if n <= 1:
        return bands, {'mode': 'sliding_window', 'num_bands': n}, np.zeros((0, 2, 0, 0), dtype=np.float32)

    schedule = (schedule or SCHEDULE_PYRAMID_THEN_WINDOWS).lower()
    if schedule not in (SCHEDULE_PYRAMID_THEN_WINDOWS, SCHEDULE_WINDOW_THEN_PYRAMID):
        raise ValueError(
            f'Unknown schedule={schedule!r}, expected '
            f'{SCHEDULE_PYRAMID_THEN_WINDOWS!r} or {SCHEDULE_WINDOW_THEN_PYRAMID!r}'
        )

    window_size = max(2, min(int(window_size), n))
    window_stride = max(1, int(window_stride))

    if fast_mode:
        nb_unet_features = nb_unet_features or compact_unet_features()
        lr = 2e-4
        lamda = 0.005

    h, w = bands[0].shape
    levels = _build_pyramid_levels(h, w, pyramid_sizes)
    window_starts = _iter_window_starts(n, window_size, window_stride)

    ep_levels = list(epochs_per_window or DEFAULT_EPOCHS_PER_WINDOW)
    pat_levels = list(patience_per_window or DEFAULT_PATIENCE_PER_WINDOW)
    while len(ep_levels) < len(levels):
        ep_levels.append(ep_levels[-1])
    while len(pat_levels) < len(levels):
        pat_levels.append(pat_levels[-1])

    if verbose:
        print(
            f'{METHOD_NAME}: {n} bands, {n} flows (no anchor), '
            f'window={window_size}, stride={window_stride}, schedule={schedule}, '
            f'network=WindowBalancedFlowNet3d, pyramid={[f"{a}x{b}" for a, b in levels]}'
        )

    working = [b.copy() for b in bands]
    flow_global = [np.zeros((2, h, w), dtype=np.float32) for _ in range(n)]

    runner_kwargs = dict(
        working=working,
        flow_global=flow_global,
        device=device,
        n=n,
        h=h,
        w=w,
        window_size=window_size,
        window_stride=window_stride,
        window_starts=window_starts,
        levels=levels,
        ep_levels=ep_levels,
        pat_levels=pat_levels,
        lr=lr,
        lamda=lamda,
        lamda_spec=lamda_spec,
        lamda_gauge=lamda_gauge,
        lamda_var=lamda_var,
        ncc_weight=ncc_weight,
        int_steps=int_steps,
        int_downsize=int_downsize,
        nb_unet_features=nb_unet_features,
        early_stop=early_stop,
        min_delta=min_delta,
        lr_schedule=lr_schedule,
        lr_min=lr_min,
        verbose=verbose,
    )

    if schedule == SCHEDULE_PYRAMID_THEN_WINDOWS:
        _run_pyramid_then_windows(**runner_kwargs)
    else:
        _run_window_then_pyramid(**runner_kwargs)

    flow_np = np.stack(flow_global, axis=0).astype(np.float32)
    registered = working

    unet_feats = nb_unet_features or compact_unet_features()
    info = {
        'mode': 'sliding_window',
        'method': METHOD_FULL_NAME,
        'num_bands': n,
        'num_flow_fields': n,
        'anchor_band_idx': None,
        'moving_band_indices': list(range(n)),
        'flow_stack_shape': list(flow_np.shape),
        'flow_representation': 'per_band_volume (N,2,H,W), no anchor',
        'window_size': window_size,
        'window_stride': window_stride,
        'schedule': schedule,
        'windows_per_pass': len(window_starts),
        'pyramid_sizes': list(pyramid_sizes),
        'pyramid_levels': [list(lv) for lv in levels],
        'epochs_per_window': ep_levels[: len(levels)],
        'patience_per_window': pat_levels[: len(levels)],
        'lr': lr,
        'lamda': lamda,
        'lamda_spec': lamda_spec,
        'lamda_gauge': lamda_gauge,
        'lamda_var': lamda_var,
        'ncc_weight': ncc_weight,
        'int_steps': int_steps,
        'int_downsize': int_downsize,
        'nb_unet_features': [list(unet_feats[0]), list(unet_feats[1])],
        'early_stop': early_stop,
        'min_delta': min_delta,
        'lr_schedule': lr_schedule,
        'lr_min': lr_min,
        'fast_mode': fast_mode,
        'network': 'WindowBalancedFlowNet3d (SpectralStackUnet3d)',
        'warp_mode': 'sequential_in_place (no flow averaging)',
        'loss': (
            'adjacent_ncc + spatial_grad + spectral_flow_smooth '
            '+ gauge_mean_flow + stack_variance'
        ),
        'device': str(device),
    }
    return registered, info, flow_np


# 兼容旧 import
warp_bands_with_flow_stack = warp_bands_with_flow_volume

__all__ = [
    'METHOD_NAME',
    'METHOD_FULL_NAME',
    'DEFAULT_WINDOW_SIZE',
    'DEFAULT_WINDOW_STRIDE',
    'DEFAULT_EPOCHS_PER_WINDOW',
    'DEFAULT_PATIENCE_PER_WINDOW',
    'DEFAULT_LAMDA_SPEC',
    'DEFAULT_LAMDA_GAUGE',
    'DEFAULT_LAMDA_VAR',
    'SCHEDULE_PYRAMID_THEN_WINDOWS',
    'SCHEDULE_WINDOW_THEN_PYRAMID',
    'register_pifreg_groupwise_sliding_window',
    'warp_bands_with_flow_volume',
    'warp_bands_with_flow_stack',
]
