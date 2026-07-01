# PIFReg Sliding-Window Groupwise — 滑动窗口内联合优化相邻波段位移场
#
# 拓扑约束：仅相邻波段 NCC（与 chain 一致），不跨波段配准。
# 每个窗口含 W 个波段，窗口内联合预测 W-1 个 flow；窗口滑动后重叠区 flow 取平均。
# 相对全栈 StackFlow：U-Net 输入为 W 通道子栈（mean+anchor），无 30→2 压缩瓶颈。

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..losses.registration_losses import Grad, NCC
from ..voxelmorph.config import compact_unet_features
from .pif_groupwise_stackflow import (
    DEFAULT_PYRAMID_SIZES,
    FEATURE_MODE_MEAN_ANCHOR,
    FEATURE_MODE_SPECTRAL_ENCODER,
    DEFAULT_SPECTRAL_ENC_CHANNELS,
    DEFAULT_SPECTRAL_ENC_KERNEL,
    PerBandStackFlowNet,
    StackInput,
    _as_band_list,
    _bands_to_tensor,
    _build_pyramid_levels,
    _compose_flow_stacks,
    _create_lr_scheduler,
    _downsample_stack,
    _flow_index_for_band,
    _moving_band_indices,
    _resolve_device,
    _tensor_to_bands,
    _upsample_flow_stack,
    _warp_stack_perband,
    flow_stack_to_numpy,
    sequential_pairwise_ncc_loss,
    stack_grad_loss,
    warp_bands_with_flow_stack,
)

METHOD_NAME = 'PIFReg-SlidingWindow'
METHOD_FULL_NAME = 'PIFReg Sliding-Window Adjacent Joint Registration'

DEFAULT_WINDOW_SIZE = 5
DEFAULT_WINDOW_STRIDE = 1
DEFAULT_EPOCHS_PER_WINDOW = (300, 500, 800)
DEFAULT_PATIENCE_PER_WINDOW = (50, 70, 90)
DEFAULT_LAMDA_SPEC = 0.02


def spectral_flow_smoothness_loss(flow_stack: torch.Tensor) -> torch.Tensor:
    """相邻 flow 在光谱维的 L2 平滑：mean(||φ_i - φ_{i+1}||²)。"""
    m = flow_stack.shape[1]
    if m <= 1:
        return flow_stack.sum() * 0.0
    total = sum(
        ((flow_stack[:, i] - flow_stack[:, i + 1]) ** 2).mean()
        for i in range(m - 1)
    )
    return total / (m - 1)


def _level_train_params(level_side: int, base_int_steps=3, base_int_downsize=2):
    if level_side <= 32:
        return min(base_int_steps, 2), 1
    return base_int_steps, base_int_downsize


def _train_window_at_level(
    window_bands: List[np.ndarray],
    device,
    window_anchor_idx: int,
    max_epochs,
    patience,
    lr,
    lamda,
    lamda_spec,
    ncc_weight,
    int_steps,
    int_downsize,
    nb_unet_features,
    early_stop,
    min_delta,
    lr_schedule,
    lr_min,
    verbose,
    feature_mode=FEATURE_MODE_MEAN_ANCHOR,
    spectral_enc_channels=DEFAULT_SPECTRAL_ENC_CHANNELS,
    spectral_enc_kernel=DEFAULT_SPECTRAL_ENC_KERNEL,
):
    """在单个窗口子栈上联合优化 W-1 个 flow。"""
    h, w = window_bands[0].shape
    n_win = len(window_bands)
    num_moving = n_win - 1
    stack_t = _bands_to_tensor(window_bands, device)

    int_steps, int_downsize = _level_train_params(min(h, w), int_steps, int_downsize)

    model = PerBandStackFlowNet(
        inshape=(h, w),
        num_bands=n_win,
        num_moving_bands=num_moving,
        anchor_idx=window_anchor_idx,
        nb_unet_features=nb_unet_features,
        int_steps=int_steps,
        int_downsize=int_downsize,
        feature_mode=feature_mode,
        spectral_enc_channels=spectral_enc_channels,
        spectral_enc_kernel=spectral_enc_kernel,
    ).to(device)

    grad_fn = Grad('l2', loss_mult=int_downsize).loss
    ncc_fn = NCC().loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler, schedule_type = _create_lr_scheduler(optimizer, lr_schedule, max_epochs, lr_min=lr_min)

    best_loss = float('inf')
    best_flow_stack = None
    best_state = None
    stale_epochs = 0
    log_every = max(max_epochs // 10, 1)

    for epoch in range(max_epochs):
        model.train()
        warped, preint = model.predict_flow_stack(stack_t, registration=False)
        loss = (
            ncc_weight * sequential_pairwise_ncc_loss(warped, ncc_fn)
            + lamda * stack_grad_loss(preint, grad_fn)
            + lamda_spec * spectral_flow_smoothness_loss(preint)
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
                best_flow_stack, _ = model.predict_flow_stack(stack_t, registration=True)
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
        flow_stack = best_flow_stack
    else:
        model.eval()
        with torch.no_grad():
            flow_stack, _ = model.predict_flow_stack(stack_t, registration=True)

    return flow_stack


def _merge_window_flows(
    flow_sum: torch.Tensor,
    flow_count: torch.Tensor,
    window_flow: torch.Tensor,
    window_start: int,
    global_anchor_idx: int,
):
    """
    将窗口 flow (1, W-1, 2, h, w) 累加到全局 flow_sum (1, N-1, 2, h, w)。
    窗口锚点为窗口内最后一 band；moving bands 为 window_start .. window_start+W-2。
    """
    w_moving = window_flow.shape[1]
    for local_i in range(w_moving):
        global_band = window_start + local_i
        if global_band == global_anchor_idx:
            continue
        fi = _flow_index_for_band(global_band, global_anchor_idx)
        flow_sum[:, fi : fi + 1] += window_flow[:, local_i : local_i + 1]
        flow_count[fi] += 1


def _average_merged_flows(flow_sum: torch.Tensor, flow_count: torch.Tensor) -> torch.Tensor:
    """重叠窗口 flow 加权平均；未被任何窗口覆盖的通道保持为零。"""
    count = flow_count.view(1, -1, 1, 1, 1).clamp(min=1.0)
    return flow_sum / count


def _iter_window_starts(num_bands: int, window_size: int, window_stride: int) -> List[int]:
    if num_bands <= window_size:
        return [0]
    last_start = num_bands - window_size
    return list(range(0, last_start + 1, window_stride))


def register_pifreg_groupwise_sliding_window(
    img_list: StackInput,
    device: str = 'cuda',
    anchor_band_idx: int = -1,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
    pyramid_sizes: Tuple[int, ...] = DEFAULT_PYRAMID_SIZES,
    epochs_per_window: Optional[Sequence[int]] = None,
    patience_per_window: Optional[Sequence[int]] = None,
    lr: float = 2e-4,
    lamda: float = 0.005,
    lamda_spec: float = DEFAULT_LAMDA_SPEC,
    ncc_weight: float = 1.0,
    int_steps: int = 3,
    int_downsize: int = 2,
    nb_unet_features=None,
    early_stop: bool = True,
    min_delta: float = 1e-5,
    lr_schedule: str = 'cosine',
    lr_min: float = 1e-6,
    fast_mode: bool = True,
    feature_mode: str = FEATURE_MODE_MEAN_ANCHOR,
    spectral_enc_channels: int = DEFAULT_SPECTRAL_ENC_CHANNELS,
    spectral_enc_kernel: int = DEFAULT_SPECTRAL_ENC_KERNEL,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any], np.ndarray]:
    """
    滑动窗口相邻联合配准：仅窗口内相邻 NCC + flow 光谱平滑，重叠区 flow 平均。

    参数:
        anchor_band_idx: 全局锚点（默认 -1 = 最高波长 band，不动）
        window_size: 每窗口波段数 W（默认 5 → 4 个 flow / 窗口）
        window_stride: 滑动步长（默认 1，最大重叠）
        epochs_per_window / patience_per_window: 各金字塔层、每窗口的训练预算
    """
    device = _resolve_device(device)
    bands = _as_band_list(img_list)
    n = len(bands)
    if n <= 1:
        return bands, {'mode': 'sliding_window', 'num_bands': n}, np.zeros((0, 2, 0, 0), dtype=np.float32)

    feature_mode = (feature_mode or FEATURE_MODE_MEAN_ANCHOR).lower()
    if feature_mode not in (FEATURE_MODE_MEAN_ANCHOR, FEATURE_MODE_SPECTRAL_ENCODER):
        raise ValueError(
            f'Unknown feature_mode={feature_mode!r}, '
            f'expected {FEATURE_MODE_MEAN_ANCHOR!r} or {FEATURE_MODE_SPECTRAL_ENCODER!r}'
        )

    window_size = max(2, min(int(window_size), n))
    window_stride = max(1, int(window_stride))

    if fast_mode:
        nb_unet_features = nb_unet_features or compact_unet_features()
        lr = 2e-4
        lamda = 0.005

    h, w = bands[0].shape
    anchor_band_idx = int(anchor_band_idx) % n
    levels = _build_pyramid_levels(h, w, pyramid_sizes)
    window_starts = _iter_window_starts(n, window_size, window_stride)
    num_moving_global = n - 1

    ep_levels = list(epochs_per_window or DEFAULT_EPOCHS_PER_WINDOW)
    pat_levels = list(patience_per_window or DEFAULT_PATIENCE_PER_WINDOW)
    while len(ep_levels) < len(levels):
        ep_levels.append(ep_levels[-1])
    while len(pat_levels) < len(levels):
        pat_levels.append(pat_levels[-1])

    if verbose:
        feat_desc = (
            f'spectral_enc(K={spectral_enc_channels})+anchor'
            if feature_mode == FEATURE_MODE_SPECTRAL_ENCODER
            else 'mean+anchor'
        )
        print(
            f'{METHOD_NAME}: {n} bands, anchor={anchor_band_idx}, '
            f'window={window_size}, stride={window_stride}, '
            f'windows/level={len(window_starts)}, features={feat_desc}, '
            f'pyramid={[f"{a}x{b}" for a, b in levels]}'
        )

    working = [b.copy() for b in bands]
    flow_stack_full = None
    window_logs: List[Dict[str, Any]] = []

    for li, (sh, sw) in enumerate(levels):
        bands_s = _downsample_stack(working, sw, sh)

        if flow_stack_full is not None:
            flow_on_level = _upsample_flow_stack(flow_stack_full, sh, sw)
            stack_t = _bands_to_tensor(bands_s, device)
            stack_t = _warp_stack_perband(stack_t, flow_on_level, anchor_band_idx, (sh, sw))
            bands_s = _tensor_to_bands(stack_t)

        ep = ep_levels[li]
        pat = pat_levels[li]
        if verbose:
            print(
                f'Level {li + 1}/{len(levels)}: {sh}x{sw}, '
                f'{len(window_starts)} windows, max_epochs/window={ep}, patience={pat}'
            )

        flow_sum = torch.zeros(1, num_moving_global, 2, sh, sw, device=device)
        flow_count = torch.zeros(num_moving_global, device=device)

        for wi, start in enumerate(window_starts):
            end = start + window_size
            window_bands = bands_s[start:end]
            win_anchor = window_size - 1

            if verbose:
                print(
                    f'  Window {wi + 1}/{len(window_starts)}: '
                    f'bands [{start}:{end}), local anchor={start + win_anchor}'
                )

            window_flow = _train_window_at_level(
                window_bands,
                device,
                win_anchor,
                ep,
                pat,
                lr,
                lamda,
                lamda_spec,
                ncc_weight,
                int_steps,
                int_downsize,
                nb_unet_features,
                early_stop,
                min_delta,
                lr_schedule,
                lr_min,
                verbose=verbose,
                feature_mode=feature_mode,
                spectral_enc_channels=spectral_enc_channels,
                spectral_enc_kernel=spectral_enc_kernel,
            )
            _merge_window_flows(flow_sum, flow_count, window_flow, start, anchor_band_idx)
            window_logs.append({
                'level': li,
                'window_index': wi,
                'band_start': start,
                'band_end': end,
            })

        flow_delta = _average_merged_flows(flow_sum, flow_count)

        if flow_stack_full is None:
            flow_stack_full = flow_delta
        else:
            flow_prev = _upsample_flow_stack(flow_stack_full, sh, sw)
            flow_stack_full = _compose_flow_stacks(flow_prev, flow_delta, (sh, sw), device)

    flow_stack_full = _upsample_flow_stack(flow_stack_full, h, w)
    stack_orig = _bands_to_tensor(working, device)
    warped_t = _warp_stack_perband(stack_orig, flow_stack_full, anchor_band_idx, (h, w))
    registered = _tensor_to_bands(warped_t)

    unet_feats = nb_unet_features or compact_unet_features()
    flow_np = flow_stack_to_numpy(flow_stack_full)
    info = {
        'mode': 'sliding_window',
        'method': METHOD_FULL_NAME,
        'num_bands': n,
        'num_flow_fields': num_moving_global,
        'anchor_band_idx': anchor_band_idx,
        'moving_band_indices': _moving_band_indices(n, anchor_band_idx),
        'flow_stack_shape': list(flow_np.shape),
        'window_size': window_size,
        'window_stride': window_stride,
        'windows_per_level': len(window_starts),
        'pyramid_sizes': list(pyramid_sizes),
        'pyramid_levels': [list(lv) for lv in levels],
        'epochs_per_window': ep_levels[: len(levels)],
        'patience_per_window': pat_levels[: len(levels)],
        'lr': lr,
        'lamda': lamda,
        'lamda_spec': lamda_spec,
        'ncc_weight': ncc_weight,
        'int_steps': int_steps,
        'int_downsize': int_downsize,
        'nb_unet_features': [list(unet_feats[0]), list(unet_feats[1])],
        'early_stop': early_stop,
        'min_delta': min_delta,
        'lr_schedule': lr_schedule,
        'lr_min': lr_min,
        'fast_mode': fast_mode,
        'feature_mode': feature_mode,
        'spectral_enc_channels': spectral_enc_channels,
        'spectral_enc_kernel': spectral_enc_kernel,
        'loss': 'window_adjacent_ncc + per_flow_grad + spectral_flow_smooth',
        'device': str(device),
        'window_logs_count': len(window_logs),
    }
    return registered, info, flow_np


__all__ = [
    'METHOD_NAME',
    'METHOD_FULL_NAME',
    'DEFAULT_WINDOW_SIZE',
    'DEFAULT_WINDOW_STRIDE',
    'DEFAULT_EPOCHS_PER_WINDOW',
    'DEFAULT_PATIENCE_PER_WINDOW',
    'DEFAULT_LAMDA_SPEC',
    'register_pifreg_groupwise_sliding_window',
    'warp_bands_with_flow_stack',
]
