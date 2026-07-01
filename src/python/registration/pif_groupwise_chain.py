# PIFReg Chain Groupwise — 高波长→低波长链式两两配准
#
# 配准顺序：690 → 680 → 670 → …（波长降序）
#   - 栈内最高波长波段为锚点（不动）
#   - 下一波段配准到「上一波段配准后的结果」
#   - 每对使用 register_pifreg，优化目标为 NCC
#
# schedule:
#   pair_then_pyramid   — 默认：每对内部 128→256→512，再下一对
#   pyramid_then_pairs  — 每层分辨率下扫完整条链，再进下一层

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..metrics.evaluation import compute_NCC
from .pif_groupwise_stackflow import DEFAULT_PYRAMID_SIZES, _build_pyramid_levels, _downsample_band
from .pif_registration import (
    _apply_flow_to_image,
    _compose_flows,
    _flow_tensor_to_numpy,
    _resolve_device,
    _upsample_flow,
    register_pifreg,
)

METHOD_CHAIN_NAME = 'PIFReg-Chain'
METHOD_CHAIN_FULL_NAME = 'PIFReg Descending Wavelength Chain Registration'

SCHEDULE_PAIR_THEN_PYRAMID = 'pair_then_pyramid'
SCHEDULE_PYRAMID_THEN_PAIRS = 'pyramid_then_pairs'

StackInput = Union[Sequence[np.ndarray], np.ndarray]


def _as_band_list(stack: StackInput) -> List[np.ndarray]:
    if isinstance(stack, list):
        return [np.asarray(b, dtype=np.float32) for b in stack]
    arr = np.asarray(stack)
    if arr.ndim == 2:
        return [arr.astype(np.float32)]
    if arr.ndim == 3:
        return [arr[i].astype(np.float32) for i in range(arr.shape[0])]
    raise ValueError(f'Expected stack shape (N,H,W) or list of (H,W), got {arr.shape}')


def _default_pifreg_kwargs() -> Dict[str, Any]:
    return dict(
        fast_mode=True,
        image_loss='ncc',
        early_stop=True,
        patience=80,
        epochs=3000,
        lr_schedule='cosine',
        multiscale=True,
        affine_init=False,
        histogram_match=True,
    )


def _chain_steps(n: int, descending: bool) -> List[Tuple[int, int]]:
    """descending: 690→680→…→400；ascending: 400→410→…→690。"""
    if descending:
        return [(i + 1, i) for i in range(n - 2, -1, -1)]
    return [(i - 1, i) for i in range(1, n)]


def _resolve_level_directions(
    num_levels: int,
    descending: bool,
    level_directions: Optional[Sequence[bool]] = None,
    alternate_direction: bool = False,
) -> List[bool]:
    """每层链扫方向；alternate 时奇偶层交替（默认 L0 与 descending 一致）。"""
    if level_directions is not None:
        dirs = [bool(d) for d in level_directions]
        while len(dirs) < num_levels:
            dirs.append(dirs[-1])
        return dirs[:num_levels]
    if alternate_direction:
        return [descending if (li % 2 == 0) else (not descending) for li in range(num_levels)]
    return [descending] * num_levels


def _parse_level_directions_arg(value: Optional[str]) -> Optional[List[bool]]:
    """解析 'desc,asc,desc' 或 '1,0,1'。"""
    if value is None:
        return None
    tokens = [t.strip().lower() for t in value.split(',') if t.strip()]
    out: List[bool] = []
    for t in tokens:
        if t in ('desc', 'descending', 'down', '690', '1', 'true'):
            out.append(True)
        elif t in ('asc', 'ascending', 'up', '400', '0', 'false'):
            out.append(False)
        else:
            raise ValueError(
                f'Unknown level direction token {t!r}; use desc/asc or 1/0'
            )
    return out


def _single_scale_pifreg_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """外层金字塔时，每对仅在当前层单尺度优化。"""
    out = dict(kwargs)
    out['multiscale'] = False
    return out


def _upsample_flow_numpy(flow_np: np.ndarray, target_h: int, target_w: int, device) -> np.ndarray:
    flow_t = torch.tensor(flow_np, dtype=torch.float32, device=device).unsqueeze(0)
    up = _upsample_flow(flow_t, target_h, target_w)
    return up.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _compose_flow_numpy(base_np: np.ndarray, delta_np: np.ndarray, device) -> np.ndarray:
    h, w = base_np.shape[1], base_np.shape[2]
    base_t = torch.tensor(base_np, dtype=torch.float32, device=device).unsqueeze(0)
    delta_t = torch.tensor(delta_np, dtype=torch.float32, device=device).unsqueeze(0)
    out = _compose_flows(base_t, delta_t, (h, w), device)
    return out.squeeze(0).detach().cpu().numpy().astype(np.float32)


def evaluate_chain_pairwise_ncc(
    registered: List[np.ndarray],
    wavelengths_nm: Optional[Sequence[str]] = None,
    descending: bool = True,
) -> Dict[str, Any]:
    """
    链式相邻波段 NCC（降序：NCC(warped[i], warped[i+1])，i 从 n-2 到 0，即 680↔690, 670↔680…）
    返回每对 NCC 及均值。
    """
    n = len(registered)
    pairs = []
    for i in range(n - 2, -1, -1):
        ref = registered[i + 1]
        mov = registered[i]
        wl_ref = wavelengths_nm[i + 1] if wavelengths_nm else str(i + 1)
        wl_mov = wavelengths_nm[i] if wavelengths_nm else str(i)
        pairs.append({
            'index_ref': i + 1,
            'index_mov': i,
            'wavelength_ref_nm': wl_ref,
            'wavelength_mov_nm': wl_mov,
            'NCC': _pairwise_ncc(ref, mov),
        })
    ncc_vals = [p['NCC'] for p in pairs]
    return {
        'direction': 'descending_wavelength' if descending else 'ascending',
        'num_pairs': len(pairs),
        'pairs': pairs,
        'mean_NCC': float(np.mean(ncc_vals)) if ncc_vals else 1.0,
    }


def _register_chain_pyramid_first(
    bands: List[np.ndarray],
    device,
    wavelengths_nm: List[str],
    kwargs: Dict[str, Any],
    pyramid_sizes: Tuple[int, ...],
    descending: bool,
    level_directions: Optional[Sequence[bool]],
    alternate_direction: bool,
    verbose: bool,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[int, np.ndarray]]:
    """先金字塔层级、每层内完整链式扫描（单尺度 PIFReg / 对）。"""
    device_obj = _resolve_device(device)

    n = len(bands)
    h, w = bands[0].shape
    levels = _build_pyramid_levels(h, w, pyramid_sizes)
    per_level_desc = _resolve_level_directions(
        len(levels), descending, level_directions, alternate_direction,
    )
    pair_kwargs = _single_scale_pifreg_kwargs(kwargs)

    registered = [b.copy() for b in bands]
    flows_by_band: Dict[int, np.ndarray] = {}
    step_logs: List[Dict[str, Any]] = []
    global_step = 0

    if verbose:
        dir_labels = [
            '690→400' if d else '400→690' for d in per_level_desc
        ]
        print(
            f'{METHOD_CHAIN_NAME}: pyramid_then_pairs, '
            f'levels={[f"{a}x{b}" for a, b in levels]}, '
            f'directions={dir_labels}'
        )

    for li, (sh, sw) in enumerate(levels):
        level_desc = per_level_desc[li]
        chain_steps = _chain_steps(n, level_desc)
        dir_label = '690→400' if level_desc else '400→690'
        if verbose:
            print(
                f'Level {li + 1}/{len(levels)}: {sh}x{sw}, '
                f'chain {dir_label} ({len(chain_steps)} pairs)'
            )

        for step, (ref_idx, mov_idx) in enumerate(chain_steps, start=1):
            global_step += 1
            fixed = _downsample_band(registered[ref_idx], sw, sh)
            moving = _downsample_band(bands[mov_idx], sw, sh)

            prev_flow = flows_by_band.get(mov_idx)
            if prev_flow is not None:
                flow_on_level = _upsample_flow_numpy(prev_flow, sh, sw, device_obj)
                flow_t = torch.tensor(flow_on_level, dtype=torch.float32, device=device_obj).unsqueeze(0)
                moving = _apply_flow_to_image(moving, flow_t, device_obj)

            wl_ref, wl_mov = wavelengths_nm[ref_idx], wavelengths_nm[mov_idx]
            ncc_before = _pairwise_ncc(fixed, moving)

            if verbose:
                print(
                    f'  L{li + 1} [{step}/{len(chain_steps)}] '
                    f'PIFReg @ {sh}x{sw}: fixed={wl_ref} nm <- moving={wl_mov} nm '
                    f'(NCC before={ncc_before:.4f})'
                )

            warped, flow_delta = register_pifreg(
                fixed, moving, device=str(device_obj), return_flow=True, **pair_kwargs,
            )
            if torch.is_tensor(flow_delta):
                flow_delta_np = _flow_tensor_to_numpy(flow_delta)
            else:
                flow_delta_np = np.asarray(flow_delta, dtype=np.float32)
            flow_full = _upsample_flow_numpy(flow_delta_np, h, w, device_obj)

            if prev_flow is None:
                flows_by_band[mov_idx] = flow_full
            else:
                flows_by_band[mov_idx] = _compose_flow_numpy(prev_flow, flow_full, device_obj)

            registered[mov_idx] = _apply_flow_to_image(
                bands[mov_idx],
                torch.tensor(flows_by_band[mov_idx], dtype=torch.float32, device=device_obj).unsqueeze(0),
                device_obj,
            )
            ncc_after = _pairwise_ncc(fixed, warped)

            step_logs.append({
                'global_step': global_step,
                'level': li + 1,
                'level_size': [sh, sw],
                'level_descending': level_desc,
                'level_direction': dir_label,
                'step_in_level': step,
                'fixed_index': ref_idx,
                'moving_index': mov_idx,
                'fixed_wavelength_nm': wl_ref,
                'moving_wavelength_nm': wl_mov,
                'NCC_before': ncc_before,
                'NCC_after': ncc_after,
            })

            if verbose:
                print(f'       NCC after={ncc_after:.4f}')

    return registered, step_logs, flows_by_band


def register_pifreg_chain(
    img_list: StackInput,
    device: str = 'cuda',
    descending: bool = True,
    wavelengths_nm: Optional[Sequence[str]] = None,
    schedule: str = SCHEDULE_PAIR_THEN_PYRAMID,
    pyramid_sizes: Tuple[int, ...] = DEFAULT_PYRAMID_SIZES,
    level_directions: Optional[Sequence[bool]] = None,
    alternate_direction: bool = False,
    verbose: bool = True,
    **pifreg_kwargs,
) -> Tuple[List[np.ndarray], Dict[str, Any], np.ndarray]:
    """
    链式全局配准（高波长锚点 → 低波长）。

    schedule:
        pair_then_pyramid   — 每对 PIFReg 内部多尺度（默认，与原先行为一致）
        pyramid_then_pairs  — 每层分辨率下完整链扫一遍，再进入更细层

    pyramid_then_pairs 专用:
        level_directions — 每层方向 True=690→400, False=400→690，如 [True,False,True]
        alternate_direction — 层间交替方向（L0 与 descending 一致）
    """
    bands = _as_band_list(img_list)
    n = len(bands)
    if n <= 1:
        return bands, {'mode': 'chain', 'num_bands': n}, np.zeros((0, 2, 0, 0), dtype=np.float32)

    schedule = (schedule or SCHEDULE_PAIR_THEN_PYRAMID).lower()
    if schedule not in (SCHEDULE_PAIR_THEN_PYRAMID, SCHEDULE_PYRAMID_THEN_PAIRS):
        raise ValueError(
            f'Unknown schedule={schedule!r}, expected '
            f'{SCHEDULE_PAIR_THEN_PYRAMID!r} or {SCHEDULE_PYRAMID_THEN_PAIRS!r}'
        )

    kwargs = _default_pifreg_kwargs()
    kwargs.update(pifreg_kwargs)

    if wavelengths_nm is None:
        wavelengths_nm = [str(i) for i in range(n)]

    if descending:
        anchor_idx = n - 1
        chain_steps = _chain_steps(n, True)
    else:
        anchor_idx = 0
        chain_steps = _chain_steps(n, False)

    per_level_desc = None
    if schedule == SCHEDULE_PYRAMID_THEN_PAIRS:
        registered, step_logs, flows_by_band = _register_chain_pyramid_first(
            bands, device, list(wavelengths_nm), kwargs, pyramid_sizes,
            descending, level_directions, alternate_direction, verbose,
        )
        levels = _build_pyramid_levels(bands[0].shape[0], bands[0].shape[1], pyramid_sizes)
        per_level_desc = _resolve_level_directions(
            len(levels), descending, level_directions, alternate_direction,
        )
    else:
        if verbose:
            print(
                f'{METHOD_CHAIN_NAME}: pair_then_pyramid, bands={n}, '
                f'anchor={wavelengths_nm[anchor_idx]} nm, '
                f'order={"descending" if descending else "ascending"}'
            )

        registered = [None] * n  # type: ignore
        registered[anchor_idx] = bands[anchor_idx].copy()
        step_logs = []
        flows_by_band: Dict[int, np.ndarray] = {}

        for step, (ref_idx, mov_idx) in enumerate(chain_steps, start=1):
            fixed = registered[ref_idx]
            moving = bands[mov_idx]
            wl_ref, wl_mov = wavelengths_nm[ref_idx], wavelengths_nm[mov_idx]
            ncc_before = _pairwise_ncc(fixed, moving)

            if verbose:
                print(
                    f'  [{step}/{len(chain_steps)}] PIFReg: fixed={wl_ref} nm <- moving={wl_mov} nm '
                    f'(NCC before={ncc_before:.4f})'
                )

            warped, flow = register_pifreg(
                fixed, moving, device=device, return_flow=True, **kwargs,
            )
            flows_by_band[mov_idx] = flow
            ncc_after = _pairwise_ncc(fixed, warped)
            registered[mov_idx] = warped

            step_logs.append({
                'step': step,
                'fixed_index': ref_idx,
                'moving_index': mov_idx,
                'fixed_wavelength_nm': wl_ref,
                'moving_wavelength_nm': wl_mov,
                'NCC_before': ncc_before,
                'NCC_after': ncc_after,
            })

            if verbose:
                print(f'       NCC after={ncc_after:.4f}')

    chain_ncc = evaluate_chain_pairwise_ncc(registered, wavelengths_nm, descending=descending)
    moving_indices = [i for i in range(n) if i != anchor_idx]
    flow_stack = np.stack([flows_by_band[i] for i in moving_indices], axis=0).astype(np.float32)

    h, w = bands[0].shape
    levels = _build_pyramid_levels(h, w, pyramid_sizes) if schedule == SCHEDULE_PYRAMID_THEN_PAIRS else None

    info = {
        'mode': 'chain',
        'method': METHOD_CHAIN_FULL_NAME,
        'schedule': schedule,
        'num_bands': n,
        'anchor_band_index': anchor_idx,
        'moving_band_indices': moving_indices,
        'flow_stack_shape': list(flow_stack.shape),
        'anchor_wavelength_nm': wavelengths_nm[anchor_idx],
        'direction': 'descending_wavelength' if descending else 'ascending_wavelength',
        'steps': step_logs,
        'chain_pairwise_ncc': chain_ncc,
        'pifreg_kwargs': {k: v for k, v in kwargs.items() if k != 'save_model_path'},
        'pyramid_sizes': list(pyramid_sizes) if schedule == SCHEDULE_PYRAMID_THEN_PAIRS else None,
        'pyramid_levels': [list(lv) for lv in levels] if levels else None,
        'level_directions': (
            ['690→400' if d else '400→690' for d in per_level_desc]
            if schedule == SCHEDULE_PYRAMID_THEN_PAIRS else None
        ),
        'alternate_direction': alternate_direction if schedule == SCHEDULE_PYRAMID_THEN_PAIRS else None,
    }
    return registered, info, flow_stack
