# PIFReg Keyframe Scaffold — 稀疏关键帧 chain + 光谱 flow 插值
#
# Stage 1 of cascade registration: run PIFReg only between keyframe bands,
# compose absolute per-band flows, linearly interpolate flows for intermediate bands.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..metrics.evaluation import compute_NCC
from .pif_groupwise_chain import (
    StackInput,
    _as_band_list,
    evaluate_chain_pairwise_ncc,
)
from .pif_groupwise_stackflow import (
    _compose_flows,
    _moving_band_indices,
    warp_bands_with_flow_stack,
)
from .pif_registration import register_pifreg

METHOD_KEYFRAME_NAME = 'PIFReg-Keyframe'
METHOD_KEYFRAME_FULL_NAME = 'PIFReg Keyframe Spectral Flow Scaffold'


def _default_keyframe_pifreg_kwargs() -> Dict[str, Any]:
    """Fast pairwise settings for scaffold stage."""
    return dict(
        fast_mode=True,
        image_loss='ncc',
        early_stop=True,
        patience=60,
        epochs=2000,
        lr_schedule='cosine',
        multiscale=True,
        scales=(0.25, 0.5, 1.0),
        affine_init=False,
        histogram_match=True,
    )


def select_keyframe_indices(
    num_bands: int,
    anchor_idx: int,
    interval: int = 5,
) -> List[int]:
    """
    选择关键帧索引（含锚点与两端波段）。

    interval: 相邻关键帧最大间隔（波段数）。
    """
    if num_bands <= 1:
        return [0]
    interval = max(int(interval), 1)
    anchor_idx = int(anchor_idx) % num_bands
    indices = {0, anchor_idx, num_bands - 1}
    for i in range(0, num_bands, interval):
        indices.add(i)
    return sorted(indices)


def _flow_to_tensor(flow: np.ndarray, device) -> torch.Tensor:
    arr = np.asarray(flow, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr.squeeze(0)
    if arr.ndim != 3 or arr.shape[0] != 2:
        raise ValueError(f'Expected flow shape (2,H,W), got {arr.shape}')
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)


def compose_flow_numpy(
    base_flow: np.ndarray,
    delta_flow: np.ndarray,
    device: str = 'cpu',
) -> np.ndarray:
    """Compose absolute flows: warp(moving, result) ≈ warp(warp(moving, delta), base)."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    base_t = _flow_to_tensor(base_flow, device)
    delta_t = _flow_to_tensor(delta_flow, device)
    h, w = base_t.shape[-2], base_t.shape[-1]
    composed = _compose_flows(base_t, delta_t, (h, w), device)
    return composed.squeeze(0).detach().cpu().numpy().astype(np.float32)


def interpolate_flows_spectral(
    keyframe_indices: Sequence[int],
    abs_flows: Dict[int, np.ndarray],
    num_bands: int,
    anchor_idx: int,
) -> Dict[int, np.ndarray]:
    """
    对非关键帧波段，在相邻关键帧 absolute flow 之间按索引线性插值。
    锚点波段 flow 视为零（恒等变换），作为插值端点。
    """
    kf = sorted(set(keyframe_indices))

    if anchor_idx in abs_flows:
        raise ValueError('anchor band must not have a flow entry in abs_flows')

    for idx in kf:
        if idx != anchor_idx and idx not in abs_flows:
            raise ValueError(f'missing absolute flow for keyframe band {idx}')

    out: Dict[int, np.ndarray] = {}
    for idx in kf:
        if idx != anchor_idx:
            out[idx] = abs_flows[idx].astype(np.float32)

    sample_flow = next(iter(out.values()))

    def _flow_at(band_idx: int) -> np.ndarray:
        if band_idx == anchor_idx:
            return np.zeros_like(sample_flow)
        return out[band_idx]

    for j in range(num_bands):
        if j == anchor_idx or j in out:
            continue
        lo = max(i for i in kf if i <= j)
        hi = min(i for i in kf if i >= j)
        if lo == hi:
            out[j] = _flow_at(lo).copy()
            continue
        t = (j - lo) / float(hi - lo)
        out[j] = ((1.0 - t) * _flow_at(lo) + t * _flow_at(hi)).astype(np.float32)
    return out


def absolute_flows_to_stack(
    abs_flows: Dict[int, np.ndarray],
    num_bands: int,
    anchor_idx: int,
) -> np.ndarray:
    """Dict[band_idx -> (2,H,W)] → (N-1, 2, H, W) stackflow layout."""
    moving = _moving_band_indices(num_bands, anchor_idx)
    return np.stack([abs_flows[i] for i in moving], axis=0).astype(np.float32)


def register_pifreg_keyframe_scaffold(
    img_list: StackInput,
    device: str = 'cuda',
    anchor_band_idx: int = -1,
    keyframe_interval: int = 5,
    descending: bool = True,
    wavelengths_nm: Optional[Sequence[str]] = None,
    verbose: bool = True,
    **pifreg_kwargs,
) -> Tuple[List[np.ndarray], Dict[str, Any], np.ndarray]:
    """
    稀疏关键帧 chain：仅对关键帧对运行 PIFReg，再插值得到全栈 per-band flow。

    返回:
        registered: 脚手架配准后的波段列表
        info: 元数据（关键帧、步数、chain NCC 等）
        flow_stack: (N-1, 2, H, W) 供 StackFlow refine / 原图 warp
    """
    bands = _as_band_list(img_list)
    n = len(bands)
    if n <= 1:
        return bands, {'mode': 'keyframe', 'num_bands': n}, np.zeros((0, 2, 0, 0), dtype=np.float32)

    anchor_idx = int(anchor_band_idx) % n
    if not descending:
        raise NotImplementedError('keyframe scaffold currently supports descending (high-λ anchor) only')

    kwargs = _default_keyframe_pifreg_kwargs()
    kwargs.update(pifreg_kwargs)

    if wavelengths_nm is None:
        wavelengths_nm = [str(i) for i in range(n)]

    keyframes = select_keyframe_indices(n, anchor_idx, keyframe_interval)
    kf_desc = sorted(keyframes, reverse=True)

    if verbose:
        wl_kf = [wavelengths_nm[i] for i in keyframes]
        print(
            f'{METHOD_KEYFRAME_NAME}: {n} bands, anchor={wavelengths_nm[anchor_idx]} nm, '
            f'keyframes={len(keyframes)}/{n} (interval={keyframe_interval}), '
            f'indices={keyframes}, λ={wl_kf}'
        )

    registered: List[Optional[np.ndarray]] = [None] * n
    registered[anchor_idx] = bands[anchor_idx].copy()
    abs_flows: Dict[int, np.ndarray] = {}
    step_logs = []

    for step, i in enumerate(range(len(kf_desc) - 1), start=1):
        ref_idx = kf_desc[i]
        mov_idx = kf_desc[i + 1]
        fixed = registered[ref_idx]
        moving = bands[mov_idx]
        wl_ref, wl_mov = wavelengths_nm[ref_idx], wavelengths_nm[mov_idx]
        ncc_before = float(compute_NCC(fixed, moving))

        if verbose:
            print(
                f'  [{step}/{len(kf_desc) - 1}] PIFReg keyframe: '
                f'fixed={wl_ref} nm <- moving={wl_mov} nm (NCC before={ncc_before:.4f})'
            )

        warped, flow_pair = register_pifreg(
            fixed, moving, device=device, return_flow=True, **kwargs,
        )
        registered[mov_idx] = warped

        if ref_idx == anchor_idx:
            abs_flows[mov_idx] = flow_pair.astype(np.float32)
        else:
            abs_flows[mov_idx] = compose_flow_numpy(abs_flows[ref_idx], flow_pair, device=device)

        ncc_after = float(compute_NCC(fixed, warped))
        step_logs.append({
            'step': step,
            'fixed_index': ref_idx,
            'moving_index': mov_idx,
            'fixed_wavelength_nm': wl_ref,
            'moving_wavelength_nm': wl_mov,
            'NCC_before': ncc_before,
            'NCC_after': ncc_after,
            'keyframe_pair': True,
        })
        if verbose:
            print(f'       NCC after={ncc_after:.4f}')

    abs_flows_all = interpolate_flows_spectral(keyframes, abs_flows, n, anchor_idx)
    flow_stack = absolute_flows_to_stack(abs_flows_all, n, anchor_idx)

    registered_full = warp_bands_with_flow_stack(
        bands, flow_stack, anchor_band_idx=anchor_idx, device=device,
    )
    chain_ncc = evaluate_chain_pairwise_ncc(registered_full, wavelengths_nm, descending=descending)

    info = {
        'mode': 'keyframe',
        'method': METHOD_KEYFRAME_FULL_NAME,
        'num_bands': n,
        'anchor_band_index': anchor_idx,
        'keyframe_indices': keyframes,
        'keyframe_interval': keyframe_interval,
        'num_pifreg_calls': len(kf_desc) - 1,
        'moving_band_indices': _moving_band_indices(n, anchor_idx),
        'flow_stack_shape': list(flow_stack.shape),
        'steps': step_logs,
        'chain_pairwise_ncc': chain_ncc,
        'pifreg_kwargs': {k: v for k, v in kwargs.items() if k != 'save_model_path'},
    }
    return registered_full, info, flow_stack
