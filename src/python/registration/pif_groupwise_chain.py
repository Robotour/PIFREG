# PIFReg Chain Groupwise — 高波长→低波长链式两两配准
#
# 配准顺序：690 → 680 → 670 → …（波长降序）
#   - 栈内最高波长波段为锚点（不动）
#   - 下一波段配准到「上一波段配准后的结果」
#   - 每对使用 register_pifreg，优化目标为 NCC
#
# 与 stackflow 联合优化不同：本方法为 N 次独立 pairwise PIFReg，实现简单、位移场逐对估计。

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..metrics.evaluation import compute_NCC
from .pif_registration import register_pifreg

METHOD_CHAIN_NAME = 'PIFReg-Chain'
METHOD_CHAIN_FULL_NAME = 'PIFReg Descending Wavelength Chain Registration'

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


def _pairwise_ncc(fixed: np.ndarray, moving: np.ndarray) -> float:
    return float(compute_NCC(fixed, moving))


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


def register_pifreg_chain(
    img_list: StackInput,
    device: str = 'cuda',
    descending: bool = True,
    wavelengths_nm: Optional[Sequence[str]] = None,
    verbose: bool = True,
    **pifreg_kwargs,
) -> Tuple[List[np.ndarray], Dict[str, Any], np.ndarray]:
    """
    链式全局配准（高波长锚点 → 低波长）。

    假设 img_list 按波长升序排列（400…690）。默认从 690 锚点起，依次配准 680、670…

    参数:
        img_list: (H,W) 列表或 (N,H,W)，升序波长
        descending: True 时从最高波长向最低波长链式传递
        wavelengths_nm: 可选波长字符串，用于日志
        **pifreg_kwargs: 传给 register_pifreg

    返回:
        registered_list: 归一化空间配准结果（用于指标）
        info: 链式元数据及配准后 pairwise NCC
        flow_stack: (N-1, 2, H, W) 各位移场，供原图 warp
    """
    bands = _as_band_list(img_list)
    n = len(bands)
    if n <= 1:
        return bands, {'mode': 'chain', 'num_bands': n}, np.zeros((0, 2, 0, 0), dtype=np.float32)

    kwargs = _default_pifreg_kwargs()
    kwargs.update(pifreg_kwargs)

    if wavelengths_nm is None:
        wavelengths_nm = [str(i) for i in range(n)]

    registered = [None] * n  # type: ignore

    if descending:
        anchor_idx = n - 1
        registered[anchor_idx] = bands[anchor_idx].copy()
        chain_steps = [(i + 1, i) for i in range(n - 2, -1, -1)]
    else:
        anchor_idx = 0
        registered[anchor_idx] = bands[anchor_idx].copy()
        chain_steps = [(i - 1, i) for i in range(1, n)]

    if verbose:
        print(
            f'{METHOD_CHAIN_NAME}: chain registration, bands={n}, '
            f'anchor={wavelengths_nm[anchor_idx]} nm, '
            f'order={"descending" if descending else "ascending"}'
        )

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

        warped, flow = register_pifreg(fixed, moving, device=device, return_flow=True, **kwargs)
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

    info = {
        'mode': 'chain',
        'method': METHOD_CHAIN_FULL_NAME,
        'num_bands': n,
        'anchor_band_index': anchor_idx,
        'moving_band_indices': moving_indices,
        'flow_stack_shape': list(flow_stack.shape),
        'anchor_wavelength_nm': wavelengths_nm[anchor_idx],
        'direction': 'descending_wavelength' if descending else 'ascending_wavelength',
        'steps': step_logs,
        'chain_pairwise_ncc': chain_ncc,
        'pifreg_kwargs': {k: v for k, v in kwargs.items() if k != 'save_model_path'},
    }
    return registered, info, flow_stack
