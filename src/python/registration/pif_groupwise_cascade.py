# PIFReg Cascade — Keyframe scaffold (Stage 1) + StackFlow residual refine (Stage 2)
#
# Combines fast sparse pairwise registration with a single joint StackFlow pass
# to correct global drift while keeping total runtime well below full chain.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .pif_groupwise_chain import StackInput, _as_band_list, evaluate_chain_pairwise_ncc
from .pif_groupwise_keyframe import register_pifreg_keyframe_scaffold
from .pif_groupwise_stackflow import (
    FEATURE_MODE_MEAN_ANCHOR,
    register_pifreg_groupwise_stackflow,
)

METHOD_CASCADE_NAME = 'PIFReg-Cascade'
METHOD_CASCADE_FULL_NAME = 'PIFReg Keyframe Scaffold + StackFlow Residual Refinement'

DEFAULT_REFINE_PYRAMID_SIZES = (128, 256, 512)
DEFAULT_REFINE_EPOCHS_PER_LEVEL = (300, 500, 800)
DEFAULT_REFINE_PATIENCE_PER_LEVEL = (50, 60, 80)


def register_pifreg_groupwise_cascade(
    img_list: StackInput,
    device: str = 'cuda',
    anchor_band_idx: int = -1,
    keyframe_interval: int = 5,
    descending: bool = True,
    wavelengths_nm: Optional[Sequence[str]] = None,
    # Stage 1 (keyframe scaffold)
    scaffold_pifreg_kwargs: Optional[Dict[str, Any]] = None,
    # Stage 2 (StackFlow refine)
    refine_pyramid_sizes: Tuple[int, ...] = DEFAULT_REFINE_PYRAMID_SIZES,
    refine_epochs_per_level: Optional[Sequence[int]] = None,
    refine_patience_per_level: Optional[Sequence[int]] = None,
    refine_lr: float = 2e-4,
    refine_lamda: float = 0.01,
    refine_ncc_weight: float = 1.0,
    refine_int_steps: int = 3,
    refine_int_downsize: int = 2,
    refine_early_stop: bool = True,
    refine_min_delta: float = 1e-4,
    refine_lr_schedule: str = 'cosine',
    refine_lr_min: float = 1e-6,
    refine_fast_mode: bool = True,
    refine_feature_mode: str = FEATURE_MODE_MEAN_ANCHOR,
    refine_spectral_enc_channels: int = 4,
    refine_spectral_enc_kernel: int = 3,
    skip_refine: bool = False,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any], np.ndarray]:
    """
    两阶段群组配准：
      Stage 1 — 稀疏关键帧 PIFReg + flow 插值（快速脚手架）
      Stage 2 — StackFlow 残差精修（init_flow_stack = Stage 1 flow）

    返回:
        registered, info, flow_stack
    """
    bands = _as_band_list(img_list)
    n = len(bands)
    anchor_idx = int(anchor_band_idx) % n if n else 0

    if n <= 1:
        return bands, {'mode': 'cascade', 'num_bands': n}, np.zeros((0, 2, 0, 0), dtype=np.float32)

    scaffold_kw = dict(scaffold_pifreg_kwargs or {})

    if verbose:
        print('=' * 60)
        print(f'{METHOD_CASCADE_NAME}: Stage 1 — Keyframe scaffold')
        print('=' * 60)

    registered_scaffold, scaffold_info, flow_scaffold = register_pifreg_keyframe_scaffold(
        bands,
        device=device,
        anchor_band_idx=anchor_idx,
        keyframe_interval=keyframe_interval,
        descending=descending,
        wavelengths_nm=wavelengths_nm,
        verbose=verbose,
        **scaffold_kw,
    )

    if skip_refine:
        info = {
            'mode': 'cascade',
            'method': METHOD_CASCADE_FULL_NAME,
            'num_bands': n,
            'anchor_band_idx': anchor_idx,
            'keyframe_interval': keyframe_interval,
            'skip_refine': True,
            'scaffold': scaffold_info,
            'chain_pairwise_ncc': scaffold_info.get('chain_pairwise_ncc'),
        }
        return registered_scaffold, info, flow_scaffold

    ep_levels = list(refine_epochs_per_level or DEFAULT_REFINE_EPOCHS_PER_LEVEL)
    pat_levels = list(refine_patience_per_level or DEFAULT_REFINE_PATIENCE_PER_LEVEL)

    if verbose:
        print('')
        print('=' * 60)
        print(f'{METHOD_CASCADE_NAME}: Stage 2 — StackFlow residual refine')
        print('=' * 60)
        print(f'  init from scaffold flow {list(flow_scaffold.shape)}')
        print(f'  pyramid={list(refine_pyramid_sizes)}, epochs={ep_levels}, patience={pat_levels}')

    registered_final, refine_info, flow_final = register_pifreg_groupwise_stackflow(
        bands,
        device=device,
        anchor_band_idx=anchor_idx,
        pyramid_sizes=refine_pyramid_sizes,
        epochs_per_level=ep_levels,
        patience_per_level=pat_levels,
        lr=refine_lr,
        lamda=refine_lamda,
        ncc_weight=refine_ncc_weight,
        int_steps=refine_int_steps,
        int_downsize=refine_int_downsize,
        early_stop=refine_early_stop,
        min_delta=refine_min_delta,
        lr_schedule=refine_lr_schedule,
        lr_min=refine_lr_min,
        fast_mode=refine_fast_mode,
        feature_mode=refine_feature_mode,
        spectral_enc_channels=refine_spectral_enc_channels,
        spectral_enc_kernel=refine_spectral_enc_kernel,
        init_flow_stack=flow_scaffold,
        verbose=verbose,
    )

    chain_before = scaffold_info.get('chain_pairwise_ncc', {})
    chain_after = evaluate_chain_pairwise_ncc(
        registered_final, wavelengths_nm, descending=descending,
    )

    info = {
        'mode': 'cascade',
        'method': METHOD_CASCADE_FULL_NAME,
        'num_bands': n,
        'anchor_band_idx': anchor_idx,
        'keyframe_interval': keyframe_interval,
        'skip_refine': False,
        'scaffold': scaffold_info,
        'refine': refine_info,
        'chain_pairwise_ncc_scaffold': chain_before,
        'chain_pairwise_ncc': chain_after,
        'chain_ncc_delta': (
            chain_after['mean_NCC'] - chain_before.get('mean_NCC', 0.0)
            if chain_before else None
        ),
    }
    return registered_final, info, flow_final
