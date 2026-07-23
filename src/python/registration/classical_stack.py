"""Whole-stack classical registration wrappers for HSI baseline comparison."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..preprocessing.band_preprocess import refresh_histogram_equalized
from .methods import (
    register_elastix_chain,
    register_elastix_groupwise,
    register_keren,
)
from pystackreg import StackReg

CLASSICAL_METHODS = (
    'elastix_groupwise',
    'elastix_chain',
    'stackreg_chain',
    'keren',
)


def chain_pair_indices(n: int, descending: bool = True) -> List[Tuple[int, int]]:
    if descending:
        return [(i + 1, i) for i in range(n - 2, -1, -1)]
    return [(i - 1, i) for i in range(1, n)]


def anchor_index(n: int, descending: bool = True) -> int:
    return n - 1 if descending else 0


def _coerce_eq_raw(
    bands_eq: Sequence[np.ndarray],
    bands_raw: Optional[Sequence[np.ndarray]],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    eq = [np.asarray(b, dtype=np.float32).copy() for b in bands_eq]
    raw = [np.asarray(b, dtype=np.float32).copy() for b in (bands_raw if bands_raw is not None else bands_eq)]
    if len(eq) != len(raw):
        raise ValueError(f'bands_eq length {len(eq)} != bands_raw length {len(raw)}')
    return eq, raw


def register_stack_elastix_chain(
    bands_eq: Sequence[np.ndarray],
    bands_raw: Optional[Sequence[np.ndarray]] = None,
    epochs: int = 20,
    spacinginvoxels: int = 20,
    descending: bool = True,
    return_steps: bool = False,
):
    """Pairwise Elastix chain: estimate on hist-eq, warp raw."""
    eq, raw = _coerce_eq_raw(bands_eq, bands_raw)
    result = register_elastix_chain(
        eq,
        epochs=epochs,
        spacinginvoxels=spacinginvoxels,
        descending=descending,
        raw_list=raw,
        return_steps=return_steps,
    )
    if return_steps:
        return result
    return result


def register_stack_stackreg_chain(
    bands_eq: Sequence[np.ndarray],
    bands_raw: Optional[Sequence[np.ndarray]] = None,
    transform_type: str = 'bilinear',
    descending: bool = True,
    return_transforms: bool = False,
):
    """Pairwise StackReg chain: estimate on hist-eq, transform raw."""
    eq_list, raw_list = _coerce_eq_raw(bands_eq, bands_raw)
    if len(eq_list) < 2:
        return (raw_list, []) if return_transforms else raw_list

    transforms = []
    for step_i, (fixed_idx, moving_idx) in enumerate(
        chain_pair_indices(len(eq_list), descending=descending), start=1,
    ):
        if transform_type == 'translation':
            sr = StackReg(StackReg.TRANSLATION)
        elif transform_type == 'rigid':
            sr = StackReg(StackReg.RIGID_BODY)
        elif transform_type == 'scaled_rotation':
            sr = StackReg(StackReg.SCALED_ROTATION)
        elif transform_type == 'affine':
            sr = StackReg(StackReg.AFFINE)
        else:
            sr = StackReg(StackReg.BILINEAR)
        sr.register(eq_list[fixed_idx], eq_list[moving_idx])
        raw_list[moving_idx] = sr.transform(raw_list[moving_idx]).astype(np.float32)
        eq_list[moving_idx] = refresh_histogram_equalized(raw_list[moving_idx])
        if return_transforms:
            transforms.append({
                'step': step_i,
                'fixed_idx': fixed_idx,
                'moving_idx': moving_idx,
                'matrix': sr.get_matrix().tolist(),
            })
    if return_transforms:
        return raw_list, transforms
    return raw_list


def register_stack_keren(
    bands_eq: Sequence[np.ndarray],
    bands_raw: Optional[Sequence[np.ndarray]] = None,
    descending: bool = True,
    return_transforms: bool = False,
):
    """KEREN on hist-eq; apply estimated rigid motion to raw bands."""
    from ..utils.image_transform import shift_and_rotate

    eq_list, raw_list = _coerce_eq_raw(bands_eq, bands_raw)
    if len(eq_list) < 2:
        return (raw_list, []) if return_transforms else raw_list

    if descending:
        eq_work = list(reversed(eq_list))
        raw_work = list(reversed(raw_list))
        index_map = list(reversed(range(len(raw_list))))
    else:
        eq_work = list(eq_list)
        raw_work = list(raw_list)
        index_map = list(range(len(raw_list)))

    delta_est, phi_est = register_keren(eq_work)
    registered_raw = [raw_work[0].copy()]
    transforms = [{
        'band_index': index_map[0],
        'dx': 0.0,
        'dy': 0.0,
        'rotation_deg': 0.0,
    }]
    for i in range(1, len(raw_work)):
        registered_raw.append(
            shift_and_rotate(raw_work[i], delta_est[i, 0], delta_est[i, 1], phi_est[i])
        )
        if return_transforms:
            transforms.append({
                'band_index': index_map[i],
                'dx': float(delta_est[i, 0]),
                'dy': float(delta_est[i, 1]),
                'rotation_deg': float(phi_est[i]),
            })

    if descending:
        registered_raw = list(reversed(registered_raw))
        if return_transforms:
            transforms = sorted(transforms, key=lambda x: x['band_index'])
    if return_transforms:
        return registered_raw, transforms
    return registered_raw


def register_stack_elastix_groupwise(
    bands_eq: Sequence[np.ndarray],
    bands_raw: Optional[Sequence[np.ndarray]] = None,
    epochs: int = 80,
    spacinginvoxels: int = 20,
    verbose: int = 0,
    return_fields: bool = False,
):
    """Elastix groupwise on hist-eq stack; apply fields to raw bands."""
    from src.python.experiments.experiment_data import warp_bands_with_elastix_fields

    eq_list, raw_list = _coerce_eq_raw(bands_eq, bands_raw)
    _, fields = register_elastix_groupwise(
        eq_list,
        epochs=epochs,
        spacinginvoxels=spacinginvoxels,
        verbose=verbose,
    )
    warped = warp_bands_with_elastix_fields(raw_list, fields)
    if return_fields:
        return warped, fields
    return warped


def register_stack_classical_detailed(
    method: str,
    bands_eq: Sequence[np.ndarray],
    bands_raw: Optional[Sequence[np.ndarray]] = None,
    descending: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Register stack and return warped raw bands plus artifacts for saving."""
    method = method.lower()
    common = dict(bands_eq=bands_eq, bands_raw=bands_raw, descending=descending)
    result: Dict[str, Any] = {
        'method': method,
        'chain_steps': None,
        'elastix_fields': None,
        'transform_meta': None,
    }

    if method == 'elastix_groupwise':
        raw_after, fields = register_stack_elastix_groupwise(
            **common,
            epochs=kwargs.get('epochs', 80),
            spacinginvoxels=kwargs.get('spacinginvoxels', 20),
            verbose=kwargs.get('elastix_verbose', kwargs.get('verbose', 0)),
            return_fields=True,
        )
        result['bands_raw_after'] = raw_after
        result['elastix_fields'] = fields
        return result

    if method == 'elastix_chain':
        raw_after, steps = register_stack_elastix_chain(
            **common,
            epochs=kwargs.get('epochs', 20),
            spacinginvoxels=kwargs.get('spacinginvoxels', 20),
            return_steps=True,
        )
        result['bands_raw_after'] = raw_after
        result['chain_steps'] = steps
        return result

    if method == 'stackreg_chain':
        raw_after, transforms = register_stack_stackreg_chain(
            **common,
            transform_type=kwargs.get('transform_type', 'bilinear'),
            return_transforms=True,
        )
        result['bands_raw_after'] = raw_after
        result['transform_meta'] = {'type': 'stackreg_chain', 'steps': transforms}
        return result

    if method == 'keren':
        raw_after, transforms = register_stack_keren(**common, return_transforms=True)
        result['bands_raw_after'] = raw_after
        result['transform_meta'] = {'type': 'keren', 'bands': transforms}
        return result

    raise ValueError(f'Unknown classical method: {method}. Choose from {CLASSICAL_METHODS}')


def register_stack_classical(
    method: str,
    bands_eq: Sequence[np.ndarray],
    bands_raw: Optional[Sequence[np.ndarray]] = None,
    descending: bool = True,
    **kwargs,
) -> List[np.ndarray]:
    """Register full stack; returns warped **raw** intensity bands."""
    detail = register_stack_classical_detailed(
        method, bands_eq, bands_raw=bands_raw, descending=descending, **kwargs,
    )
    return detail['bands_raw_after']


def make_classical_register_fn(
    method: str,
    descending: bool = True,
    **register_kwargs,
):
    """Return (bands_eq, bands_raw) -> warped raw bands."""

    def _register(bands_eq: List[np.ndarray], bands_raw: List[np.ndarray]) -> List[np.ndarray]:
        return register_stack_classical(
            method,
            bands_eq,
            bands_raw=bands_raw,
            descending=descending,
            **register_kwargs,
        )

    return _register


def evaluate_classical_sessions(
    method: str,
    folders: Sequence,
    image_size=None,
    descending: bool = True,
    max_sessions: Optional[int] = None,
    verbose: bool = True,
    **register_kwargs,
) -> Dict[str, Any]:
    """All-band pairwise mean metrics on test sessions (metrics on raw bands)."""
    from src.python.experiments.stack_pairwise_metrics import evaluate_test_sessions_all_pairs

    return evaluate_test_sessions_all_pairs(
        folders,
        image_size=image_size,
        register_fn=make_classical_register_fn(method, descending=descending, **register_kwargs),
        max_sessions=max_sessions,
        verbose=verbose,
    )
