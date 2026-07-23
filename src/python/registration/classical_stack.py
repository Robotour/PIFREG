"""Whole-stack classical registration wrappers for HSI baseline comparison."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..preprocessing.band_preprocess import refresh_histogram_equalized
from .methods import (
    register_elastix_chain,
    register_elastix_groupwise,
    register_keren,
    register_stackreg,
)

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
) -> List[np.ndarray]:
    """Pairwise Elastix chain: estimate on hist-eq, warp raw."""
    eq, raw = _coerce_eq_raw(bands_eq, bands_raw)
    return register_elastix_chain(
        eq,
        epochs=epochs,
        spacinginvoxels=spacinginvoxels,
        descending=descending,
        raw_list=raw,
    )


def register_stack_stackreg_chain(
    bands_eq: Sequence[np.ndarray],
    bands_raw: Optional[Sequence[np.ndarray]] = None,
    transform_type: str = 'bilinear',
    descending: bool = True,
) -> List[np.ndarray]:
    """Pairwise StackReg chain: estimate on hist-eq, transform raw."""
    eq_list, raw_list = _coerce_eq_raw(bands_eq, bands_raw)
    if len(eq_list) < 2:
        return raw_list
    for fixed_idx, moving_idx in chain_pair_indices(len(eq_list), descending=descending):
        raw_list[moving_idx] = register_stackreg(
            eq_list[fixed_idx],
            eq_list[moving_idx],
            transform_type=transform_type,
            moving_raw=raw_list[moving_idx],
        )
        eq_list[moving_idx] = refresh_histogram_equalized(raw_list[moving_idx])
    return raw_list


def register_stack_keren(
    bands_eq: Sequence[np.ndarray],
    bands_raw: Optional[Sequence[np.ndarray]] = None,
    descending: bool = True,
) -> List[np.ndarray]:
    """KEREN on hist-eq; apply estimated rigid motion to raw bands."""
    from ..utils.image_transform import shift_and_rotate

    eq_list, raw_list = _coerce_eq_raw(bands_eq, bands_raw)
    if len(eq_list) < 2:
        return raw_list

    if descending:
        eq_work = list(reversed(eq_list))
        raw_work = list(reversed(raw_list))
    else:
        eq_work = list(eq_list)
        raw_work = list(raw_list)

    delta_est, phi_est = register_keren(eq_work)
    registered_raw = [raw_work[0].copy()]
    for i in range(1, len(raw_work)):
        registered_raw.append(
            shift_and_rotate(raw_work[i], delta_est[i, 0], delta_est[i, 1], phi_est[i])
        )

    if descending:
        return list(reversed(registered_raw))
    return registered_raw


def register_stack_elastix_groupwise(
    bands_eq: Sequence[np.ndarray],
    bands_raw: Optional[Sequence[np.ndarray]] = None,
    epochs: int = 80,
    spacinginvoxels: int = 20,
    verbose: int = 0,
) -> List[np.ndarray]:
    """Elastix groupwise on hist-eq stack; apply fields to raw bands."""
    from src.python.experiments.experiment_data import warp_bands_with_elastix_fields

    eq_list, raw_list = _coerce_eq_raw(bands_eq, bands_raw)
    _, fields = register_elastix_groupwise(
        eq_list,
        epochs=epochs,
        spacinginvoxels=spacinginvoxels,
        verbose=verbose,
    )
    return warp_bands_with_elastix_fields(raw_list, fields)


def register_stack_classical(
    method: str,
    bands_eq: Sequence[np.ndarray],
    bands_raw: Optional[Sequence[np.ndarray]] = None,
    descending: bool = True,
    **kwargs,
) -> List[np.ndarray]:
    """Register full stack; returns warped **raw** intensity bands."""
    method = method.lower()
    common = dict(bands_eq=bands_eq, bands_raw=bands_raw, descending=descending)
    if method == 'elastix_groupwise':
        return register_stack_elastix_groupwise(
            **common,
            epochs=kwargs.get('epochs', 80),
            spacinginvoxels=kwargs.get('spacinginvoxels', 20),
            verbose=kwargs.get('elastix_verbose', kwargs.get('verbose', 0)),
        )
    if method == 'elastix_chain':
        return register_stack_elastix_chain(
            **common,
            epochs=kwargs.get('epochs', 20),
            spacinginvoxels=kwargs.get('spacinginvoxels', 20),
        )
    if method == 'stackreg_chain':
        return register_stack_stackreg_chain(
            **common,
            transform_type=kwargs.get('transform_type', 'bilinear'),
        )
    if method == 'keren':
        return register_stack_keren(**common)
    raise ValueError(f'Unknown classical method: {method}. Choose from {CLASSICAL_METHODS}')


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
    image_size=(512, 512),
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
