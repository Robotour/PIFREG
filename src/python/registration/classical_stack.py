"""Whole-stack classical registration wrappers for HSI baseline comparison."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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


def register_stack_elastix_chain(
    bands: Sequence[np.ndarray],
    epochs: int = 20,
    spacinginvoxels: int = 20,
    descending: bool = True,
) -> List[np.ndarray]:
    """Pairwise Elastix along wavelength chain (same inference graph as VoxelMorph chain)."""
    return register_elastix_chain(
        bands,
        epochs=epochs,
        spacinginvoxels=spacinginvoxels,
        descending=descending,
    )


def register_stack_stackreg_chain(
    bands: Sequence[np.ndarray],
    transform_type: str = 'bilinear',
    descending: bool = True,
) -> List[np.ndarray]:
    """Pairwise StackReg along wavelength chain."""
    registered = [np.asarray(b, dtype=np.float32).copy() for b in bands]
    if len(registered) < 2:
        return registered
    for fixed_idx, moving_idx in chain_pair_indices(len(registered), descending=descending):
        registered[moving_idx] = register_stackreg(
            registered[fixed_idx],
            registered[moving_idx],
            transform_type=transform_type,
        )
    return registered


def register_stack_keren(
    bands: Sequence[np.ndarray],
    descending: bool = True,
) -> List[np.ndarray]:
    """
    KEREN pyramid LK: all bands aligned to one reference band.

    VoxelMorph chain uses the longest-wavelength band as anchor; KEREN uses img_list[0]
    as reference, so we reverse order when descending=True.
    """
    from ..utils.image_transform import shift_and_rotate

    bands = [np.asarray(b, dtype=np.float32) for b in bands]
    if len(bands) < 2:
        return [b.copy() for b in bands]

    if descending:
        work = list(reversed(bands))
    else:
        work = list(bands)

    delta_est, phi_est = register_keren(work)
    registered_work = [work[0].copy()]
    for i in range(1, len(work)):
        registered_work.append(
            shift_and_rotate(work[i], delta_est[i, 0], delta_est[i, 1], phi_est[i])
        )

    if descending:
        return list(reversed(registered_work))
    return registered_work


def register_stack_elastix_groupwise(
    bands: Sequence[np.ndarray],
    epochs: int = 80,
    spacinginvoxels: int = 20,
    verbose: int = 0,
) -> List[np.ndarray]:
    """Elastix BSplineStackTransform groupwise registration on the full stack."""
    bands = [np.asarray(b, dtype=np.float32) for b in bands]
    registered, _ = register_elastix_groupwise(
        bands,
        epochs=epochs,
        spacinginvoxels=spacinginvoxels,
        verbose=verbose,
    )
    return registered


def register_stack_classical(
    method: str,
    bands: Sequence[np.ndarray],
    descending: bool = True,
    **kwargs,
) -> List[np.ndarray]:
    method = method.lower()
    if method == 'elastix_groupwise':
        return register_stack_elastix_groupwise(
            bands,
            epochs=kwargs.get('epochs', 80),
            spacinginvoxels=kwargs.get('spacinginvoxels', 20),
            verbose=kwargs.get('verbose', 0),
        )
    if method == 'elastix_chain':
        return register_stack_elastix_chain(
            bands,
            epochs=kwargs.get('epochs', 20),
            spacinginvoxels=kwargs.get('spacinginvoxels', 20),
            descending=descending,
        )
    if method == 'stackreg_chain':
        return register_stack_stackreg_chain(
            bands,
            transform_type=kwargs.get('transform_type', 'bilinear'),
            descending=descending,
        )
    if method == 'keren':
        return register_stack_keren(bands, descending=descending)
    raise ValueError(f'Unknown classical method: {method}. Choose from {CLASSICAL_METHODS}')


def make_classical_register_fn(
    method: str,
    descending: bool = True,
    **register_kwargs,
):
    """Return a bands -> registered_bands callable for stack_pairwise_metrics."""

    def _register(bands: List[np.ndarray]) -> List[np.ndarray]:
        return register_stack_classical(
            method,
            bands,
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
    """All-band pairwise mean metrics on test sessions (aligned with metrics CSV)."""
    from src.python.experiments.stack_pairwise_metrics import evaluate_test_sessions_all_pairs

    return evaluate_test_sessions_all_pairs(
        folders,
        image_size=image_size,
        register_fn=make_classical_register_fn(method, descending=descending, **register_kwargs),
        max_sessions=max_sessions,
        verbose=verbose,
    )
