"""All-band pairwise metrics: C(N,2) pairs averaged per session, then over test sessions."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.python.experiments.experiment_data import load_hsi_stack
from src.python.metrics.evaluation import (
    compute_MI,
    compute_MSE,
    compute_NCC,
    compute_NMI,
    compute_NTG,
)

METRIC_KEYS = ('MI', 'NMI', 'NCC', 'NTG', 'MSE')

_COMPUTE_FNS = {
    'MI': compute_MI,
    'NMI': compute_NMI,
    'NCC': compute_NCC,
    'NTG': compute_NTG,
    'MSE': compute_MSE,
}


def compute_stack_all_pairs_mean(bands: Sequence[np.ndarray]) -> Dict[str, float]:
    """
    For N bands, compute each metric on every unordered pair (i, j), i < j,
    then return the mean over all C(N, 2) pairs.
    """
    bands = [np.asarray(b, dtype=np.float32) for b in bands]
    n = len(bands)
    if n < 2:
        return {k: float('nan') for k in METRIC_KEYS}

    sums = {k: 0.0 for k in METRIC_KEYS}
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for key in METRIC_KEYS:
                sums[key] += float(_COMPUTE_FNS[key](bands[i], bands[j]))
            count += 1
    return {k: sums[k] / count for k in METRIC_KEYS}


def evaluate_test_sessions_all_pairs(
    test_folders: Sequence,
    image_size=(512, 512),
    register_fn: Optional[Callable[[List[np.ndarray]], List[np.ndarray]]] = None,
    max_sessions: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate all test sessions with all-band pairwise mean metrics.

    register_fn=None  -> unregistered stacks (before registration).
    register_fn(bands) -> registered stacks; metrics computed on warped bands.
    """
    eval_folders = list(test_folders)
    if max_sessions is not None and max_sessions < len(eval_folders):
        rng = np.random.default_rng(0)
        eval_folders = list(rng.choice(eval_folders, size=max_sessions, replace=False))

    rows = []
    for idx, folder in enumerate(eval_folders):
        bands, _, _ = load_hsi_stack(folder, image_size=image_size)
        before = compute_stack_all_pairs_mean(bands)
        if register_fn is None:
            after = before
        else:
            bands_after = register_fn(bands)
            after = compute_stack_all_pairs_mean(bands_after)

        row = {
            'session': str(folder),
            'num_bands': len(bands),
            'num_pairs': len(bands) * (len(bands) - 1) // 2,
            'before': before,
            'after': after,
        }
        for key in METRIC_KEYS:
            row[f'{key}_before'] = before[key]
            row[f'{key}_after'] = after[key]
            row[f'{key}_delta'] = after[key] - before[key]
        rows.append(row)

        if verbose:
            print(
                f'  session {idx + 1}/{len(eval_folders)}  '
                f'NCC {before["NCC"]:.4f}->{after["NCC"]:.4f}  '
                f'MSE {before["MSE"]:.6f}->{after["MSE"]:.6f}',
                flush=True,
            )

    summary_before = {k: float(np.mean([r['before'][k] for r in rows])) for k in METRIC_KEYS}
    summary_after = {k: float(np.mean([r['after'][k] for r in rows])) for k in METRIC_KEYS}
    summary_delta = {k: summary_after[k] - summary_before[k] for k in METRIC_KEYS}

    return {
        'metric_definition': 'mean over all C(N,2) band pairs per session, then mean over sessions',
        'metric_keys': list(METRIC_KEYS),
        'summary_before': summary_before,
        'summary_after': summary_after,
        'summary_delta': summary_delta,
        'num_sessions': len(rows),
        'num_bands_mean': float(np.mean([r['num_bands'] for r in rows])),
        'num_pairs_mean': float(np.mean([r['num_pairs'] for r in rows])),
        'per_session': rows,
    }
