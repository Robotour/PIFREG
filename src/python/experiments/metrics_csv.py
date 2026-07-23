"""CSV table for registration comparison: one file per random seed."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .stack_pairwise_metrics import METRIC_KEYS, evaluate_test_sessions_all_pairs

CSV_COLUMNS = [
    'method',
    'stage',
    'seed',
    *METRIC_KEYS,
    'num_test_sessions',
    'num_bands_mean',
    'num_pairs_mean',
    'image_size',
    'run_dir',
    'timestamp',
    'notes',
]


def default_metrics_csv_path(project_root: Path, seed: int) -> Path:
    return project_root / 'outputs' / 'metrics_tables' / f'seed_{seed}.csv'


def _read_existing_rows(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def _write_rows(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, '') for col in CSV_COLUMNS})


def _metrics_to_row_values(summary: Dict[str, float]) -> Dict[str, float]:
    return {k: summary.get(k, '') for k in METRIC_KEYS}


def ensure_unregistered_row(
    csv_path: Path,
    seed: int,
    test_folders: Sequence,
    image_size=(512, 512),
    max_sessions: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Ensure CSV exists and row 1 is unregistered (before registration) metrics.
    Recomputes only if the unregistered row is missing.
    """
    existing = _read_existing_rows(csv_path)
    for row in existing:
        if row.get('method') == 'unregistered' and row.get('stage') == 'before':
            if verbose:
                print(f'CSV already has unregistered row: {csv_path}', flush=True)
            return {'csv_path': str(csv_path), 'skipped': True}

    if verbose:
        print('Computing unregistered (all-pairs) metrics on test set ...', flush=True)
    eval_result = evaluate_test_sessions_all_pairs(
        test_folders,
        image_size=image_size,
        register_fn=None,
        max_sessions=max_sessions,
        verbose=verbose,
    )
    summary = eval_result['summary_before']
    new_row = {
        'method': 'unregistered',
        'stage': 'before',
        'seed': seed,
        **_metrics_to_row_values(summary),
        'num_test_sessions': eval_result['num_sessions'],
        'num_bands_mean': f'{eval_result["num_bands_mean"]:.2f}',
        'num_pairs_mean': f'{eval_result["num_pairs_mean"]:.1f}',
        'image_size': f'{image_size[0]},{image_size[1]}',
        'run_dir': '',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'notes': 'mean over all band pairs per session, then mean over test sessions',
    }

    # unregistered row always first
    others = [r for r in existing if r.get('method') != 'unregistered']
    _write_rows(csv_path, [new_row, *others])

    if verbose:
        print(f'Wrote unregistered row -> {csv_path}', flush=True)
    return {'csv_path': str(csv_path), 'eval': eval_result, 'row': new_row}


def append_method_row(
    csv_path: Path,
    method: str,
    seed: int,
    summary_after: Dict[str, float],
    test_folders: Sequence,
    image_size=(512, 512),
    run_dir: Optional[Path] = None,
    notes: str = '',
    max_sessions: Optional[int] = None,
    num_bands_mean: Optional[float] = None,
    num_pairs_mean: Optional[float] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Append (or replace) one method row after ensuring unregistered row exists."""
    ensure_unregistered_row(
        csv_path,
        seed,
        test_folders,
        image_size=image_size,
        max_sessions=max_sessions,
        verbose=False,
    )

    rows = _read_existing_rows(csv_path)
    new_row = {
        'method': method,
        'stage': 'after',
        'seed': seed,
        **_metrics_to_row_values(summary_after),
        'num_test_sessions': len(test_folders) if max_sessions is None else min(max_sessions, len(test_folders)),
        'num_bands_mean': f'{num_bands_mean:.2f}' if num_bands_mean is not None else '',
        'num_pairs_mean': f'{num_pairs_mean:.1f}' if num_pairs_mean is not None else '',
        'image_size': f'{image_size[0]},{image_size[1]}',
        'run_dir': str(run_dir.resolve()) if run_dir else '',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'notes': notes,
    }

    if overwrite:
        rows = [r for r in rows if r.get('method') != method]
    else:
        for r in rows:
            if r.get('method') == method and r.get('stage') == 'after':
                if verbose:
                    print(f'Row for method={method!r} already exists; use overwrite=True to replace.', flush=True)
                return csv_path

    rows.append(new_row)
    _write_rows(csv_path, rows)
    if verbose:
        print(f'Appended method={method!r} -> {csv_path}', flush=True)
    return csv_path
