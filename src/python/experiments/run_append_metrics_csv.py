#!/usr/bin/env python3
"""
Initialize or append rows to per-seed metrics CSV (all-band pairwise averages).

Each CSV file: outputs/metrics_tables/seed_{seed}.csv
  Row 1: unregistered (before registration)
  Row 2+: one row per method (after registration)

Use --init-only to create row 1 without running any registration.
Use --from-run-dir to append from an existing test_metrics.json (VoxelMorph or classical).

Linux examples:
  # Create CSV row 1 only (unregistered baseline for seed=42)
  python src/python/experiments/run_append_metrics_csv.py \\
    --init-only --seed 42 --data-dir data/cut_images_all

  # Append your improved method from a finished VoxelMorph run
  python src/python/experiments/run_append_metrics_csv.py \\
    --method voxelmorph_stack_spatial \\
    --from-run-dir outputs/voxelmorph_runs/stack_spatial/stack_spatial_v1_YYYYMMDD_HHMMSS \\
    --seed 42 --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.python.experiments.experiment_data import DEFAULT_IMAGE_SIZE
from src.python.experiments.metrics_csv import (
    append_method_row,
    default_metrics_csv_path,
    ensure_unregistered_row,
)
from src.python.voxelmorph.training import (
    discover_band_folders,
    split_folders_train_test,
)


def load_test_folders(args) -> list[Path]:
    if args.from_run_dir:
        run_dir = Path(args.from_run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        manifest = run_dir / 'split_manifest.json'
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        return [Path(p) for p in payload['test_sessions']]

    data_dir = PROJECT_ROOT / args.data_dir
    folders = discover_band_folders([data_dir])
    _, test_folders = split_folders_train_test(
        folders, train_ratio=args.train_ratio, seed=args.seed,
    )
    return test_folders


def parse_args():
    p = argparse.ArgumentParser(description='Init/append per-seed metrics comparison CSV')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-dir', default='data/cut_images_all')
    p.add_argument('--train-ratio', type=float, default=0.7)
    p.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=list(DEFAULT_IMAGE_SIZE),
        metavar=('W', 'H'),
        help='Resize all bands (default: 512 512)',
    )
    p.add_argument('--metrics-csv', default=None)
    p.add_argument('--init-only', action='store_true', help='Only write unregistered row (row 1)')
    p.add_argument('--method', default=None, help='Method name for CSV row (required unless --init-only)')
    p.add_argument('--from-run-dir', default=None, help='Load all_pairs_eval from existing run dir')
    p.add_argument('--overwrite', action='store_true', help='Replace existing method row')
    p.add_argument('--max-sessions', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    image_size = tuple(args.image_size)
    csv_path = (
        Path(args.metrics_csv)
        if args.metrics_csv
        else default_metrics_csv_path(PROJECT_ROOT, args.seed)
    )
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    test_folders = load_test_folders(args)
    ensure_unregistered_row(
        csv_path,
        args.seed,
        test_folders,
        image_size=image_size,
        max_sessions=args.max_sessions,
        verbose=True,
    )

    if args.init_only:
        print(f'Done. Unregistered row ready in {csv_path}')
        return

    if not args.method:
        raise SystemExit('--method is required unless --init-only')

    if args.from_run_dir:
        run_dir = Path(args.from_run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        metrics_path = run_dir / 'test_metrics.json'
        payload = json.loads(metrics_path.read_text(encoding='utf-8'))
        if 'all_pairs_eval' in payload:
            all_pairs = payload['all_pairs_eval']
        else:
            raise KeyError(f'{metrics_path} missing all_pairs_eval; re-run evaluation with updated scripts')
        summary_after = all_pairs['summary_after']
        num_bands_mean = all_pairs.get('num_bands_mean')
        num_pairs_mean = all_pairs.get('num_pairs_mean')
        notes = f'imported from {run_dir.name}'
    else:
        raise SystemExit('Provide --from-run-dir to append a method row, or run classical/VoxelMorph eval scripts.')

    append_method_row(
        csv_path,
        method=args.method,
        seed=args.seed,
        summary_after=summary_after,
        test_folders=test_folders,
        image_size=image_size,
        run_dir=Path(args.from_run_dir) if args.from_run_dir else None,
        notes=notes,
        max_sessions=args.max_sessions,
        num_bands_mean=num_bands_mean,
        num_pairs_mean=num_pairs_mean,
        overwrite=args.overwrite,
        verbose=True,
    )
    print(f'Updated CSV: {csv_path}')


if __name__ == '__main__':
    main()
