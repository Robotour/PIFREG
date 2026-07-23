#!/usr/bin/env python3
"""
在 VoxelMorph 相同 test split 上评估经典非学习方法：Elastix / StackReg / KEREN。

输出目录（每次实验独立）:
  outputs/classical_baselines/{method}/{exp_name}_{timestamp}/
    config.json
    split_manifest.json
    test_metrics.json
    visualizations/   # --visualize

Linux 示例:
  python src/python/experiments/run_classical_baseline_eval.py \\
    --method stackreg_chain \\
    --exp-name stackreg_v1 \\
    --split-from-run-dir outputs/voxelmorph_runs/baseline/baseline_v1_YYYYMMDD_HHMMSS \\
    --visualize

  python src/python/experiments/run_classical_baseline_eval.py \\
    --method all \\
    --exp-name compare_v1 \\
    --data-dir data/cut_images_all \\
    --train-ratio 0.7 \\
    --seed 42 \\
    --max-sessions 5
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.python.experiments.experiment_data import load_hsi_stack
from src.python.experiments.experiment_recorder import save_rgb_outputs
from src.python.experiments.metrics_csv import (
    append_method_row,
    default_metrics_csv_path,
    ensure_unregistered_row,
    save_unregistered_metrics_report,
)
from src.python.experiments.session_outputs import print_metrics_summary, save_session_registration_outputs
from src.python.preprocessing import hsi_to_rgb
from src.python.registration.classical_stack import (
    CLASSICAL_METHODS,
    anchor_index,
    evaluate_classical_sessions,
    register_stack_classical_detailed,
)
from src.python.voxelmorph.training import (
    discover_band_folders,
    save_split_manifest,
    split_folders_train_test,
)

DEFAULT_SPECTRAL_PATH = PROJECT_ROOT / 'HSI2RGB20240517.xlsx'


def create_run_dir(method: str, exp_name: str) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = PROJECT_ROOT / 'outputs' / 'classical_baselines' / method / f'{exp_name}_{ts}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'visualizations').mkdir(exist_ok=True)
    (run_dir / 'session_exports').mkdir(exist_ok=True)
    return run_dir


def load_or_build_split(
    data_dir: Path,
    train_ratio: float,
    seed: int,
    split_from_run_dir: Path | None,
) -> tuple[list[Path], list[Path], Path | None]:
    if split_from_run_dir is not None:
        manifest = split_from_run_dir / 'split_manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError(f'Missing split manifest: {manifest}')
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        train_folders = [Path(p) for p in payload['train_sessions']]
        test_folders = [Path(p) for p in payload['test_sessions']]
        return train_folders, test_folders, manifest

    folders = discover_band_folders([data_dir])
    train_folders, test_folders = split_folders_train_test(
        folders, train_ratio=train_ratio, seed=seed,
    )
    return train_folders, test_folders, None


def export_test_sessions(
    run_dir: Path,
    method: str,
    test_folders: list[Path],
    image_size,
    descending: bool,
    spectral_path: Path,
    register_kwargs: dict,
    max_sessions: int | None = None,
    save_rgb: bool = True,
):
    folders = list(test_folders)
    if max_sessions is not None:
        folders = folders[:max_sessions]

    export_root = run_dir / 'session_exports'
    viz_root = run_dir / 'visualizations'
    index_rows = []
    for si, folder in enumerate(folders, start=1):
        slug = f'{si:02d}_{folder.name}'
        out_dir = export_root / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        bands_eq, bands_raw, band_files = load_hsi_stack(folder, image_size=image_size)
        detail = register_stack_classical_detailed(
            method,
            bands_eq,
            bands_raw=bands_raw,
            descending=descending,
            **register_kwargs,
        )
        save_session_registration_outputs(
            out_dir,
            bands_raw,
            detail['bands_raw_after'],
            band_files,
            chain_steps=detail.get('chain_steps'),
            elastix_fields=detail.get('elastix_fields'),
            transform_meta=detail.get('transform_meta'),
            anchor_idx=anchor_index(len(band_files), descending=descending),
            descending=descending,
        )
        print(f'  exported: {out_dir}', flush=True)

        if save_rgb:
            viz_dir = viz_root / slug
            viz_dir.mkdir(parents=True, exist_ok=True)
            rgb_before = hsi_to_rgb(bands_raw, spectral_data_path=str(spectral_path))
            rgb_after = hsi_to_rgb(detail['bands_raw_after'], spectral_data_path=str(spectral_path))
            rgb_paths = save_rgb_outputs(
                viz_dir, rgb_before, rgb_after, title_after=f'{method} registered',
            )
            index_rows.append({
                'session': str(folder),
                'export_dir': str(out_dir),
                'viz_dir': str(viz_dir),
                **{k: str(v) for k, v in rgb_paths.items()},
            })

    if save_rgb and index_rows:
        with open(viz_root / 'index.json', 'w', encoding='utf-8') as f:
            json.dump(index_rows, f, indent=2)


def visualize_sessions(
    run_dir: Path,
    method: str,
    test_folders: list[Path],
    image_size,
    descending: bool,
    spectral_path: Path,
    register_kwargs: dict,
    max_sessions: int | None = None,
):
    """Backward-compatible RGB-only wrapper."""
    export_test_sessions(
        run_dir, method, test_folders, image_size, descending,
        spectral_path, register_kwargs, max_sessions=max_sessions, save_rgb=True,
    )


def run_one_method(
    method: str,
    exp_name: str,
    test_folders: list[Path],
    train_folders: list[Path],
    image_size,
    descending: bool,
    max_sessions: int | None,
    register_kwargs: dict,
    visualize: bool,
    save_outputs: bool,
    spectral_path: Path,
    split_manifest_src: Path | None,
    train_ratio: float,
    seed: int,
    metrics_csv: Path | None,
    overwrite_csv_row: bool,
) -> Path:
    run_dir = create_run_dir(method, exp_name)
    print('=' * 60)
    print(f'Classical baseline eval: {method}')
    print(f'Run dir: {run_dir}')
    print(f'Test sessions: {len(test_folders)}')
    if max_sessions:
        print(f'Max sessions: {max_sessions}')
    print('=' * 60)

    if split_manifest_src is not None:
        shutil.copy2(split_manifest_src, run_dir / 'split_manifest.json')
    else:
        save_split_manifest(
            run_dir / 'split_manifest.json',
            train_folders,
            test_folders,
            train_ratio,
            seed,
        )

    config = {
        'method': method,
        'exp_name': exp_name,
        'image_size': list(image_size) if image_size else None,
        'descending_chain': descending,
        'max_sessions': max_sessions,
        'num_test_sessions': len(test_folders),
        'registration': register_kwargs,
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    t0 = time.time()
    stack_eval = evaluate_classical_sessions(
        method,
        test_folders,
        image_size=image_size,
        descending=descending,
        max_sessions=max_sessions,
        verbose=True,
        **register_kwargs,
    )
    elapsed = time.time() - t0

    payload = {
        'method': method,
        'elapsed_seconds': elapsed,
        'all_pairs_eval': stack_eval,
        'stack_eval': {
            'summary': stack_eval['summary_after'],
            'summary_before': stack_eval['summary_before'],
            'summary_delta': stack_eval['summary_delta'],
            'per_session': stack_eval['per_session'],
            'num_sessions': stack_eval['num_sessions'],
        },
    }
    with open(run_dir / 'test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    summary = stack_eval['summary_after']
    print('\n--- Registration metrics on test set ---')
    print_metrics_summary('Before (unregistered)', stack_eval['summary_before'])
    print_metrics_summary('After  (registered)  ', summary)
    print_metrics_summary('Delta (after-before) ', stack_eval['summary_delta'])
    print(f'Elapsed: {elapsed:.1f}s')
    print(f'Metrics: {run_dir / "test_metrics.json"}')

    if metrics_csv is not None:
        append_method_row(
            metrics_csv,
            method=method,
            seed=seed,
            summary_after=summary,
            test_folders=test_folders,
            image_size=image_size,
            run_dir=run_dir,
            notes='all C(N,2) band pairs mean; classical baseline',
            max_sessions=max_sessions,
            num_bands_mean=stack_eval.get('num_bands_mean'),
            num_pairs_mean=stack_eval.get('num_pairs_mean'),
            overwrite=overwrite_csv_row,
            verbose=True,
        )
        print(f'Metrics CSV: {metrics_csv}')

    if save_outputs or visualize:
        print('\nSaving per-session bands / flows / RGB ...')
        export_test_sessions(
            run_dir,
            method,
            test_folders,
            image_size,
            descending,
            spectral_path,
            register_kwargs,
            max_sessions=max_sessions,
            save_rgb=visualize or save_outputs,
        )

    return run_dir


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate classical HSI baselines on test split')
    p.add_argument(
        '--method',
        required=True,
        choices=[*CLASSICAL_METHODS, 'all'],
        help='Registration method; all runs every classical method sequentially',
    )
    p.add_argument('--exp-name', default='run')
    p.add_argument('--data-dir', default='data/cut_images_all')
    p.add_argument('--train-ratio', type=float, default=0.7)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument(
        '--split-from-run-dir',
        default=None,
        help='Reuse split_manifest.json from a VoxelMorph run (fair comparison)',
    )
    p.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=None,
        metavar=('W', 'H'),
        help='Optional resize; default: native resolution (no resize)',
    )
    p.add_argument('--ascending-chain', action='store_true', help='Anchor shortest wavelength (default: longest)')
    p.add_argument('--max-sessions', type=int, default=None, help='Limit test sessions (debug)')
    p.add_argument('--visualize', action='store_true', help='Save fake RGB compare images')
    p.add_argument('--save-outputs', action='store_true', default=True,
                   help='Save per-session bands/flows (default: on)')
    p.add_argument('--no-save-outputs', action='store_false', dest='save_outputs',
                   help='Skip saving band images and flow visualizations')
    p.add_argument('--metrics-csv', default=None, help='CSV path; default outputs/metrics_tables/seed_{seed}.csv')
    p.add_argument('--no-metrics-csv', action='store_true', help='Do not write comparison CSV')
    p.add_argument('--overwrite-csv-row', action='store_true', help='Replace existing method row in CSV')
    p.add_argument('--spectral-path', default=str(DEFAULT_SPECTRAL_PATH))
    p.add_argument('--elastix-epochs', type=int, default=20, help='Pairwise/groupwise Elastix iterations')
    p.add_argument('--elastix-spacing', type=int, default=20)
    p.add_argument('--stackreg-transform', default='bilinear',
                   choices=['translation', 'rigid', 'scaled_rotation', 'affine', 'bilinear'])
    p.add_argument('--elastix-verbose', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = PROJECT_ROOT / args.data_dir
    split_from = Path(args.split_from_run_dir) if args.split_from_run_dir else None
    if split_from is not None and not split_from.is_absolute():
        split_from = PROJECT_ROOT / split_from

    train_folders, test_folders, manifest_src = load_or_build_split(
        data_dir, args.train_ratio, args.seed, split_from,
    )
    descending = not args.ascending_chain
    image_size = tuple(args.image_size) if args.image_size else None
    spectral_path = Path(args.spectral_path)
    if not spectral_path.is_file():
        spectral_path = PROJECT_ROOT / args.spectral_path

    register_kwargs = {
        'epochs': args.elastix_epochs,
        'spacinginvoxels': args.elastix_spacing,
        'transform_type': args.stackreg_transform,
        # Named elastix_verbose so it does not collide with evaluate_*(verbose=...)
        'elastix_verbose': args.elastix_verbose,
    }

    metrics_csv = None
    if not args.no_metrics_csv:
        metrics_csv = (
            Path(args.metrics_csv)
            if args.metrics_csv
            else default_metrics_csv_path(PROJECT_ROOT, args.seed)
        )
        if not metrics_csv.is_absolute():
            metrics_csv = PROJECT_ROOT / metrics_csv

    if not args.no_metrics_csv:
        save_unregistered_metrics_report(
            PROJECT_ROOT,
            args.seed,
            test_folders,
            image_size,
            args.max_sessions,
        )
        ensure_unregistered_row(
            metrics_csv,
            args.seed,
            test_folders,
            image_size=image_size,
            max_sessions=args.max_sessions,
            verbose=True,
        )

    methods = list(CLASSICAL_METHODS) if args.method == 'all' else [args.method]
    run_dirs = []
    for method in methods:
        run_dirs.append(
            run_one_method(
                method,
                args.exp_name,
                test_folders,
                train_folders,
                image_size,
                descending,
                args.max_sessions,
                register_kwargs,
                args.visualize,
                args.save_outputs,
                spectral_path,
                manifest_src,
                args.train_ratio,
                args.seed,
                metrics_csv,
                args.overwrite_csv_row,
            )
        )

    if len(run_dirs) > 1:
        print('\nAll methods finished:')
        for rd in run_dirs:
            metrics = json.loads((rd / 'test_metrics.json').read_text(encoding='utf-8'))
            s = metrics['stack_eval']['summary']
            print(
                f'  {metrics["method"]:18s}  NCC {s["NCC"]:.4f}  '
                f'MSE {s["MSE"]:.6f}  -> {rd}',
            )


if __name__ == '__main__':
    main()
