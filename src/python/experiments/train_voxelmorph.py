#!/usr/bin/env python3
"""
VoxelMorph HSI 实验入口：独立 run 目录 + best.pt + 全测试集评估与可视化

方法 (--method):
  baseline       — 随机相邻波段对（原版 VoxelMorph 预训练）
  stack_spatial  — session 子链 + 空间加权（你的改进方法）

每次实验输出到:
  outputs/voxelmorph_runs/{method}/{exp_name}_{timestamp}/

Linux 示例见 .cursor/skills/voxelmorph-hsi-training/SKILL.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.python.voxelmorph.experiment import run_full_experiment


def parse_args():
    p = argparse.ArgumentParser(description='VoxelMorph HSI train/eval/visualize experiment')
    p.add_argument(
        '--method',
        choices=['baseline', 'stack_spatial'],
        required=True,
        help='baseline=pairwise; stack_spatial=your proposed method',
    )
    p.add_argument('--exp-name', default='run', help='Experiment label in run folder name')
    p.add_argument('--data-dir', default='data/cut_images_all')
    p.add_argument('--train-ratio', type=float, default=0.7)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image-size', type=int, nargs=2, default=[512, 512], metavar=('W', 'H'))
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--steps-per-epoch', type=int, default=80)
    p.add_argument('--val-steps', type=int, default=100, help='Baseline: random val pairs')
    p.add_argument('--val-session-steps', type=int, default=15, help='Stack: val sessions per epoch')
    p.add_argument('--val-interval', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lambda', dest='lamda', type=float, default=0.01)
    p.add_argument('--int-steps', type=int, default=7)
    p.add_argument('--int-downsize', type=int, default=2)
    p.add_argument('--image-loss', choices=['ncc', 'mse'], default=None,
                   help='Default: ncc for baseline, mse for stack_spatial')
    p.add_argument('--subchain-len', type=int, default=6)
    p.add_argument('--center-floor', type=float, default=0.35)
    p.add_argument('--edge-gain', type=float, default=1.0)
    p.add_argument('--ascending-chain', action='store_true')
    p.add_argument('--smooth-flow-sigma', type=float, default=1.5)
    p.add_argument('--load-model', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--run-dir', default=None, help='Existing run dir for --eval-only')
    p.add_argument('--metrics-csv', default=None, help='Comparison CSV; default outputs/metrics_tables/seed_{seed}.csv')
    p.add_argument('--no-metrics-csv', action='store_true')
    p.add_argument('--overwrite-csv-row', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    image_loss = args.image_loss
    if image_loss is None:
        image_loss = 'ncc' if args.method == 'baseline' else 'mse'

    train_kwargs = {
        'epochs': args.epochs,
        'steps_per_epoch': args.steps_per_epoch,
        'val_steps': args.val_steps,
        'val_session_steps': args.val_session_steps,
        'val_interval': args.val_interval,
        'lr': args.lr,
        'lamda': args.lamda,
        'int_steps': args.int_steps,
        'int_downsize': args.int_downsize,
        'image_loss': image_loss,
        'subchain_len': args.subchain_len,
        'center_floor': args.center_floor,
        'edge_gain': args.edge_gain,
        'chain_descending': not args.ascending_chain,
        'load_model': args.load_model,
        'device': args.device,
    }

    metrics_csv = None
    if not args.no_metrics_csv and args.metrics_csv:
        metrics_csv = Path(args.metrics_csv)
        if not metrics_csv.is_absolute():
            metrics_csv = PROJECT_ROOT / metrics_csv

    run_dir, summary = run_full_experiment(
        project_root=PROJECT_ROOT,
        method=args.method,
        exp_name=args.exp_name,
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        image_size=tuple(args.image_size),
        train_kwargs=train_kwargs,
        chain_descending=not args.ascending_chain,
        smooth_flow_sigma=args.smooth_flow_sigma,
        skip_train=args.eval_only,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        metrics_csv=metrics_csv,
        write_metrics_csv=not args.no_metrics_csv,
        overwrite_csv_row=args.overwrite_csv_row,
    )

    print('\n' + '=' * 60)
    print('Experiment complete')
    print(f'  run_dir         : {run_dir}')
    print(f'  best checkpoint : {summary["best_checkpoint"]}')
    print(f'  visualizations  : {run_dir / "visualizations"}')
    print(f'  test_metrics    : {run_dir / "test_metrics.json"}')
    if not args.no_metrics_csv:
        from src.python.experiments.metrics_csv import default_metrics_csv_path
        csv_path = metrics_csv or default_metrics_csv_path(PROJECT_ROOT, args.seed)
        print(f'  metrics_csv     : {csv_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
