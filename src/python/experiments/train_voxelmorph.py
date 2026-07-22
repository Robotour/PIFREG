#!/usr/bin/env python3
"""
VoxelMorph 无监督预训练 + 7:3 session 划分 + 测试集评估

默认数据: data/cut_images_all （每个子文件夹 = 一次拍摄 / 30 波段）

示例:
    python src/python/experiments/train_voxelmorph.py \\
        --data-dir data/cut_images_all \\
        --train-ratio 0.7 \\
        --epochs 300 \\
        --model-dir models/voxelmorph_cut_images_all
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.python.voxelmorph.training import (
    build_adjacent_band_pairs,
    discover_band_folders,
    evaluate_voxelmorph_pairs,
    save_split_manifest,
    split_folders_train_test,
    train_voxelmorph,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unsupervised VoxelMorph pre-training on HSI adjacent-band pairs',
    )
    parser.add_argument(
        '--data-dir',
        default='data/cut_images_all',
        help='Root folder containing one subfolder per HSI session',
    )
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256], metavar=('W', 'H'))
    parser.add_argument('--model-dir', default='models/voxelmorph_cut_images_all')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--steps-per-epoch', type=int, default=80)
    parser.add_argument('--val-steps', type=int, default=100, help='Random val pairs per validation')
    parser.add_argument('--val-interval', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda', dest='lamda', type=float, default=0.01)
    parser.add_argument('--int-steps', type=int, default=7)
    parser.add_argument('--int-downsize', type=int, default=2)
    parser.add_argument('--image-loss', choices=['ncc', 'mse'], default='ncc')
    parser.add_argument('--load-model', default=None, help='Optional checkpoint to resume from')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--eval-only', action='store_true', help='Skip training, only evaluate checkpoint')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint for --eval-only')
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = PROJECT_ROOT / args.data_dir
    image_size = tuple(args.image_size)
    model_dir = PROJECT_ROOT / args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    folders = discover_band_folders([data_root])
    train_folders, test_folders = split_folders_train_test(
        folders, train_ratio=args.train_ratio, seed=args.seed,
    )

    split_path = save_split_manifest(
        model_dir / 'split_manifest.json',
        train_folders,
        test_folders,
        args.train_ratio,
        args.seed,
    )

    train_pairs = build_adjacent_band_pairs(train_folders, image_size=image_size)
    test_pairs = build_adjacent_band_pairs(test_folders, image_size=image_size)
    inshape = train_pairs[0][0].shape

    config = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'data_dir': str(data_root),
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        'image_size': list(image_size),
        'num_sessions': len(folders),
        'num_train_sessions': len(train_folders),
        'num_test_sessions': len(test_folders),
        'num_train_pairs': len(train_pairs),
        'num_test_pairs': len(test_pairs),
        'epochs': args.epochs,
        'steps_per_epoch': args.steps_per_epoch,
        'image_loss': args.image_loss,
        'lamda': args.lamda,
        'inshape': list(inshape),
    }
    with open(model_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print('=' * 60)
    print('VoxelMorph unsupervised training')
    print(f'Data root: {data_root}')
    print(f'Sessions: {len(folders)}  train={len(train_folders)}  test={len(test_folders)}')
    print(f'Pairs: train={len(train_pairs)}  test={len(test_pairs)}')
    print(f'Split manifest: {split_path}')
    print(f'Model dir: {model_dir}')
    print('=' * 60)

    if args.eval_only:
        ckpt = args.checkpoint or str(model_dir / 'final.pt')
        from src.python.voxelmorph.networks import VxmDense

        model = VxmDense.load(ckpt, args.device)
        final_path = ckpt
    else:
        _, final_path = train_voxelmorph(
            train_pairs,
            model_dir=model_dir,
            inshape=inshape,
            device=args.device,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            lr=args.lr,
            image_loss=args.image_loss,
            lamda=args.lamda,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize,
            load_model=args.load_model,
            val_pairs=test_pairs,
            val_steps=args.val_steps,
            val_interval=args.val_interval,
        )
        from src.python.voxelmorph.networks import VxmDense

        model = VxmDense.load(final_path, args.device)

    print('\nEvaluating on held-out test pairs (all)...')
    test_result = evaluate_voxelmorph_pairs(
        model, test_pairs, device=args.device, max_pairs=None, verbose=True,
    )
    summary = test_result['summary']

    print('\nTest set summary (mean over all test pairs):')
    for key in sorted(summary):
        print(f'  {key}: {summary[key]:.6f}')

    eval_path = model_dir / 'test_metrics.json'
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'checkpoint': final_path,
                'num_test_pairs': test_result['num_pairs'],
                'summary': summary,
            },
            f,
            indent=2,
        )

    print(f'\nTraining/eval complete.')
    print(f'  checkpoint: {final_path}')
    print(f'  test metrics: {eval_path}')


if __name__ == '__main__':
    main()
