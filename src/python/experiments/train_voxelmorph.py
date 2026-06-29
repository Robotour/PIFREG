#!/usr/bin/env python3
"""
VoxelMorph 预训练脚本（P3：全数据集预训练官方 VxmDense）

用法示例:
    python src/python/experiments/train_voxelmorph.py \\
        --data-dirs "data/Test dataset" "data/cut_images_all" \\
        --image-size 256 256 \\
        --epochs 1500 \\
        --model-dir models/voxelmorph_hsi
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.python.voxelmorph.training import (
    build_adjacent_band_pairs,
    discover_band_folders,
    train_voxelmorph,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-train official VoxelMorph on HSI band pairs')
    parser.add_argument(
        '--data-dirs',
        nargs='+',
        default=['data/Test dataset', 'data/cut_images_all'],
        help='Root folders containing wavelength-named band images',
    )
    parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256], metavar=('W', 'H'))
    parser.add_argument('--model-dir', default='models/voxelmorph_hsi')
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--steps-per-epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda', dest='lamda', type=float, default=0.01)
    parser.add_argument('--int-steps', type=int, default=7)
    parser.add_argument('--int-downsize', type=int, default=2)
    parser.add_argument('--image-loss', choices=['ncc', 'mse'], default='ncc')
    parser.add_argument('--load-model', default=None, help='Optional checkpoint to resume from')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    data_roots = [PROJECT_ROOT / d for d in args.data_dirs]
    image_size = tuple(args.image_size)

    folders = discover_band_folders(data_roots)
    print(f'Found {len(folders)} band folders under data roots.')
    pairs = build_adjacent_band_pairs(folders, image_size=image_size)
    print(f'Built {len(pairs)} adjacent-band training pairs.')

    inshape = pairs[0][0].shape
    model_dir = PROJECT_ROOT / args.model_dir

    _, final_path = train_voxelmorph(
        pairs,
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
    )
    print(f'Training complete. Final model: {final_path}')


if __name__ == '__main__':
    main()
