#!/usr/bin/env python3
"""
高光谱图像波段配准实验脚本

本脚本演示如何使用重构后的配准工具包进行图像配准实验
"""

import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

# Allow running this file directly: add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入配准包
from src.python.registration import (
    register_pifreg,
    register_elastix,
    register_elastix_edge,
    register_stackreg,
    register_keren
)
from src.python.metrics import compute_MI, compute_NMI, compute_NCC, compute_NTG
from src.python.preprocessing import hsi_to_rgb
from src.python.experiments.experiment_data import load_pair_images, pairwise_metrics_dict
from src.python.experiments.experiment_recorder import (
    _band_to_uint8,
    compute_pairwise_error_maps,
    create_run_dir,
    describe_pairwise_architecture,
    record_pairwise_experiment,
)

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "registration_pairwise"


def _warp_raw_with_flow(moving_raw, flow, device):
    from src.python.voxelmorph.layers import SpatialTransformer
    h, w = moving_raw.shape
    m = torch.tensor(moving_raw, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    flow_t = torch.tensor(flow, dtype=torch.float32, device=device).unsqueeze(0)
    transformer = SpatialTransformer((h, w)).to(device)
    with torch.no_grad():
        warped = transformer(m, flow_t)
    return warped.squeeze().cpu().numpy().astype(np.float32)


def evaluate_registration(fixed, moving, warped):
    """
    评估配准效果
    
    参数:
        fixed: 固定图像
        moving: 配准前的移动图像
        warped: 配准后的变形图像
    
    返回:
        metrics: 评价指标字典
    """
    metrics = {
        'MI_before': compute_MI(fixed, moving),
        'MI_after': compute_MI(fixed, warped),
        'NMI_before': compute_NMI(fixed, moving),
        'NMI_after': compute_NMI(fixed, warped),
        'NCC_before': compute_NCC(fixed, moving),
        'NCC_after': compute_NCC(fixed, warped),
        'NTG_before': compute_NTG(fixed, moving),
        'NTG_after': compute_NTG(fixed, warped),
    }
    return metrics


def _save_pairwise_experiment(
    run_dir,
    config,
    architecture_text,
    fixed,
    moving,
    warped,
    fixed_raw,
    moving_raw,
    warped_raw,
    elapsed_seconds,
    flow=None,
):
    pm = pairwise_metrics_dict(fixed, moving, warped)
    row_before = {"band_index": 0, **pm["before"]}
    row_after = {"band_index": 0, **pm["after"]}
    metrics_before = {"ref_band_index": 0, "per_band": [row_before], "mean": pm["before"]}
    metrics_after = {"ref_band_index": 0, "per_band": [row_after], "mean": pm["after"]}

    return record_pairwise_experiment(
        run_dir=run_dir,
        config=config,
        architecture_text=architecture_text,
        fixed_raw=fixed_raw,
        moving_raw=moving_raw,
        warped_raw=warped_raw,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        metrics_summary=pm["summary"],
        elapsed_seconds=elapsed_seconds,
        flow=flow,
    )


def visualize_results(fixed_raw, moving_raw, warped_raw, method_name, show=True):
    """可视化原图强度配准结果（弹窗预览，保存由 record_pairwise_experiment 负责）。"""
    error_before, error_after = compute_pairwise_error_maps(
        fixed_raw, moving_raw, warped_raw, normalized=True,
    )
    vmax = 0.1

    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Registration Results - {method_name}')

    plt.subplot(1, 5, 1)
    plt.imshow(_band_to_uint8(fixed_raw), cmap='gray')
    plt.title('Fixed Image')
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(_band_to_uint8(moving_raw), cmap='gray')
    plt.title('Moving Image')
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(_band_to_uint8(warped_raw), cmap='gray')
    plt.title('Warped Image')
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(error_before, cmap='jet', vmin=0, vmax=vmax)
    plt.title('Error Before')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(error_after, cmap='jet', vmin=0, vmax=vmax)
    plt.title('Error After')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


def run_pifreg_experiment(
    fixed_path,
    moving_path,
    device='cuda',
    multiscale_mode='sequential',
    epochs=10000,
    fast_mode=True,
    output_dir=None,
    exp_name='run',
    no_show=True,
):
    """
    运行 PIFReg 金字塔实例流配准实验。

    默认：直方图匹配 + 多尺度 + 余弦退火学习率 + 早停。
    multiscale_mode:
        - sequential: 各金字塔层级独立训练（原默认）
        - unrolled: 同一 U-Net，每个 epoch 内走完 128→256→512 再 backward
    """
    base_output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    run_dir = create_run_dir(base_output, exp_name=exp_name)

    print("=" * 50)
    print("Running PIFReg (Pyramid Instance Flow Registration)")
    print(f"multiscale_mode={multiscale_mode}")
    print(f"Run folder: {run_dir}")
    print("=" * 50)

    fixed, moving, fixed_raw, moving_raw = load_pair_images(
        fixed_path, moving_path, PROJECT_ROOT, DATA_DIR,
    )
    image_size = (fixed.shape[0], fixed.shape[1])

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = {
        "experiment": "registration_pairwise",
        "exp_name": exp_name,
        "method": "pifreg",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fixed_image": str(Path(fixed_path).resolve()),
        "moving_image": str(Path(moving_path).resolve()),
        "image_size": list(image_size),
        "device": str(device),
        "registration": {
            "epochs": epochs,
            "fast_mode": fast_mode,
            "multiscale": True,
            "multiscale_mode": multiscale_mode,
            "affine_init": True,
            "histogram_match": True,
            "early_stop": True,
            "patience": 120,
            "lr_schedule": "cosine",
        },
    }

    t0 = time.perf_counter()
    warped_norm, flow = register_pifreg(
        fixed, moving,
        device=device,
        epochs=epochs,
        lamda=0.01,
        image_loss='ncc',
        affine_init=True,
        histogram_match=True,
        multiscale=True,
        multiscale_mode=multiscale_mode,
        early_stop=True,
        patience=120,
        lr_schedule='cosine',
        lr_min=1e-6,
        fast_mode=fast_mode,
        return_flow=True,
    )
    elapsed = time.perf_counter() - t0
    warped_raw = _warp_raw_with_flow(moving_raw, flow, device)

    metrics = evaluate_registration(fixed, moving, warped_norm)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    manifest = _save_pairwise_experiment(
        run_dir=run_dir,
        config=config,
        architecture_text=describe_pairwise_architecture(
            image_size, fast_mode=fast_mode, multiscale=True, multiscale_mode=multiscale_mode,
        ),
        fixed=fixed,
        moving=moving,
        warped=warped_norm,
        fixed_raw=fixed_raw,
        moving_raw=moving_raw,
        warped_raw=warped_raw,
        elapsed_seconds=elapsed,
        flow=flow,
    )
    print(f"\nExperiment saved to: {run_dir}")

    visualize_results(
        fixed_raw, moving_raw, warped_raw,
        f"PIFReg ({multiscale_mode})",
        show=not no_show,
    )

    return warped_raw, metrics, manifest


def run_voxelmorph_experiment(fixed_path, moving_path, device='cuda'):
    """向后兼容别名，内部调用 run_pifreg_experiment。"""
    return run_pifreg_experiment(fixed_path, moving_path, device=device)


def run_elastix_experiment(
    fixed_path,
    moving_path,
    output_dir=None,
    exp_name='run',
    no_show=True,
):
    """
    运行Elastix配准实验
    """
    base_output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    run_dir = create_run_dir(base_output, exp_name=exp_name)

    print("=" * 50)
    print("Running Elastix Registration Experiment")
    print(f"Run folder: {run_dir}")
    print("=" * 50)

    fixed, moving, fixed_raw, moving_raw = load_pair_images(
        fixed_path, moving_path, PROJECT_ROOT, DATA_DIR,
    )

    config = {
        "experiment": "registration_pairwise",
        "exp_name": exp_name,
        "method": "elastix",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fixed_image": str(Path(fixed_path).resolve()),
        "moving_image": str(Path(moving_path).resolve()),
    }

    t0 = time.perf_counter()
    warped_raw = register_elastix(fixed_raw, moving_raw, epochs=100, spacinginvoxels=20)
    elapsed = time.perf_counter() - t0

    metrics = evaluate_registration(fixed, moving, warped_raw)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    manifest = _save_pairwise_experiment(
        run_dir=run_dir,
        config=config,
        architecture_text="Elastix pairwise registration",
        fixed=fixed,
        moving=moving,
        warped=warped_raw,
        fixed_raw=fixed_raw,
        moving_raw=moving_raw,
        warped_raw=warped_raw,
        elapsed_seconds=elapsed,
    )
    print(f"\nExperiment saved to: {run_dir}")

    visualize_results(fixed_raw, moving_raw, warped_raw, "Elastix", show=not no_show)

    return warped_raw, metrics, manifest


def run_stackreg_experiment(
    fixed_path,
    moving_path,
    output_dir=None,
    exp_name='run',
    no_show=True,
):
    """
    运行StackReg配准实验
    """
    base_output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    run_dir = create_run_dir(base_output, exp_name=exp_name)

    print("=" * 50)
    print("Running StackReg Registration Experiment")
    print(f"Run folder: {run_dir}")
    print("=" * 50)

    fixed, moving, fixed_raw, moving_raw = load_pair_images(
        fixed_path, moving_path, PROJECT_ROOT, DATA_DIR,
    )

    config = {
        "experiment": "registration_pairwise",
        "exp_name": exp_name,
        "method": "stackreg",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fixed_image": str(Path(fixed_path).resolve()),
        "moving_image": str(Path(moving_path).resolve()),
    }

    t0 = time.perf_counter()
    warped_raw = register_stackreg(fixed_raw, moving_raw, transform_type='bilinear')
    elapsed = time.perf_counter() - t0

    metrics = evaluate_registration(fixed, moving, warped_raw)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    manifest = _save_pairwise_experiment(
        run_dir=run_dir,
        config=config,
        architecture_text="StackReg pairwise registration (bilinear)",
        fixed=fixed,
        moving=moving,
        warped=warped_raw,
        fixed_raw=fixed_raw,
        moving_raw=moving_raw,
        warped_raw=warped_raw,
        elapsed_seconds=elapsed,
    )
    print(f"\nExperiment saved to: {run_dir}")

    visualize_results(fixed_raw, moving_raw, warped_raw, "StackReg", show=not no_show)

    return warped_raw, metrics, manifest


def _parse_args():
    parser = argparse.ArgumentParser(description='高光谱单对配准实验')
    parser.add_argument('--fixed', default='cut_images_all/2024-06-25_10-12-29-white/650.jpeg')
    parser.add_argument('--moving', default='cut_images_all/2024-06-25_10-12-29-white/639.jpeg')
    parser.add_argument('--output-dir', type=str, default=None, help='实验输出根目录')
    parser.add_argument('--exp-name', type=str, default='run', help='实验名称后缀')
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--multiscale-mode',
        choices=['sequential', 'unrolled'],
        default='sequential',
        help='PIFReg 多尺度策略：sequential=逐层独立训练;unrolled=每 epoch 内 unroll 金字塔',
    )
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--no-fast-mode', action='store_true', help='关闭 fast_mode 加速配置')
    parser.add_argument(
        '--method',
        choices=['pifreg', 'elastix', 'stackreg'],
        default='pifreg',
    )
    parser.add_argument('--show', action='store_true', help='弹窗显示可视化（默认只保存到 runs 目录）')
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    common = dict(
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        no_show=not args.show,
    )

    if args.method == 'pifreg':
        warped, metrics, manifest = run_pifreg_experiment(
            args.fixed,
            args.moving,
            device=args.device,
            multiscale_mode=args.multiscale_mode,
            epochs=args.epochs,
            fast_mode=not args.no_fast_mode,
            **common,
        )
    elif args.method == 'elastix':
        warped, metrics, manifest = run_elastix_experiment(
            args.fixed, args.moving, **common,
        )
    elif args.method == 'stackreg':
        warped, metrics, manifest = run_stackreg_experiment(
            args.fixed, args.moving, **common,
        )
