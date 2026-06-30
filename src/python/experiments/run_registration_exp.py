#!/usr/bin/env python3
"""
高光谱图像波段配准实验脚本

本脚本演示如何使用重构后的配准工具包进行图像配准实验
"""

import sys
from pathlib import Path

# Allow running this file directly: add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
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
from src.python.experiments.experiment_data import load_pair_images
from src.python.experiments.experiment_recorder import _band_to_uint8

DATA_DIR = PROJECT_ROOT / "data"


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


def visualize_results(fixed_raw, moving_raw, warped_raw, method_name):
    """可视化原图强度配准结果。"""
    error_before = np.abs(
        _band_to_uint8(fixed_raw).astype(np.float32) - _band_to_uint8(moving_raw).astype(np.float32)
    )
    error_after = np.abs(
        _band_to_uint8(fixed_raw).astype(np.float32) - _band_to_uint8(warped_raw).astype(np.float32)
    )

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
    plt.imshow(error_before, cmap='jet', vmin=0, vmax=0.1)
    plt.title('Error Before')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(error_after, cmap='jet', vmin=0, vmax=0.1)
    plt.title('Error After')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def run_pifreg_experiment(fixed_path, moving_path, device='cuda'):
    """
    运行 PIFReg 金字塔实例流配准实验。

    默认：直方图匹配 + 多尺度 + 余弦退火学习率 + 早停（每尺度最多 3000 epoch）。
    """
    print("=" * 50)
    print("Running PIFReg (Pyramid Instance Flow Registration)")
    print("=" * 50)

    fixed, moving, fixed_raw, moving_raw = load_pair_images(
        fixed_path, moving_path, PROJECT_ROOT, DATA_DIR,
    )

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    warped_norm, flow = register_pifreg(
        fixed, moving,
        device=device,
        epochs=10000,
        lamda=0.01,
        image_loss='ncc',
        affine_init=True,
        histogram_match=True,
        multiscale=True,
        early_stop=True,
        patience=120,
        lr_schedule='cosine',
        lr_min=1e-6,
        fast_mode=True,
        return_flow=True,
    )
    warped_raw = _warp_raw_with_flow(moving_raw, flow, device)

    metrics = evaluate_registration(fixed, moving, warped_norm)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    visualize_results(fixed_raw, moving_raw, warped_raw, "PIFReg")

    return warped_raw, metrics


def run_voxelmorph_experiment(fixed_path, moving_path, device='cuda'):
    """向后兼容别名，内部调用 run_pifreg_experiment。"""
    return run_pifreg_experiment(fixed_path, moving_path, device=device)


def run_elastix_experiment(fixed_path, moving_path):
    """
    运行Elastix配准实验
    """
    print("=" * 50)
    print("Running Elastix Registration Experiment")
    print("=" * 50)

    fixed, moving, fixed_raw, moving_raw = load_pair_images(
        fixed_path, moving_path, PROJECT_ROOT, DATA_DIR,
    )

    warped_raw = register_elastix(fixed_raw, moving_raw, epochs=100, spacinginvoxels=20)

    metrics = evaluate_registration(fixed, moving, warped_raw)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    visualize_results(fixed_raw, moving_raw, warped_raw, "Elastix")

    return warped_raw, metrics


def run_stackreg_experiment(fixed_path, moving_path):
    """
    运行StackReg配准实验
    """
    print("=" * 50)
    print("Running StackReg Registration Experiment")
    print("=" * 50)

    fixed, moving, fixed_raw, moving_raw = load_pair_images(
        fixed_path, moving_path, PROJECT_ROOT, DATA_DIR,
    )

    warped_raw = register_stackreg(fixed_raw, moving_raw, transform_type='bilinear')

    metrics = evaluate_registration(fixed, moving, warped_raw)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    visualize_results(fixed_raw, moving_raw, warped_raw, "StackReg")

    return warped_raw, metrics


if __name__ == "__main__":
    # 示例路径（需要根据实际情况修改）
    fixed_image_path = "cut_images_all/2024-06-25_10-12-29-white/650.jpeg"
    moving_image_path = "cut_images_all/2024-06-25_10-12-29-white/639.jpeg"

    # 运行实验
    # 1. PIFReg (金字塔实例流配准)
    warped, metrics = run_pifreg_experiment(fixed_image_path, moving_image_path)

    # 2. Elastix (传统方法)
    # warped, metrics = run_elastix_experiment(fixed_image_path, moving_image_path)

    # 3. StackReg (传统方法)
    # warped, metrics = run_stackreg_experiment(fixed_image_path, moving_image_path)

    # print("请取消注释相应的实验代码来运行不同的配准方法")
