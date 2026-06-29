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

import cv2
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
from src.python.utils import normalize_image

DATA_DIR = PROJECT_ROOT / "data"


def resolve_image_path(path):
    """Resolve image path relative to project root or data directory."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate

    for base in (PROJECT_ROOT, DATA_DIR):
        resolved = base / candidate
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(
        f"Image not found: {path}\n"
        f"Tried: {candidate.resolve()}, {PROJECT_ROOT / candidate}, {DATA_DIR / candidate}"
    )


def load_images(fixed_path, moving_path, image_size=(512, 512)):
    """
    加载并预处理图像
    
    参数:
        fixed_path: 固定图像路径
        moving_path: 移动图像路径
        image_size: 目标图像大小
    
    返回:
        fixed_image, moving_image: 预处理后的图像
    """
    fixed_image = cv2.imread(str(resolve_image_path(fixed_path)), cv2.IMREAD_GRAYSCALE)
    moving_image = cv2.imread(str(resolve_image_path(moving_path)), cv2.IMREAD_GRAYSCALE)

    if fixed_image is None:
        raise ValueError(f"Failed to read fixed image: {fixed_path}")
    if moving_image is None:
        raise ValueError(f"Failed to read moving image: {moving_path}")

    fixed_image = fixed_image.astype(np.float32)
    moving_image = moving_image.astype(np.float32)

    fixed_image = cv2.resize(fixed_image, image_size).astype(np.float32)
    moving_image = cv2.resize(moving_image, image_size).astype(np.float32)

    # Min-Max归一化
    f_min, f_max = np.min(fixed_image), np.max(fixed_image)
    m_min, m_max = np.min(moving_image), np.max(moving_image)
    fixed_image = (fixed_image - f_min) / (f_max - f_min)
    moving_image = (moving_image - m_min) / (m_max - m_min)

    return fixed_image, moving_image


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


def visualize_results(fixed, moving, warped, method_name):
    """
    可视化配准结果
    
    参数:
        fixed: 固定图像
        moving: 移动图像
        warped: 变形图像
        method_name: 配准方法名称
    """
    error_before = abs(fixed - moving)
    error_after = abs(fixed - warped)

    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Registration Results - {method_name}')

    plt.subplot(1, 5, 1)
    plt.imshow(fixed, cmap='gray')
    plt.title('Fixed Image')
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(moving, cmap='gray')
    plt.title('Moving Image')
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(warped, cmap='gray')
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

    fixed, moving = load_images(fixed_path, moving_path)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    warped = register_pifreg(
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
    )

    metrics = evaluate_registration(fixed, moving, warped)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    visualize_results(fixed, moving, warped, "PIFReg")

    return warped, metrics


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

    # 加载图像
    fixed, moving = load_images(fixed_path, moving_path)

    # 配准
    warped = register_elastix(fixed, moving, epochs=100, spacinginvoxels=20)

    # 评估
    metrics = evaluate_registration(fixed, moving, warped)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # 可视化
    visualize_results(fixed, moving, warped, "Elastix")

    return warped, metrics


def run_stackreg_experiment(fixed_path, moving_path):
    """
    运行StackReg配准实验
    """
    print("=" * 50)
    print("Running StackReg Registration Experiment")
    print("=" * 50)

    # 加载图像
    fixed, moving = load_images(fixed_path, moving_path)

    # 配准
    warped = register_stackreg(fixed, moving, transform_type='bilinear')

    # 评估
    metrics = evaluate_registration(fixed, moving, warped)
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # 可视化
    visualize_results(fixed, moving, warped, "StackReg")

    return warped, metrics


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
