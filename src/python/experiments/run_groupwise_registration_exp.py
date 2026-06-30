#!/usr/bin/env python3
"""
高光谱群组配准实验（Elastix Groupwise / BSplineStackTransform）

从单个文件夹加载 30 个波段图像，整栈输入 pyelastix 群组配准，
用 hsi_to_rgb 合成配准前后彩色图，并输出评价指标。
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.python.metrics import compute_MI, compute_NMI, compute_NCC, compute_NTG
from src.python.preprocessing import hsi_to_rgb
from src.python.registration import register_elastix_groupwise

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SPECTRAL_PATH = PROJECT_ROOT / "HSI2RGB20240517.xlsx"
DEFAULT_STACK_DIR = DATA_DIR / "cut_images_all" / "2024-06-25_10-12-29-white"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "groupwise"


def resolve_path(path):
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()

    for base in (PROJECT_ROOT, DATA_DIR):
        resolved = base / candidate
        if resolved.exists():
            return resolved.resolve()

    raise FileNotFoundError(f"Path not found: {path}")


def sort_band_files(folder):
    """按文件名中的波长数值升序排列波段图像。"""
    files = list(Path(folder).glob("*.jpeg")) + list(Path(folder).glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No jpeg images found in {folder}")

    def band_key(path):
        stem = path.stem
        try:
            return int(stem)
        except ValueError:
            return stem

    return sorted(files, key=band_key)


def load_hsi_stack(folder, image_size=(256, 256)):
    """
    加载文件夹内全部波段，返回按波长排序的 float32 列表。

    每个波段独立做 Min-Max 归一化到 [0, 1]。
    """
    band_files = sort_band_files(folder)
    bands = []

    for path in band_files:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        img = img.astype(np.float32)
        if image_size is not None:
            img = cv2.resize(img, image_size)

        lo, hi = float(np.min(img)), float(np.max(img))
        if hi > lo:
            img = (img - lo) / (hi - lo)
        bands.append(img)

    return bands, band_files


def stack_to_list(stack):
    if isinstance(stack, list):
        return stack
    stack = np.asarray(stack)
    return [stack[i] for i in range(stack.shape[0])]


def evaluate_stack(bands, ref_idx):
    """以中间波段为参考，统计全栈相对参考波段的指标。"""
    ref = bands[ref_idx]
    metrics = {
        "ref_band_index": ref_idx,
        "per_band": [],
        "mean": {},
    }

    keys = ("MI", "NMI", "NCC", "NTG")
    sums = {k: 0.0 for k in keys}

    for i, band in enumerate(bands):
        if i == ref_idx:
            row = {
                "band_index": i,
                "MI": float(compute_MI(ref, ref)),
                "NMI": float(compute_NMI(ref, ref)),
                "NCC": float(compute_NCC(ref, ref)),
                "NTG": float(compute_NTG(ref, ref)),
            }
        else:
            row = {
                "band_index": i,
                "MI": float(compute_MI(ref, band)),
                "NMI": float(compute_NMI(ref, band)),
                "NCC": float(compute_NCC(ref, band)),
                "NTG": float(compute_NTG(ref, band)),
            }
        metrics["per_band"].append(row)
        for k in keys:
            sums[k] += row[k]

    n = len(bands)
    for k in keys:
        metrics["mean"][k] = sums[k] / n

    return metrics


def compare_metrics(before, after):
    """汇总配准前后均值指标变化。"""
    summary = {}
    for key in ("MI", "NMI", "NCC", "NTG"):
        b = before["mean"][key]
        a = after["mean"][key]
        summary[key] = {
            "before": b,
            "after": a,
            "delta": a - b,
        }
    return summary


def save_rgb_comparison(rgb_before, rgb_after, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb_before)
    axes[0].set_title("RGB Before Registration")
    axes[0].axis("off")

    axes[1].imshow(rgb_after)
    axes[1].set_title("RGB After Groupwise Registration")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_experiment(
    stack_dir,
    output_dir=None,
    image_size=(256, 256),
    epochs=80,
    spacinginvoxels=20,
    spectral_path=None,
    ref_band_idx=None,
    verbose=2,
    save_bands=True,
):
    stack_dir = resolve_path(stack_dir)
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR / stack_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    spectral_path = resolve_path(spectral_path or DEFAULT_SPECTRAL_PATH)

    print("=" * 60)
    print("Groupwise Elastix Registration Experiment")
    print("=" * 60)
    print(f"Input folder : {stack_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Image size   : {image_size}")
    print(f"Epochs/layer : {epochs}")
    print(f"Grid spacing : {spacinginvoxels} voxels")

    bands_before, band_files = load_hsi_stack(stack_dir, image_size=image_size)
    n_bands = len(bands_before)
    print(f"Loaded bands : {n_bands}")
    print(f"Wavelength range: {band_files[0].stem} - {band_files[-1].stem} nm")

    if ref_band_idx is None:
        ref_band_idx = n_bands // 2
    ref_band_idx = int(ref_band_idx)

    print("\n[1/4] Evaluating stack BEFORE registration ...")
    metrics_before = evaluate_stack(bands_before, ref_band_idx)

    print("[2/4] Running groupwise Elastix registration ...")
    bands_after, fields = register_elastix_groupwise(
        bands_before,
        epochs=epochs,
        spacinginvoxels=spacinginvoxels,
        verbose=verbose,
    )

    print("[3/4] Evaluating stack AFTER registration ...")
    metrics_after = evaluate_stack(bands_after, ref_band_idx)
    metrics_summary = compare_metrics(metrics_before, metrics_after)

    print("[4/4] Synthesizing RGB images ...")
    rgb_before = hsi_to_rgb(bands_before, spectral_data_path=str(spectral_path))
    rgb_after = hsi_to_rgb(bands_after, spectral_data_path=str(spectral_path))

    cv2.imwrite(str(output_dir / "rgb_before.png"), rgb_before)
    cv2.imwrite(str(output_dir / "rgb_after.png"), rgb_after)
    save_rgb_comparison(rgb_before, rgb_after, output_dir / "rgb_compare.png")

    if save_bands:
        reg_dir = output_dir / "registered_bands"
        reg_dir.mkdir(exist_ok=True)
        for path, band in zip(band_files, bands_after):
            out = (np.clip(band, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(reg_dir / path.name), out)

    report = {
        "stack_dir": str(stack_dir),
        "num_bands": n_bands,
        "wavelength_range_nm": [band_files[0].stem, band_files[-1].stem],
        "image_size": list(image_size),
        "epochs": epochs,
        "spacinginvoxels": spacinginvoxels,
        "ref_band_index": ref_band_idx,
        "ref_wavelength_nm": band_files[ref_band_idx].stem,
        "metrics_before_mean": metrics_before["mean"],
        "metrics_after_mean": metrics_after["mean"],
        "metrics_summary": metrics_summary,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Mean metrics vs reference band")
    print("=" * 60)
    print(f"{'Metric':<8} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 42)
    for key, vals in metrics_summary.items():
        print(
            f"{key:<8} {vals['before']:>10.4f} {vals['after']:>10.4f} {vals['delta']:>10.4f}"
        )

    print("\nOutputs:")
    print(f"  {output_dir / 'rgb_before.png'}")
    print(f"  {output_dir / 'rgb_after.png'}")
    print(f"  {output_dir / 'rgb_compare.png'}")
    print(f"  {output_dir / 'metrics.json'}")

    return bands_after, fields, report


def parse_args():
    parser = argparse.ArgumentParser(description="Run Elastix groupwise HSI registration")
    parser.add_argument(
        "--stack-dir",
        type=str,
        default=str(DEFAULT_STACK_DIR),
        help="包含 30 个波段 jpeg 的文件夹",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="结果输出目录，默认 outputs/groupwise/<文件夹名>",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("W", "H"),
        help="重采样大小，默认 256 256",
    )
    parser.add_argument("--epochs", type=int, default=80, help="每层最大迭代次数")
    parser.add_argument(
        "--spacing",
        type=int,
        default=24,
        dest="spacinginvoxels",
        help="B 样条网格间距（体素）",
    )
    parser.add_argument(
        "--spectral-path",
        type=str,
        default=str(DEFAULT_SPECTRAL_PATH),
        help="HSI2RGB 光谱响应 Excel 路径",
    )
    parser.add_argument(
        "--ref-band",
        type=int,
        default=None,
        help="参考波段索引，默认中间波段",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument(
        "--save-bands",
        action="store_true",
        help="是否保存配准后各波段图像",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        stack_dir=args.stack_dir,
        output_dir=args.output_dir,
        image_size=tuple(args.image_size),
        epochs=args.epochs,
        spacinginvoxels=args.spacinginvoxels,
        spectral_path=args.spectral_path,
        ref_band_idx=args.ref_band,
        verbose=args.verbose,
        save_bands=args.save_bands,
    )
