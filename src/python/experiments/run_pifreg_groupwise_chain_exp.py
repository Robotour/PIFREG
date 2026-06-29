#!/usr/bin/env python3
"""
PIFReg 链式群组配准实验

从高波长向低波长依次两两 PIFReg（690 → 680 → 670 → …），
上一张配准结果作为下一张 fixed；以相邻波段 NCC 为主要评价指标。
评价/可视化流程与 run_groupwise_registration_exp.py 一致。
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.python.metrics import compute_MI, compute_NMI, compute_NCC, compute_NTG
from src.python.preprocessing import hsi_to_rgb
from src.python.registration.pif_groupwise_chain import (
    evaluate_chain_pairwise_ncc,
    register_pifreg_chain,
)

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SPECTRAL_PATH = PROJECT_ROOT / "HSI2RGB20240517.xlsx"
DEFAULT_STACK_DIR = (
    PROJECT_ROOT / "All code" / "cut_images_all" / "2024-06-25_10-12-29-white"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pifreg_groupwise_chain"


def resolve_path(path):
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    for base in (PROJECT_ROOT, DATA_DIR, PROJECT_ROOT / "All code"):
        resolved = base / candidate
        if resolved.exists():
            return resolved.resolve()
    raise FileNotFoundError(f"Path not found: {path}")


def sort_band_files(folder):
    files = list(Path(folder).glob("*.jpeg")) + list(Path(folder).glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No jpeg images found in {folder}")

    def band_key(path):
        try:
            return int(path.stem)
        except ValueError:
            return path.stem

    return sorted(files, key=band_key)


def load_hsi_stack(folder, image_size=(256, 256)):
    band_files = sort_band_files(folder)
    bands, wavelengths = [], []
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
        wavelengths.append(path.stem)
    return bands, band_files, wavelengths


def evaluate_stack(bands, ref_idx):
    ref = bands[ref_idx]
    metrics = {"ref_band_index": ref_idx, "per_band": [], "mean": {}}
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
    summary = {}
    for key in ("MI", "NMI", "NCC", "NTG"):
        b, a = before["mean"][key], after["mean"][key]
        summary[key] = {"before": b, "after": a, "delta": a - b}
    return summary


def save_rgb_comparison(rgb_before, rgb_after, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb_before)
    axes[0].set_title("RGB Before Registration")
    axes[0].axis("off")
    axes[1].imshow(rgb_after)
    axes[1].set_title("RGB After PIFReg Chain (690→400 nm)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_experiment(
    stack_dir,
    output_dir=None,
    image_size=(512, 512),
    device="cuda",
    fast_mode=True,
    eval_ref_band_idx=None,
    spectral_path=None,
    save_bands=True,
):
    stack_dir = resolve_path(stack_dir)
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR / stack_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    spectral_path = resolve_path(spectral_path or DEFAULT_SPECTRAL_PATH)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("PIFReg Chain Groupwise Experiment")
    print("=" * 60)
    print("Pairwise PIFReg: 690 → 680 → 670 → … (prev warped = next fixed)")
    print("Primary metric: mean adjacent-band NCC along chain")
    print("-" * 60)
    print(f"Input folder : {stack_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Image size   : {image_size}")
    print(f"Device       : {device}")

    bands_before, band_files, wavelengths = load_hsi_stack(stack_dir, image_size=image_size)
    n_bands = len(bands_before)
    print(f"Loaded bands : {n_bands}")
    print(f"Wavelength range: {wavelengths[0]} - {wavelengths[-1]} nm (ascending on disk)")

    chain_before = evaluate_chain_pairwise_ncc(bands_before, wavelengths, descending=True)
    print(f"Chain mean NCC BEFORE: {chain_before['mean_NCC']:.4f}")

    if eval_ref_band_idx is None:
        eval_ref_band_idx = n_bands // 2

    print("\n[1/4] Evaluating stack BEFORE registration (ref-band metrics) ...")
    metrics_before = evaluate_stack(bands_before, eval_ref_band_idx)

    print("[2/4] Running chain PIFReg registration (descending wavelength) ...")
    t0 = time.perf_counter()
    bands_after, reg_info = register_pifreg_chain(
        bands_before,
        device=str(device),
        descending=True,
        wavelengths_nm=wavelengths,
        fast_mode=fast_mode,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"Registration finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    chain_after = reg_info["chain_pairwise_ncc"]
    print(f"Chain mean NCC AFTER : {chain_after['mean_NCC']:.4f}")

    print("[3/4] Evaluating stack AFTER registration ...")
    metrics_after = evaluate_stack(bands_after, eval_ref_band_idx)
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
            cv2.imwrite(str(reg_dir / path.name), (np.clip(band, 0, 1) * 255).astype(np.uint8))

    report = {
        "stack_dir": str(stack_dir),
        "num_bands": n_bands,
        "wavelength_range_nm": [wavelengths[0], wavelengths[-1]],
        "image_size": list(image_size),
        "registration_info": reg_info,
        "chain_ncc_before": chain_before,
        "chain_ncc_after": chain_after,
        "elapsed_seconds": elapsed,
        "eval_ref_band_index": eval_ref_band_idx,
        "metrics_before_mean": metrics_before["mean"],
        "metrics_after_mean": metrics_after["mean"],
        "metrics_summary": metrics_summary,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Chain pairwise NCC (690↔680, 680↔670, …)")
    print("=" * 60)
    print(f"  Before: {chain_before['mean_NCC']:.4f}")
    print(f"  After : {chain_after['mean_NCC']:.4f}")
    print(f"  Delta : {chain_after['mean_NCC'] - chain_before['mean_NCC']:.4f}")

    print("\nMean metrics vs reference band")
    print(f"{'Metric':<8} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 42)
    for key, vals in metrics_summary.items():
        print(f"{key:<8} {vals['before']:>10.4f} {vals['after']:>10.4f} {vals['delta']:>10.4f}")

    print("\nOutputs:")
    for name in ("rgb_before.png", "rgb_after.png", "rgb_compare.png", "metrics.json"):
        print(f"  {output_dir / name}")

    return bands_after, reg_info, report


def parse_args():
    p = argparse.ArgumentParser(description="PIFReg descending chain groupwise registration")
    p.add_argument("--stack-dir", type=str, default=str(DEFAULT_STACK_DIR))
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--image-size", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-fast-mode", action="store_true")
    p.add_argument("--eval-ref-band", type=int, default=None)
    p.add_argument("--spectral-path", type=str, default=str(DEFAULT_SPECTRAL_PATH))
    p.add_argument("--save-bands", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        stack_dir=args.stack_dir,
        output_dir=args.output_dir,
        image_size=tuple(args.image_size),
        device=args.device,
        fast_mode=not args.no_fast_mode,
        eval_ref_band_idx=args.eval_ref_band,
        spectral_path=args.spectral_path,
        save_bands=args.save_bands,
    )
