#!/usr/bin/env python3
"""
PIFReg Per-Band StackFlow 群组配准实验

一次性联合优化 N-1 个位移场，金字塔逐级至全分辨率。
每次运行自动在 outputs/.../runs/<timestamp>_<exp_name>/ 下保存完整实验记录。
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from src.python.experiments.experiment_recorder import (
    create_run_dir,
    describe_stackflow_architecture,
    record_stackflow_experiment,
)
from src.python.metrics import compute_MI, compute_NMI, compute_NCC, compute_NTG
from src.python.preprocessing import hsi_to_rgb
from src.python.registration.pif_groupwise_stackflow import (
    DEFAULT_EPOCHS_PER_LEVEL,
    DEFAULT_PATIENCE_PER_LEVEL,
    DEFAULT_PYRAMID_SIZES,
    DEFAULT_SPECTRAL_ENC_CHANNELS,
    DEFAULT_SPECTRAL_ENC_KERNEL,
    FEATURE_MODE_MEAN_ANCHOR,
    FEATURE_MODE_SPECTRAL_ENCODER,
    register_pifreg_groupwise_stackflow,
    warp_bands_with_flow_stack,
)

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SPECTRAL_PATH = PROJECT_ROOT / "HSI2RGB20240517.xlsx"
DEFAULT_STACK_DIR = (
    PROJECT_ROOT / "All code" / "cut_images_all" / "2024-06-25_10-12-29-white"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pifreg_groupwise_stackflow"


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


def load_hsi_stack(folder, image_size=(512, 512)):
    """加载高光谱栈：返回归一化波段（用于配准/指标）与原图强度（用于保存）。"""
    band_files = sort_band_files(folder)
    bands_norm = []
    bands_raw = []
    for path in band_files:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = img.astype(np.float32)
        if image_size is not None:
            img = cv2.resize(img, image_size)
        bands_raw.append(img.copy())
        lo, hi = float(np.min(img)), float(np.max(img))
        if hi > lo:
            img = (img - lo) / (hi - lo)
        bands_norm.append(img)
    return bands_norm, bands_raw, band_files


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


def build_experiment_config(
    stack_dir,
    image_size,
    device,
    anchor_band,
    eval_ref_band_idx,
    spectral_path,
    registration_kwargs,
    n_bands,
    band_files,
    exp_name,
):
    return {
        "experiment": "pifreg_groupwise_stackflow",
        "exp_name": exp_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stack_dir": str(stack_dir),
        "spectral_path": str(spectral_path),
        "num_bands": n_bands,
        "wavelength_range_nm": [band_files[0].stem, band_files[-1].stem],
        "image_size": list(image_size),
        "device": str(device),
        "anchor_band": anchor_band,
        "eval_ref_band": eval_ref_band_idx,
        "registration": registration_kwargs,
    }


def run_experiment(
    stack_dir,
    output_dir=None,
    exp_name="run",
    image_size=(512, 512),
    device="cuda",
    anchor_band=-1,
    fast_mode=True,
    eval_ref_band_idx=None,
    spectral_path=None,
    save_before_bands=True,
    # --- 配准超参数（可 CLI 覆盖）---
    pyramid_sizes=None,
    epochs_per_level=None,
    patience_per_level=None,
    lr=2e-4,
    lamda=0.005,
    ncc_weight=1.0,
    int_steps=3,
    int_downsize=2,
    early_stop=True,
    min_delta=1e-5,
    lr_schedule="cosine",
    lr_min=1e-6,
    feature_mode=FEATURE_MODE_SPECTRAL_ENCODER,
    spectral_enc_channels=DEFAULT_SPECTRAL_ENC_CHANNELS,
    spectral_enc_kernel=DEFAULT_SPECTRAL_ENC_KERNEL,
):
    stack_dir = resolve_path(stack_dir)
    base_output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    run_dir = create_run_dir(base_output, exp_name=exp_name)
    spectral_path = resolve_path(spectral_path or DEFAULT_SPECTRAL_PATH)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    pyramid_sizes = tuple(pyramid_sizes or DEFAULT_PYRAMID_SIZES)
    epochs_per_level = list(epochs_per_level or DEFAULT_EPOCHS_PER_LEVEL)
    patience_per_level = list(patience_per_level or DEFAULT_PATIENCE_PER_LEVEL)

    print("=" * 60)
    print("PIFReg StackFlow Groupwise Experiment")
    print("=" * 60)
    print("Joint optimization of N-1 per-band displacement fields")
    print(f"Pyramid sizes (requested): {list(pyramid_sizes)}")
    print("-" * 60)
    print(f"Input folder : {stack_dir}")
    print(f"Run folder   : {run_dir}")
    print(f"Exp name     : {exp_name}")
    print(f"Device       : {device}")

    bands_norm, bands_raw, band_files = load_hsi_stack(stack_dir, image_size=image_size)
    n_bands = len(bands_norm)
    anchor_band = int(anchor_band) % n_bands
    print(f"Anchor band  : {anchor_band} ({band_files[anchor_band].stem} nm, fixed, no flow)")
    print(f"Loaded bands : {n_bands} → {n_bands - 1} displacement fields")
    print(f"Wavelength range: {band_files[0].stem} - {band_files[-1].stem} nm")

    registration_kwargs = {
        "anchor_band_idx": anchor_band,
        "pyramid_sizes": list(pyramid_sizes),
        "epochs_per_level": epochs_per_level,
        "patience_per_level": patience_per_level,
        "lr": lr,
        "lamda": lamda,
        "ncc_weight": ncc_weight,
        "int_steps": int_steps,
        "int_downsize": int_downsize,
        "early_stop": early_stop,
        "min_delta": min_delta,
        "lr_schedule": lr_schedule,
        "lr_min": lr_min,
        "fast_mode": fast_mode,
        "feature_mode": feature_mode,
        "spectral_enc_channels": spectral_enc_channels,
        "spectral_enc_kernel": spectral_enc_kernel,
    }

    print(f"Feature mode : {feature_mode}")
    if feature_mode == FEATURE_MODE_SPECTRAL_ENCODER:
        print(f"Spectral enc : K={spectral_enc_channels}, kernel={spectral_enc_kernel}")

    if eval_ref_band_idx is None:
        eval_ref_band_idx = n_bands // 2

    config = build_experiment_config(
        stack_dir, image_size, device, anchor_band, eval_ref_band_idx,
        spectral_path, registration_kwargs, n_bands, band_files, exp_name,
    )

    architecture_text = describe_stackflow_architecture(
        image_size=image_size,
        num_bands=n_bands,
        anchor_band_idx=anchor_band,
        int_steps=int_steps,
        int_downsize=int_downsize,
        fast_mode=fast_mode,
        feature_mode=feature_mode,
        spectral_enc_channels=spectral_enc_channels,
        spectral_enc_kernel=spectral_enc_kernel,
    )

    print("\n[1/4] Evaluating stack BEFORE registration ...")
    metrics_before = evaluate_stack(bands_norm, eval_ref_band_idx)

    print("[2/4] Running StackFlow groupwise registration ...")
    t0 = time.perf_counter()
    bands_norm_after, reg_info, flow_stack = register_pifreg_groupwise_stackflow(
        bands_norm,
        device=str(device),
        anchor_band_idx=anchor_band,
        pyramid_sizes=pyramid_sizes,
        epochs_per_level=epochs_per_level,
        patience_per_level=patience_per_level,
        lr=lr,
        lamda=lamda,
        ncc_weight=ncc_weight,
        int_steps=int_steps,
        int_downsize=int_downsize,
        early_stop=early_stop,
        min_delta=min_delta,
        lr_schedule=lr_schedule,
        lr_min=lr_min,
        fast_mode=fast_mode,
        feature_mode=feature_mode,
        spectral_enc_channels=spectral_enc_channels,
        spectral_enc_kernel=spectral_enc_kernel,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"Registration finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    config["registration_result"] = reg_info

    print("[3/4] Evaluating stack AFTER registration ...")
    metrics_after = evaluate_stack(bands_norm_after, eval_ref_band_idx)
    metrics_summary = compare_metrics(metrics_before, metrics_after)

    print("[4/4] Applying flows to raw images & saving experiment record ...")
    bands_raw_after = warp_bands_with_flow_stack(
        bands_raw, flow_stack, anchor_band_idx=anchor_band, device=str(device),
    )
    rgb_before = hsi_to_rgb(bands_raw, spectral_data_path=str(spectral_path))
    rgb_after = hsi_to_rgb(bands_raw_after, spectral_data_path=str(spectral_path))

    manifest = record_stackflow_experiment(
        run_dir=run_dir,
        config=config,
        architecture_text=architecture_text,
        bands_raw_before=bands_raw,
        bands_raw_after=bands_raw_after,
        band_files=band_files,
        rgb_before=rgb_before,
        rgb_after=rgb_after,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        metrics_summary=metrics_summary,
        elapsed_seconds=elapsed,
        flow_stack=flow_stack,
        moving_band_indices=reg_info.get("moving_band_indices"),
        anchor_band_idx=anchor_band,
        save_before_bands=save_before_bands,
    )

    print("\n" + "=" * 60)
    print("Mean metrics vs reference band")
    print("=" * 60)
    print(f"{'Metric':<8} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 42)
    for key, vals in metrics_summary.items():
        print(f"{key:<8} {vals['before']:>10.4f} {vals['after']:>10.4f} {vals['delta']:>10.4f}")

    print("\nExperiment saved to:")
    print(f"  {run_dir}")
    print("\nKey outputs:")
    for label, rel in manifest["outputs"].items():
        print(f"  [{label}] {run_dir / rel}")

    return bands_raw_after, reg_info, manifest


def _parse_int_list(value):
    if value is None:
        return None
    return [int(x) for x in value.split(",")]


def parse_args():
    p = argparse.ArgumentParser(
        description="PIFReg per-band stackflow groupwise registration with full experiment logging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据与输出
    p.add_argument("--stack-dir", type=str, default=str(DEFAULT_STACK_DIR))
    p.add_argument("--output-dir", type=str, default=None,
                   help="实验根目录（默认 outputs/pifreg_groupwise_stackflow）")
    p.add_argument("--exp-name", type=str, default="run",
                   help="实验名称，与时间戳组成 runs/<ts>_<exp_name>/")
    p.add_argument("--image-size", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--anchor-band", type=int, default=-1,
                   help="锚点波段索引（默认 -1 = 最后一波段）")
    p.add_argument("--eval-ref-band", type=int, default=None)
    p.add_argument("--spectral-path", type=str, default=str(DEFAULT_SPECTRAL_PATH))
    p.add_argument("--no-save-before-bands", action="store_true",
                   help="不保存配准前各波段图（默认保存）")

    # 网络 / 优化超参数
    p.add_argument(
        "--feature-mode",
        type=str,
        default=FEATURE_MODE_SPECTRAL_ENCODER,
        choices=[FEATURE_MODE_MEAN_ANCHOR, FEATURE_MODE_SPECTRAL_ENCODER],
        help="U-Net 输入特征：mean_anchor 或 spectral_encoder（方案6）",
    )
    p.add_argument(
        "--spectral-enc-channels",
        type=int,
        default=DEFAULT_SPECTRAL_ENC_CHANNELS,
        help="光谱编码器输出通道数 K（feature-mode=spectral_encoder 时有效）",
    )
    p.add_argument(
        "--spectral-enc-kernel",
        type=int,
        default=DEFAULT_SPECTRAL_ENC_KERNEL,
        help="光谱编码器 1D 卷积核大小",
    )
    p.add_argument("--no-fast-mode", action="store_true",
                   help="使用 default_unet_features 而非 compact")
    p.add_argument("--pyramid-sizes", type=str, default=None,
                   help="金字塔边长，逗号分隔，如 128,256,512")
    p.add_argument("--epochs-per-level", type=str, default=None,
                   help="各层最大 epoch，逗号分隔，如 900,1500,2500")
    p.add_argument("--patience-per-level", type=str, default=None,
                   help="各层早停耐心，逗号分隔")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lamda", type=float, default=0.005, help="平滑正则权重")
    p.add_argument("--ncc-weight", type=float, default=1.0)
    p.add_argument("--int-steps", type=int, default=3)
    p.add_argument("--int-downsize", type=int, default=2)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    p.add_argument("--lr-min", type=float, default=1e-6)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        stack_dir=args.stack_dir,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        image_size=tuple(args.image_size),
        device=args.device,
        anchor_band=args.anchor_band,
        fast_mode=not args.no_fast_mode,
        eval_ref_band_idx=args.eval_ref_band,
        spectral_path=args.spectral_path,
        save_before_bands=not args.no_save_before_bands,
        pyramid_sizes=_parse_int_list(args.pyramid_sizes),
        epochs_per_level=_parse_int_list(args.epochs_per_level),
        patience_per_level=_parse_int_list(args.patience_per_level),
        lr=args.lr,
        lamda=args.lamda,
        ncc_weight=args.ncc_weight,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
        early_stop=not args.no_early_stop,
        min_delta=args.min_delta,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        feature_mode=args.feature_mode,
        spectral_enc_channels=args.spectral_enc_channels,
        spectral_enc_kernel=args.spectral_enc_kernel,
    )
