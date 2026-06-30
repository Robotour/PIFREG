#!/usr/bin/env python3
"""
PIFReg Cascade 群组配准实验 — Keyframe 脚手架 + StackFlow 残差精修

Stage 1: 稀疏关键帧 pairwise PIFReg + flow 插值（快速）
Stage 2: StackFlow 联合精修（init_flow_stack 来自 Stage 1）
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.python.experiments.experiment_data import (
    build_groupwise_config,
    compare_metrics,
    evaluate_stack,
    load_hsi_stack,
    resolve_path,
)
from src.python.experiments.experiment_recorder import (
    create_run_dir,
    describe_cascade_architecture,
    record_groupwise_experiment,
)
from src.python.preprocessing import hsi_to_rgb
from src.python.registration.pif_groupwise_cascade import (
    DEFAULT_REFINE_EPOCHS_PER_LEVEL,
    DEFAULT_REFINE_PATIENCE_PER_LEVEL,
    DEFAULT_REFINE_PYRAMID_SIZES,
    register_pifreg_groupwise_cascade,
)
from src.python.registration.pif_groupwise_stackflow import (
    FEATURE_MODE_MEAN_ANCHOR,
    FEATURE_MODE_SPECTRAL_ENCODER,
    warp_bands_with_flow_stack,
)

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SPECTRAL_PATH = PROJECT_ROOT / "HSI2RGB20240517.xlsx"
DEFAULT_STACK_DIR = DATA_DIR / "cut_images_all" / "2024-06-25_10-12-29-white"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pifreg_groupwise_cascade"
EXPERIMENT_ID = "pifreg_groupwise_cascade"


def run_experiment(
    stack_dir,
    output_dir=None,
    exp_name="run",
    image_size=(512, 512),
    device="cuda",
    anchor_band=-1,
    keyframe_interval=5,
    eval_ref_band_idx=None,
    spectral_path=None,
    save_before_bands=True,
    skip_refine=False,
    # Stage 2 refine hyperparameters
    refine_pyramid_sizes=None,
    refine_epochs_per_level=None,
    refine_patience_per_level=None,
    refine_lr=2e-4,
    refine_lamda=0.01,
    refine_ncc_weight=1.0,
    refine_int_steps=3,
    refine_int_downsize=2,
    refine_early_stop=True,
    refine_min_delta=1e-4,
    refine_lr_schedule="cosine",
    refine_lr_min=1e-6,
    refine_fast_mode=True,
    refine_feature_mode=FEATURE_MODE_MEAN_ANCHOR,
    refine_spectral_enc_channels=4,
    refine_spectral_enc_kernel=3,
):
    stack_dir = resolve_path(stack_dir, PROJECT_ROOT, [DATA_DIR])
    base_output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    run_dir = create_run_dir(base_output, exp_name=exp_name)
    spectral_path = resolve_path(spectral_path or DEFAULT_SPECTRAL_PATH, PROJECT_ROOT, [DATA_DIR])
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    refine_pyramid_sizes = tuple(refine_pyramid_sizes or DEFAULT_REFINE_PYRAMID_SIZES)
    refine_epochs_per_level = list(refine_epochs_per_level or DEFAULT_REFINE_EPOCHS_PER_LEVEL)
    refine_patience_per_level = list(refine_patience_per_level or DEFAULT_REFINE_PATIENCE_PER_LEVEL)

    print("=" * 60)
    print("PIFReg Cascade Groupwise Experiment")
    print("=" * 60)
    print("Stage 1: Keyframe scaffold  |  Stage 2: StackFlow refine")
    print(f"Keyframe interval: {keyframe_interval}")
    print(f"Refine pyramid   : {list(refine_pyramid_sizes)}")
    print("-" * 60)
    print(f"Input folder : {stack_dir}")
    print(f"Run folder   : {run_dir}")

    bands_norm, bands_raw, band_files, wavelengths = load_hsi_stack(
        stack_dir, image_size=image_size, return_wavelengths=True,
    )
    n_bands = len(bands_norm)
    anchor_band = int(anchor_band) % n_bands
    print(f"Anchor band  : {anchor_band} ({band_files[anchor_band].stem} nm)")
    print(f"Loaded bands : {n_bands}")

    if eval_ref_band_idx is None:
        eval_ref_band_idx = n_bands // 2

    registration_kwargs = {
        "anchor_band_idx": anchor_band,
        "keyframe_interval": keyframe_interval,
        "skip_refine": skip_refine,
        "refine_pyramid_sizes": list(refine_pyramid_sizes),
        "refine_epochs_per_level": refine_epochs_per_level,
        "refine_patience_per_level": refine_patience_per_level,
        "refine_lr": refine_lr,
        "refine_lamda": refine_lamda,
        "refine_ncc_weight": refine_ncc_weight,
        "refine_int_steps": refine_int_steps,
        "refine_int_downsize": refine_int_downsize,
        "refine_early_stop": refine_early_stop,
        "refine_min_delta": refine_min_delta,
        "refine_lr_schedule": refine_lr_schedule,
        "refine_lr_min": refine_lr_min,
        "refine_fast_mode": refine_fast_mode,
        "refine_feature_mode": refine_feature_mode,
        "refine_spectral_enc_channels": refine_spectral_enc_channels,
        "refine_spectral_enc_kernel": refine_spectral_enc_kernel,
    }

    config = build_groupwise_config(
        EXPERIMENT_ID, exp_name, stack_dir, image_size, device, n_bands,
        band_files, eval_ref_band_idx, spectral_path, registration_kwargs,
        anchor_band=anchor_band,
    )

    architecture_text = describe_cascade_architecture(
        num_bands=n_bands,
        keyframe_interval=keyframe_interval,
        anchor_band_idx=anchor_band,
        refine_pyramid=refine_pyramid_sizes,
        refine_epochs=refine_epochs_per_level,
        feature_mode=refine_feature_mode,
    )

    print("\n[1/4] Evaluating stack BEFORE registration ...")
    metrics_before = evaluate_stack(bands_norm, eval_ref_band_idx)

    print("[2/4] Running Cascade registration ...")
    t0 = time.perf_counter()
    bands_norm_after, reg_info, flow_stack = register_pifreg_groupwise_cascade(
        bands_norm,
        device=str(device),
        anchor_band_idx=anchor_band,
        keyframe_interval=keyframe_interval,
        wavelengths_nm=wavelengths,
        refine_pyramid_sizes=refine_pyramid_sizes,
        refine_epochs_per_level=refine_epochs_per_level,
        refine_patience_per_level=refine_patience_per_level,
        refine_lr=refine_lr,
        refine_lamda=refine_lamda,
        refine_ncc_weight=refine_ncc_weight,
        refine_int_steps=refine_int_steps,
        refine_int_downsize=refine_int_downsize,
        refine_early_stop=refine_early_stop,
        refine_min_delta=refine_min_delta,
        refine_lr_schedule=refine_lr_schedule,
        refine_lr_min=refine_lr_min,
        refine_fast_mode=refine_fast_mode,
        refine_feature_mode=refine_feature_mode,
        refine_spectral_enc_channels=refine_spectral_enc_channels,
        refine_spectral_enc_kernel=refine_spectral_enc_kernel,
        skip_refine=skip_refine,
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

    chain_scaffold = reg_info.get("chain_pairwise_ncc_scaffold") or reg_info.get("scaffold", {}).get(
        "chain_pairwise_ncc", {}
    )
    chain_after = reg_info.get("chain_pairwise_ncc", {})
    cascade_section = [
        "",
        "## Cascade Chain Pairwise NCC",
        "",
        f"- Scaffold (Stage 1): {chain_scaffold.get('mean_NCC', float('nan')):.4f}",
        f"- Final (Stage 2)  : {chain_after.get('mean_NCC', float('nan')):.4f}",
    ]
    delta = reg_info.get("chain_ncc_delta")
    if delta is not None:
        cascade_section.append(f"- Refine delta     : {delta:+.4f}")
    scaffold_calls = reg_info.get("scaffold", {}).get("num_pifreg_calls")
    if scaffold_calls is not None:
        cascade_section.append(f"- Stage 1 PIFReg calls: {scaffold_calls} (vs {n_bands - 1} full chain)")

    manifest = record_groupwise_experiment(
        run_dir=run_dir,
        config=config,
        architecture_text=architecture_text,
        bands_raw_before=bands_raw,
        bands_raw_after=bands_raw_after,
        band_files=band_files,
        rgb_before=hsi_to_rgb(bands_raw, spectral_data_path=str(spectral_path)),
        rgb_after=hsi_to_rgb(bands_raw_after, spectral_data_path=str(spectral_path)),
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        metrics_summary=metrics_summary,
        elapsed_seconds=elapsed,
        flow_stack=flow_stack,
        moving_band_indices=reg_info.get("scaffold", {}).get("moving_band_indices"),
        anchor_band_idx=anchor_band,
        save_before_bands=save_before_bands,
        rgb_title_after="RGB After PIFReg Cascade",
        summary_extra_sections=cascade_section,
    )

    print("\n" + "=" * 60)
    print("Mean metrics vs reference band")
    print("=" * 60)
    print(f"{'Metric':<8} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 42)
    for key, vals in metrics_summary.items():
        print(f"{key:<8} {vals['before']:>10.4f} {vals['after']:>10.4f} {vals['delta']:>10.4f}")

    if chain_after:
        print(f"\nChain NCC (final): {chain_after.get('mean_NCC', float('nan')):.4f}")
    print(f"\nExperiment saved to: {run_dir}")
    return bands_raw_after, reg_info, manifest


def _parse_int_list(value):
    if value is None:
        return None
    return [int(x) for x in value.split(",")]


def parse_args():
    p = argparse.ArgumentParser(
        description="PIFReg cascade: keyframe scaffold + StackFlow residual refine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stack-dir", type=str, default=str(DEFAULT_STACK_DIR))
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--exp-name", type=str, default="run")
    p.add_argument("--image-size", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--anchor-band", type=int, default=-1)
    p.add_argument("--keyframe-interval", type=int, default=5,
                   help="Stage 1 关键帧间隔（波段数）")
    p.add_argument("--eval-ref-band", type=int, default=None)
    p.add_argument("--spectral-path", type=str, default=str(DEFAULT_SPECTRAL_PATH))
    p.add_argument("--no-save-before-bands", action="store_true")
    p.add_argument("--skip-refine", action="store_true",
                   help="仅运行 Stage 1 脚手架，不做 StackFlow 精修")

    p.add_argument("--refine-pyramid-sizes", type=str, default=None,
                   help="Stage 2 金字塔，逗号分隔，默认 128,256,512")
    p.add_argument("--refine-epochs-per-level", type=str, default=None,
                   help="Stage 2 各层 epoch，默认 300,500,800")
    p.add_argument("--refine-patience-per-level", type=str, default=None)
    p.add_argument("--refine-lr", type=float, default=2e-4)
    p.add_argument("--refine-lamda", type=float, default=0.01)
    p.add_argument("--refine-ncc-weight", type=float, default=1.0)
    p.add_argument("--refine-int-steps", type=int, default=3)
    p.add_argument("--refine-int-downsize", type=int, default=2)
    p.add_argument("--no-refine-early-stop", action="store_true")
    p.add_argument("--refine-min-delta", type=float, default=1e-4)
    p.add_argument("--refine-lr-schedule", type=str, default="cosine",
                   choices=["cosine", "plateau", "none"])
    p.add_argument("--refine-lr-min", type=float, default=1e-6)
    p.add_argument("--no-refine-fast-mode", action="store_true")
    p.add_argument(
        "--refine-feature-mode",
        type=str,
        default=FEATURE_MODE_MEAN_ANCHOR,
        choices=[FEATURE_MODE_MEAN_ANCHOR, FEATURE_MODE_SPECTRAL_ENCODER],
    )
    p.add_argument("--refine-spectral-enc-channels", type=int, default=4)
    p.add_argument("--refine-spectral-enc-kernel", type=int, default=3)
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
        keyframe_interval=args.keyframe_interval,
        eval_ref_band_idx=args.eval_ref_band,
        spectral_path=args.spectral_path,
        save_before_bands=not args.no_save_before_bands,
        skip_refine=args.skip_refine,
        refine_pyramid_sizes=_parse_int_list(args.refine_pyramid_sizes),
        refine_epochs_per_level=_parse_int_list(args.refine_epochs_per_level),
        refine_patience_per_level=_parse_int_list(args.refine_patience_per_level),
        refine_lr=args.refine_lr,
        refine_lamda=args.refine_lamda,
        refine_ncc_weight=args.refine_ncc_weight,
        refine_int_steps=args.refine_int_steps,
        refine_int_downsize=args.refine_int_downsize,
        refine_early_stop=not args.no_refine_early_stop,
        refine_min_delta=args.refine_min_delta,
        refine_lr_schedule=args.refine_lr_schedule,
        refine_lr_min=args.refine_lr_min,
        refine_fast_mode=not args.no_refine_fast_mode,
        refine_feature_mode=args.refine_feature_mode,
        refine_spectral_enc_channels=args.refine_spectral_enc_channels,
        refine_spectral_enc_kernel=args.refine_spectral_enc_kernel,
    )
