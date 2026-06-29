#!/usr/bin/env python3
"""PIFReg StackFlow3D 群组配准实验 — 标准实验记录。"""

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
    describe_stackflow3d_architecture,
    record_groupwise_experiment,
)
from src.python.preprocessing import hsi_to_rgb
from src.python.registration.pif_groupwise_stackflow import warp_bands_with_flow_stack
from src.python.registration.pif_groupwise_stackflow3d import register_pifreg_groupwise_stackflow3d

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SPECTRAL_PATH = PROJECT_ROOT / "HSI2RGB20240517.xlsx"
DEFAULT_STACK_DIR = DATA_DIR / "cut_images_all" / "2024-06-25_10-12-29-white"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pifreg_groupwise_stackflow3d"
EXPERIMENT_ID = "pifreg_groupwise_stackflow3d"


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
):
    stack_dir = resolve_path(stack_dir, PROJECT_ROOT, [DATA_DIR])
    base_output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    run_dir = create_run_dir(base_output, exp_name=exp_name)
    spectral_path = resolve_path(spectral_path or DEFAULT_SPECTRAL_PATH, PROJECT_ROOT, [DATA_DIR])
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    bands_norm, bands_raw, band_files = load_hsi_stack(stack_dir, image_size=image_size)
    n_bands = len(bands_norm)
    anchor_band = int(anchor_band) % n_bands
    if eval_ref_band_idx is None:
        eval_ref_band_idx = n_bands // 2

    registration_kwargs = {"anchor_band_idx": anchor_band, "fast_mode": fast_mode}
    config = build_groupwise_config(
        EXPERIMENT_ID, exp_name, stack_dir, image_size, device, n_bands,
        band_files, eval_ref_band_idx, spectral_path, registration_kwargs,
        anchor_band=anchor_band,
    )

    print("=" * 60)
    print("PIFReg StackFlow3D Experiment")
    print(f"Run folder: {run_dir}, anchor={anchor_band}")

    metrics_before = evaluate_stack(bands_norm, eval_ref_band_idx)
    t0 = time.perf_counter()
    bands_norm_after, reg_info, flow_stack = register_pifreg_groupwise_stackflow3d(
        bands_norm, device=str(device), anchor_band_idx=anchor_band,
        fast_mode=fast_mode, verbose=True,
    )
    elapsed = time.perf_counter() - t0
    config["registration_result"] = reg_info

    metrics_after = evaluate_stack(bands_norm_after, eval_ref_band_idx)
    metrics_summary = compare_metrics(metrics_before, metrics_after)
    bands_raw_after = warp_bands_with_flow_stack(
        bands_raw, flow_stack, anchor_band_idx=anchor_band, device=str(device),
    )

    manifest = record_groupwise_experiment(
        run_dir=run_dir,
        config=config,
        architecture_text=describe_stackflow3d_architecture(
            image_size, n_bands, anchor_band, fast_mode,
        ),
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
        moving_band_indices=reg_info.get("moving_band_indices"),
        anchor_band_idx=anchor_band,
        save_before_bands=save_before_bands,
        rgb_title_after="RGB After PIFReg StackFlow3D",
    )

    print(f"\nExperiment saved to: {run_dir}")
    return bands_raw_after, reg_info, manifest


def parse_args():
    p = argparse.ArgumentParser(description="PIFReg StackFlow3D groupwise registration")
    p.add_argument("--stack-dir", type=str, default=str(DEFAULT_STACK_DIR))
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--exp-name", type=str, default="run")
    p.add_argument("--image-size", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--anchor-band", type=int, default=-1)
    p.add_argument("--no-fast-mode", action="store_true")
    p.add_argument("--eval-ref-band", type=int, default=None)
    p.add_argument("--spectral-path", type=str, default=str(DEFAULT_SPECTRAL_PATH))
    p.add_argument("--no-save-before-bands", action="store_true")
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
    )
