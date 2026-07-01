#!/usr/bin/env python3
"""PIFReg 滑动窗口均衡配准实验 — 无锚点 3D U-Net + 顺序 warp。"""

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
    describe_sliding_window_architecture,
    record_groupwise_experiment,
)
from src.python.preprocessing import hsi_to_rgb
from src.python.registration.pif_groupwise_chain import evaluate_chain_pairwise_ncc
from src.python.registration.pif_groupwise_sliding_window import (
    DEFAULT_EPOCHS_PER_WINDOW,
    DEFAULT_HISTOGRAM_MATCH,
    DEFAULT_AFFINE_INIT,
    DEFAULT_LAMDA_GAUGE,
    DEFAULT_LAMDA_SPEC,
    DEFAULT_LAMDA_VAR,
    DEFAULT_PATIENCE_PER_WINDOW,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_STRIDE,
    SCHEDULE_PYRAMID_THEN_WINDOWS,
    SCHEDULE_WINDOW_THEN_PYRAMID,
    register_pifreg_groupwise_sliding_window,
    warp_bands_with_flow_volume,
)
from src.python.registration.pif_groupwise_stackflow import DEFAULT_PYRAMID_SIZES

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SPECTRAL_PATH = PROJECT_ROOT / "HSI2RGB20240517.xlsx"
DEFAULT_STACK_DIR = DATA_DIR / "cut_images_all" / "2024-06-25_10-12-29-white"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pifreg_groupwise_sliding_window"
EXPERIMENT_ID = "pifreg_groupwise_sliding_window"


def run_experiment(
    stack_dir,
    output_dir=None,
    exp_name="run",
    image_size=(512, 512),
    device="cuda",
    window_size=DEFAULT_WINDOW_SIZE,
    window_stride=DEFAULT_WINDOW_STRIDE,
    schedule=SCHEDULE_PYRAMID_THEN_WINDOWS,
    fast_mode=True,
    eval_ref_band_idx=None,
    spectral_path=None,
    save_before_bands=True,
    pyramid_sizes=None,
    epochs_per_window=None,
    patience_per_window=None,
    lr=2e-4,
    lamda=0.005,
    lamda_spec=DEFAULT_LAMDA_SPEC,
    lamda_gauge=DEFAULT_LAMDA_GAUGE,
    lamda_var=DEFAULT_LAMDA_VAR,
    ncc_weight=1.0,
    int_steps=3,
    int_downsize=2,
    early_stop=True,
    min_delta=1e-5,
    lr_schedule="cosine",
    lr_min=1e-6,
    histogram_match=DEFAULT_HISTOGRAM_MATCH,
    affine_init=DEFAULT_AFFINE_INIT,
):
    stack_dir = resolve_path(stack_dir, PROJECT_ROOT, [DATA_DIR])
    base_output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    run_dir = create_run_dir(base_output, exp_name=exp_name)
    spectral_path = resolve_path(spectral_path or DEFAULT_SPECTRAL_PATH, PROJECT_ROOT, [DATA_DIR])
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    pyramid_sizes = tuple(pyramid_sizes or DEFAULT_PYRAMID_SIZES)
    epochs_per_window = list(epochs_per_window or DEFAULT_EPOCHS_PER_WINDOW)
    patience_per_window = list(patience_per_window or DEFAULT_PATIENCE_PER_WINDOW)

    bands_norm, bands_raw, band_files, wavelengths = load_hsi_stack(
        stack_dir, image_size=image_size, return_wavelengths=True,
    )
    n_bands = len(bands_norm)
    if eval_ref_band_idx is None:
        eval_ref_band_idx = n_bands // 2

    registration_kwargs = {
        "window_size": window_size,
        "window_stride": window_stride,
        "schedule": schedule,
        "pyramid_sizes": list(pyramid_sizes),
        "epochs_per_window": epochs_per_window,
        "patience_per_window": patience_per_window,
        "lr": lr,
        "lamda": lamda,
        "lamda_spec": lamda_spec,
        "lamda_gauge": lamda_gauge,
        "lamda_var": lamda_var,
        "ncc_weight": ncc_weight,
        "int_steps": int_steps,
        "int_downsize": int_downsize,
        "early_stop": early_stop,
        "min_delta": min_delta,
        "lr_schedule": lr_schedule,
        "lr_min": lr_min,
        "fast_mode": fast_mode,
        "histogram_match": histogram_match,
        "affine_init": affine_init,
    }
    config = build_groupwise_config(
        EXPERIMENT_ID, exp_name, stack_dir, image_size, device, n_bands,
        band_files, eval_ref_band_idx, spectral_path, registration_kwargs,
        anchor_band=None,
    )

    chain_before = evaluate_chain_pairwise_ncc(bands_norm, wavelengths, descending=True)
    print("=" * 60)
    print("PIFReg Sliding-Window Experiment (balanced, no anchor)")
    print(f"Run folder: {run_dir}")
    print(f"Window: size={window_size}, stride={window_stride}, schedule={schedule}")
    print(f"Preprocess: histogram_match={histogram_match}, affine_init={affine_init}")
    print(f"Chain NCC before: {chain_before['mean_NCC']:.4f}")

    metrics_before = evaluate_stack(bands_norm, eval_ref_band_idx)
    t0 = time.perf_counter()
    bands_norm_after, reg_info, flow_volume = register_pifreg_groupwise_sliding_window(
        bands_norm,
        device=str(device),
        window_size=window_size,
        window_stride=window_stride,
        schedule=schedule,
        pyramid_sizes=pyramid_sizes,
        epochs_per_window=epochs_per_window,
        patience_per_window=patience_per_window,
        lr=lr,
        lamda=lamda,
        lamda_spec=lamda_spec,
        lamda_gauge=lamda_gauge,
        lamda_var=lamda_var,
        ncc_weight=ncc_weight,
        int_steps=int_steps,
        int_downsize=int_downsize,
        early_stop=early_stop,
        min_delta=min_delta,
        lr_schedule=lr_schedule,
        lr_min=lr_min,
        fast_mode=fast_mode,
        histogram_match=histogram_match,
        affine_init=affine_init,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    config["registration_result"] = reg_info
    chain_after = evaluate_chain_pairwise_ncc(bands_norm_after, wavelengths, descending=True)

    metrics_after = evaluate_stack(bands_norm_after, eval_ref_band_idx)
    metrics_summary = compare_metrics(metrics_before, metrics_after)

    bands_raw_after = warp_bands_with_flow_volume(
        bands_raw, flow_volume, device=str(device),
    )

    chain_section = [
        "",
        "## Chain Pairwise NCC",
        "",
        f"- Before: {chain_before['mean_NCC']:.4f}",
        f"- After: {chain_after['mean_NCC']:.4f}",
        f"- Delta: {chain_after['mean_NCC'] - chain_before['mean_NCC']:+.4f}",
        "",
        "## Schedule",
        "",
        f"- Mode: `{schedule}`",
        f"- Warp: sequential in-place (no flow averaging)",
    ]

    manifest = record_groupwise_experiment(
        run_dir=run_dir,
        config=config,
        architecture_text=describe_sliding_window_architecture(
            image_size=image_size,
            num_bands=n_bands,
            window_size=window_size,
            window_stride=window_stride,
            schedule=schedule,
            fast_mode=fast_mode,
            histogram_match=histogram_match,
            affine_init=affine_init,
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
        flow_stack=flow_volume,
        moving_band_indices=list(range(n_bands)),
        anchor_band_idx=eval_ref_band_idx,
        save_before_bands=save_before_bands,
        rgb_title_after="RGB After PIFReg Sliding Window",
        summary_extra_sections=chain_section,
    )

    print(f"Chain NCC after: {chain_after['mean_NCC']:.4f}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"\nExperiment saved to: {run_dir}")
    return bands_raw_after, reg_info, manifest


def _parse_int_list(value):
    if value is None:
        return None
    return [int(x) for x in value.split(",")]


def parse_args():
    p = argparse.ArgumentParser(
        description="PIFReg sliding-window balanced groupwise registration (3D U-Net)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stack-dir", type=str, default=str(DEFAULT_STACK_DIR))
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--exp-name", type=str, default="run")
    p.add_argument("--image-size", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    p.add_argument("--window-stride", type=int, default=DEFAULT_WINDOW_STRIDE)
    p.add_argument(
        "--schedule",
        type=str,
        default=SCHEDULE_PYRAMID_THEN_WINDOWS,
        choices=[SCHEDULE_PYRAMID_THEN_WINDOWS, SCHEDULE_WINDOW_THEN_PYRAMID],
        help="pyramid_then_windows=先空间后光谱; window_then_pyramid=先光谱后空间",
    )
    p.add_argument("--eval-ref-band", type=int, default=None)
    p.add_argument("--spectral-path", type=str, default=str(DEFAULT_SPECTRAL_PATH))
    p.add_argument("--no-save-before-bands", action="store_true")
    p.add_argument("--no-fast-mode", action="store_true")
    p.add_argument("--pyramid-sizes", type=str, default=None)
    p.add_argument("--epochs-per-window", type=str, default=None)
    p.add_argument("--patience-per-window", type=str, default=None)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lamda", type=float, default=0.005)
    p.add_argument("--lamda-spec", type=float, default=DEFAULT_LAMDA_SPEC)
    p.add_argument("--lamda-gauge", type=float, default=DEFAULT_LAMDA_GAUGE)
    p.add_argument("--lamda-var", type=float, default=DEFAULT_LAMDA_VAR)
    p.add_argument("--ncc-weight", type=float, default=1.0)
    p.add_argument("--int-steps", type=int, default=3)
    p.add_argument("--int-downsize", type=int, default=2)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument(
        "--no-histogram-match",
        action="store_true",
        help="关闭窗口内直方图匹配（Chain 默认开启）",
    )
    p.add_argument(
        "--affine-init",
        action="store_true",
        help="窗口内逐对 StackReg 仿射预配准（Chain 默认关闭）",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        stack_dir=args.stack_dir,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        image_size=tuple(args.image_size),
        device=args.device,
        window_size=args.window_size,
        window_stride=args.window_stride,
        schedule=args.schedule,
        fast_mode=not args.no_fast_mode,
        eval_ref_band_idx=args.eval_ref_band,
        spectral_path=args.spectral_path,
        save_before_bands=not args.no_save_before_bands,
        pyramid_sizes=_parse_int_list(args.pyramid_sizes),
        epochs_per_window=_parse_int_list(args.epochs_per_window),
        patience_per_window=_parse_int_list(args.patience_per_window),
        lr=args.lr,
        lamda=args.lamda,
        lamda_spec=args.lamda_spec,
        lamda_gauge=args.lamda_gauge,
        lamda_var=args.lamda_var,
        ncc_weight=args.ncc_weight,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
        early_stop=not args.no_early_stop,
        min_delta=args.min_delta,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        histogram_match=not args.no_histogram_match,
        affine_init=args.affine_init,
    )
