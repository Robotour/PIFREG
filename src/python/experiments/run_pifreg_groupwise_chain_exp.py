#!/usr/bin/env python3
"""PIFReg 链式群组配准实验 — 标准实验记录。"""

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
    describe_chain_architecture,
    record_groupwise_experiment,
)
from src.python.preprocessing import hsi_to_rgb
from src.python.registration.pif_groupwise_chain import (
    SCHEDULE_PAIR_THEN_PYRAMID,
    SCHEDULE_PYRAMID_THEN_PAIRS,
    _parse_level_directions_arg,
    evaluate_chain_pairwise_ncc,
    register_pifreg_chain,
)
from src.python.registration.pif_groupwise_stackflow import (
    DEFAULT_PYRAMID_SIZES,
    warp_bands_with_flow_stack,
)

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SPECTRAL_PATH = PROJECT_ROOT / "HSI2RGB20240517.xlsx"
DEFAULT_STACK_DIR = DATA_DIR / "cut_images_all" / "2024-06-25_10-12-29-white"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pifreg_groupwise_chain"
EXPERIMENT_ID = "pifreg_groupwise_chain"


def run_experiment(
    stack_dir,
    output_dir=None,
    exp_name="run",
    image_size=(512, 512),
    device="cuda",
    fast_mode=True,
    schedule=SCHEDULE_PAIR_THEN_PYRAMID,
    pyramid_sizes=None,
    level_directions=None,
    alternate_direction=False,
    eval_ref_band_idx=None,
    spectral_path=None,
    save_before_bands=True,
):
    stack_dir = resolve_path(stack_dir, PROJECT_ROOT, [DATA_DIR])
    base_output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    run_dir = create_run_dir(base_output, exp_name=exp_name)
    spectral_path = resolve_path(spectral_path or DEFAULT_SPECTRAL_PATH, PROJECT_ROOT, [DATA_DIR])
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    pyramid_sizes = tuple(pyramid_sizes or DEFAULT_PYRAMID_SIZES)

    bands_norm, bands_raw, band_files, wavelengths = load_hsi_stack(
        stack_dir, image_size=image_size, return_wavelengths=True,
    )
    n_bands = len(bands_norm)
    anchor_band = n_bands - 1
    if eval_ref_band_idx is None:
        eval_ref_band_idx = n_bands // 2

    registration_kwargs = {
        "fast_mode": fast_mode,
        "descending": True,
        "schedule": schedule,
        "pyramid_sizes": list(pyramid_sizes),
        "level_directions": (
            ['690→400' if d else '400→690' for d in level_directions]
            if level_directions else None
        ),
        "alternate_direction": alternate_direction,
    }
    config = build_groupwise_config(
        EXPERIMENT_ID, exp_name, stack_dir, image_size, device, n_bands,
        band_files, eval_ref_band_idx, spectral_path, registration_kwargs,
        anchor_band=anchor_band,
    )

    chain_before = evaluate_chain_pairwise_ncc(bands_norm, wavelengths, descending=True)
    print("=" * 60)
    print("PIFReg Chain Groupwise Experiment")
    print(f"Run folder: {run_dir}")
    print(f"Schedule: {schedule}")
    if schedule == SCHEDULE_PYRAMID_THEN_PAIRS:
        print(f"Pyramid sizes: {list(pyramid_sizes)}")
        if level_directions:
            print(f"Level directions: {['690→400' if d else '400→690' for d in level_directions]}")
        elif alternate_direction:
            print("Level directions: alternate (690→400 / 400→690 / …)")
    print(f"Chain NCC before: {chain_before['mean_NCC']:.4f}")

    metrics_before = evaluate_stack(bands_norm, eval_ref_band_idx)
    t0 = time.perf_counter()
    bands_norm_after, reg_info, flow_stack = register_pifreg_chain(
        bands_norm,
        device=str(device),
        descending=True,
        wavelengths_nm=wavelengths,
        schedule=schedule,
        pyramid_sizes=pyramid_sizes,
        level_directions=level_directions,
        alternate_direction=alternate_direction,
        fast_mode=fast_mode,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    config["registration_result"] = reg_info
    chain_after = reg_info["chain_pairwise_ncc"]

    metrics_after = evaluate_stack(bands_norm_after, eval_ref_band_idx)
    metrics_summary = compare_metrics(metrics_before, metrics_after)

    bands_raw_after = warp_bands_with_flow_stack(
        bands_raw, flow_stack, anchor_band_idx=anchor_band, device=str(device),
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
    ]
    if schedule == SCHEDULE_PYRAMID_THEN_PAIRS:
        chain_section.append(f"- Pyramid sizes: `{list(pyramid_sizes)}`")
        if reg_info.get("level_directions"):
            chain_section.append(f"- Level directions: `{reg_info['level_directions']}`")
        elif alternate_direction:
            chain_section.append("- Level directions: alternate per pyramid level")

    manifest = record_groupwise_experiment(
        run_dir=run_dir,
        config=config,
        architecture_text=describe_chain_architecture(
            n_bands,
            descending=True,
            schedule=schedule,
            pyramid_sizes=pyramid_sizes,
            level_directions=reg_info.get("level_directions"),
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
        rgb_title_after="RGB After PIFReg Chain",
        summary_extra_sections=chain_section,
    )

    print(f"Chain NCC after: {chain_after['mean_NCC']:.4f}")
    print(f"\nExperiment saved to: {run_dir}")
    return bands_raw_after, reg_info, manifest


def _parse_int_list(value):
    if value is None:
        return None
    return [int(x) for x in value.split(",")]


def parse_args():
    p = argparse.ArgumentParser(
        description="PIFReg descending chain groupwise registration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stack-dir", type=str, default=str(DEFAULT_STACK_DIR))
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--exp-name", type=str, default="run")
    p.add_argument("--image-size", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-fast-mode", action="store_true")
    p.add_argument(
        "--schedule",
        type=str,
        default=SCHEDULE_PAIR_THEN_PYRAMID,
        choices=[SCHEDULE_PAIR_THEN_PYRAMID, SCHEDULE_PYRAMID_THEN_PAIRS],
        help="pair_then_pyramid=默认每对内部金字塔; pyramid_then_pairs=每层先扫完整条链",
    )
    p.add_argument(
        "--pyramid-sizes",
        type=str,
        default=None,
        help="schedule=pyramid_then_pairs 时生效，如 128,256,512",
    )
    p.add_argument(
        "--level-directions",
        type=str,
        default=None,
        help="每层链扫方向，逗号分隔 desc/asc，如 desc,asc,desc（仅 pyramid_then_pairs）",
    )
    p.add_argument(
        "--alternate-direction",
        action="store_true",
        help="层间交替方向：128 690→400, 256 400→690, 512 690→400（仅 pyramid_then_pairs）",
    )
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
        fast_mode=not args.no_fast_mode,
        schedule=args.schedule,
        pyramid_sizes=_parse_int_list(args.pyramid_sizes),
        level_directions=_parse_level_directions_arg(args.level_directions),
        alternate_direction=args.alternate_direction,
        eval_ref_band_idx=args.eval_ref_band,
        spectral_path=args.spectral_path,
        save_before_bands=not args.no_save_before_bands,
    )
