#!/usr/bin/env python3
"""
用验证集上表现最佳的 VoxelMorph checkpoint，在测试 session 上链式配准，
并输出 fake RGB（配准前 / 后 / 对比图）。

示例:
    python src/python/experiments/visualize_voxelmorph_test.py \\
        --model-dir models/voxelmorph_cut_images_all \\
        --num-sessions 6 \\
        --device cuda
"""

from __future__ import annotations

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
import torch

from src.python.experiments.experiment_recorder import save_rgb_outputs
from src.python.preprocessing import hsi_to_rgb
from src.python.voxelmorph.networks import VxmDense
from src.python.voxelmorph.training import (
    _list_readable_band_files,
    _normalize_band,
    register_raw_stack_with_chain_flows,
    register_stack_with_voxelmorph_chain,
    select_best_checkpoint,
)

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "voxelmorph_cut_images_all"
DEFAULT_SPECTRAL_PATH = PROJECT_ROOT / "HSI2RGB20240517.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "voxelmorph_test_rgb"


def _load_stack(folder, image_size):
    files = _list_readable_band_files(Path(folder))
    if len(files) < 2:
        raise ValueError(f"Need >=2 readable bands in {folder}")
    bands_norm, bands_raw, wavelengths = [], [], []
    for path in files:
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        raw = raw.astype(np.float32)
        if image_size is not None:
            raw = cv2.resize(raw, image_size)
        bands_raw.append(raw.copy())
        bands_norm.append(_normalize_band(raw))
        wavelengths.append(path.stem)
    return bands_norm, bands_raw, files, wavelengths


def _load_test_folders(model_dir, split_manifest=None):
    manifest_path = Path(split_manifest) if split_manifest else Path(model_dir) / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [Path(p) for p in payload["test_sessions"]]


def _session_slug(folder: Path) -> str:
    return folder.name


def visualize_test_sessions(
    model_dir=None,
    run_dir=None,
    output_dir=None,
    checkpoint=None,
    image_size=(256, 256),
    device="cuda",
    num_sessions=6,
    session_indices=None,
    all_test_sessions=False,
    test_folders=None,
    spectral_path=None,
    descending=True,
    metric="NCC_after",
    smooth_flow_sigma=1.5,
):
    run_dir = Path(run_dir) if run_dir else None
    model_dir = Path(model_dir) if model_dir else (run_dir or DEFAULT_MODEL_DIR)
    if run_dir is not None:
        model_dir = run_dir
    output_dir = Path(output_dir) if output_dir else (
        (run_dir / "visualizations") if run_dir else DEFAULT_OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    spectral_path = Path(spectral_path or DEFAULT_SPECTRAL_PATH)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if checkpoint:
        ckpt_path = str(Path(checkpoint).resolve())
        best_info = {
            "epoch": None,
            "metric": metric,
            "metric_value": None,
            "checkpoint": ckpt_path,
            "val": None,
            "note": "user-specified checkpoint",
        }
    else:
        default_ckpt = model_dir / "checkpoints" / "best.pt"
        if default_ckpt.is_file():
            ckpt_path = str(default_ckpt.resolve())
            best_info = {
                "epoch": None,
                "metric": metric,
                "metric_value": None,
                "checkpoint": ckpt_path,
                "val": None,
                "note": "checkpoints/best.pt",
            }
        else:
            best_info = select_best_checkpoint(model_dir, metric=metric)
            ckpt_path = best_info["checkpoint"]
    print("=" * 60)
    print("VoxelMorph test-set fake-RGB visualization")
    print(f"Model dir : {model_dir}")
    print(f"Checkpoint: {ckpt_path}")
    if best_info.get("epoch") is not None:
        print(
            f"Best val  : epoch={best_info['epoch']}  "
            f"{best_info['metric']}={best_info['metric_value']:.6f}"
        )
    if best_info.get("note"):
        print(f"Note      : {best_info['note']}")
    print(f"Device    : {device}")
    print("=" * 60)

    model = VxmDense.load(ckpt_path, device)
    model.eval()

    config_path = model_dir / "config.json"
    if config_path.is_file() and image_size is None:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        image_size = tuple(cfg.get("image_size", [256, 256]))

    if test_folders is None:
        test_folders = _load_test_folders(model_dir)
    if session_indices:
        chosen = [test_folders[i] for i in session_indices if 0 <= i < len(test_folders)]
    elif all_test_sessions:
        chosen = list(test_folders)
    else:
        chosen = test_folders[: max(1, int(num_sessions))]

    print(f"Test sessions available: {len(test_folders)}")
    print(f"Visualizing {len(chosen)} session(s)")

    summary_rows = []
    for si, folder in enumerate(chosen, start=1):
        slug = _session_slug(folder)
        run_dir = output_dir / f"{si:02d}_{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{si}/{len(chosen)}] {folder}")

        bands_norm, bands_raw, band_files, wavelengths = _load_stack(folder, image_size)
        bands_norm_after, chain_steps = register_stack_with_voxelmorph_chain(
            model, bands_norm, device=device, descending=descending,
            smooth_flow_sigma=smooth_flow_sigma,
        )
        bands_raw_after = register_raw_stack_with_chain_flows(
            bands_raw, chain_steps, device=device,
        )

        rgb_before = hsi_to_rgb(bands_raw, spectral_data_path=str(spectral_path))
        rgb_after = hsi_to_rgb(bands_raw_after, spectral_data_path=str(spectral_path))
        paths = save_rgb_outputs(
            run_dir,
            rgb_before,
            rgb_after,
            title_after="RGB After VoxelMorph Chain",
        )

        # also save a larger side-by-side with session name
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(cv2.cvtColor(rgb_before, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Before — {slug}")
        axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(rgb_after, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"After — {slug}")
        axes[1].axis("off")
        plt.tight_layout()
        overview = run_dir / "rgb_overview.png"
        plt.savefig(overview, dpi=160, bbox_inches="tight")
        plt.close()

        meta = {
            "session": str(folder),
            "num_bands": len(band_files),
            "wavelengths": wavelengths,
            "checkpoint": ckpt_path,
            "best_info": {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in best_info.items()
                if k != "val"
            },
            "outputs": {
                "rgb_before": str(paths["before"]),
                "rgb_after": str(paths["after"]),
                "rgb_compare": str(paths["compare"]),
                "rgb_overview": str(overview),
            },
        }
        with open(run_dir / "session_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        summary_rows.append(meta)
        print(f"  saved: {overview}")

    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
                "best_info": {
                    k: v for k, v in best_info.items() if k != "val"
                },
                "sessions": summary_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nDone.")
    print(f"Output root: {output_dir}")
    print(f"Index      : {index_path}")
    return output_dir, summary_rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize VoxelMorph best-checkpoint registration on test sessions (fake RGB)",
    )
    p.add_argument("--run-dir", type=str, default=None, help="Experiment run dir (uses checkpoints/best.pt)")
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None, help="Override; default checkpoints/best.pt")
    p.add_argument("--image-size", type=int, nargs=2, default=[256, 256], metavar=("W", "H"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-sessions", type=int, default=6, help="How many test sessions (ignored if --all-test-sessions)")
    p.add_argument("--all-test-sessions", action="store_true", help="Visualize every test session")
    p.add_argument(
        "--session-indices",
        type=str,
        default=None,
        help="Comma-separated indices into test_sessions, e.g. 0,3,5",
    )
    p.add_argument("--spectral-path", type=str, default=str(DEFAULT_SPECTRAL_PATH))
    p.add_argument("--ascending", action="store_true", help="Chain toward short wavelength (default: descending)")
    p.add_argument("--metric", type=str, default="NCC_after", help="Val metric for best checkpoint")
    p.add_argument(
        "--smooth-flow-sigma",
        type=float,
        default=1.5,
        help="Gaussian smooth on chain flows to reduce spikes (0=off)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    indices = None
    if args.session_indices:
        indices = [int(x) for x in args.session_indices.split(",") if x.strip() != ""]
    visualize_test_sessions(
        run_dir=args.run_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        image_size=tuple(args.image_size),
        device=args.device,
        num_sessions=args.num_sessions,
        session_indices=indices,
        all_test_sessions=args.all_test_sessions,
        spectral_path=args.spectral_path,
        descending=not args.ascending,
        metric=args.metric,
        smooth_flow_sigma=args.smooth_flow_sigma,
    )


if __name__ == "__main__":
    main()
