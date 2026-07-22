#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adjacent-band pairwise difference heatmaps for HSI stacks.

Reads all band images from the stack folder (sorted by filename stem, e.g. 404.jpeg),
then generates heatmaps for each adjacent pair on disk.
Default sample: 2024-06-25_10-12-29-white.

Methods include raw MSE plus brightness-robust variants:
  min-max, histogram equalization, histogram matching, CLAHE, z-score,
  gradient MSE, local NCC dissimilarity, local MI dissimilarity.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pairwise_heatmap import (
    HeatmapMethod,
    adjacent_pairs,
    compute_all_heatmaps,
    default_heatmap_methods,
    heatmap_display_vmax,
    load_wavelength_stack,
)

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_STACK_DIR = DATA_DIR / "cut_images_all" / "2024-06-25_10-12-29-white"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pairwise_rmse_heatmap"


def _band_to_uint8(band: np.ndarray) -> np.ndarray:
    arr = np.asarray(band, dtype=np.float32)
    if arr.max() <= 1.0 and arr.min() >= 0.0:
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _save_single_heatmap(
    heatmap: np.ndarray,
    out_path: Path,
    title: str,
    vmax: float,
    cmap: str = "jet",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heatmap, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_pair_panel(
    fixed_raw: np.ndarray,
    moving_raw: np.ndarray,
    heatmaps: dict,
    methods: list[HeatmapMethod],
    out_path: Path,
    pair_label: str,
    percentile: float,
    cmap: str = "jet",
) -> dict:
    """Save one composite figure with fixed/moving previews + all heatmaps."""
    n_methods = len(methods)
    n_cols = 3
    n_rows = int(np.ceil((n_methods + 2) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    vmax_by_key = {
        key: heatmap_display_vmax(arr, percentile=percentile)
        for key, arr in heatmaps.items()
    }

    axes[0].imshow(_band_to_uint8(fixed_raw), cmap="gray")
    axes[0].set_title(f"Fixed {pair_label.split('_')[0]} nm")
    axes[0].axis("off")

    axes[1].imshow(_band_to_uint8(moving_raw), cmap="gray")
    axes[1].set_title(f"Moving {pair_label.split('_')[1]} nm")
    axes[1].axis("off")

    for idx, method in enumerate(methods, start=2):
        ax = axes[idx]
        arr = heatmaps[method.key]
        vmax = vmax_by_key[method.key]
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(method.title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for j in range(2 + n_methods, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Pairwise Heatmaps: {pair_label} nm", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return vmax_by_key


def run_pairwise_heatmap_experiment(
    stack_dir,
    output_dir=None,
    wl_min: int | None = None,
    wl_max: int | None = None,
    image_size=None,
    percentile: float = 99.0,
    cmap: str = "jet",
    methods: list[HeatmapMethod] | None = None,
    exp_name: str = "run",
) -> Path:
    stack_dir = Path(stack_dir)
    if not stack_dir.is_dir():
        raise FileNotFoundError(f"Stack directory not found: {stack_dir}")

    methods = list(methods or default_heatmap_methods())
    bands, wavelengths, band_paths = load_wavelength_stack(
        stack_dir,
        wl_min=wl_min,
        wl_max=wl_max,
        image_size=image_size,
    )
    pairs = adjacent_pairs(wavelengths)
    if not pairs:
        raise ValueError("Need at least two wavelengths to form adjacent pairs.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir or DEFAULT_OUTPUT_DIR) / stack_dir.name / f"{exp_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "pairwise_rmse_heatmap",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stack_dir": str(stack_dir.resolve()),
        "wavelengths_nm": [str(w) for w in wavelengths],
        "wavelength_filter": {"wl_min": wl_min, "wl_max": wl_max},
        "num_bands": len(wavelengths),
        "num_pairs": len(pairs),
        "image_size": list(image_size) if image_size else None,
        "percentile_vmax": percentile,
        "methods": [
            {"key": m.key, "title": m.title, "description": m.description}
            for m in methods
        ],
        "pairs": [],
    }

    print("=" * 60)
    print("Pairwise RMSE / Difference Heatmap Experiment")
    print(f"Stack: {stack_dir}")
    print(f"Bands ({len(wavelengths)}): {wavelengths}")
    print(f"Adjacent pairs: {len(pairs)}")
    print(f"Output: {run_dir}")
    print("=" * 60)

    wl_to_band = {wl: band for wl, band in zip(wavelengths, bands)}
    wl_to_path = {wl: p for wl, p in zip(wavelengths, band_paths)}

    for pair_idx, (wl_fixed, wl_moving) in enumerate(pairs, start=1):
        pair_label = f"{wl_fixed}_{wl_moving}"
        pair_dir = run_dir / pair_label
        pair_dir.mkdir(parents=True, exist_ok=True)

        fixed_raw = wl_to_band[wl_fixed]
        moving_raw = wl_to_band[wl_moving]
        heatmaps = compute_all_heatmaps(fixed_raw, moving_raw, methods=methods)

        panel_path = pair_dir / f"{pair_label}_panel.png"
        vmax_by_key = _save_pair_panel(
            fixed_raw,
            moving_raw,
            heatmaps,
            methods,
            panel_path,
            pair_label,
            percentile=percentile,
            cmap=cmap,
        )

        method_paths = {}
        for method in methods:
            arr = heatmaps[method.key]
            np.save(pair_dir / f"{method.key}.npy", arr.astype(np.float32))
            single_path = pair_dir / f"{method.key}.png"
            _save_single_heatmap(
                arr,
                single_path,
                f"{pair_label} nm - {method.title}",
                vmax=vmax_by_key[method.key],
                cmap=cmap,
            )
            method_paths[method.key] = str(single_path.relative_to(run_dir))

        pair_entry = {
            "pair_index": pair_idx,
            "fixed_nm": wl_fixed,
            "moving_nm": wl_moving,
            "label": pair_label,
            "fixed_path": str(wl_to_path[wl_fixed]),
            "moving_path": str(wl_to_path[wl_moving]),
            "panel": str(panel_path.relative_to(run_dir)),
            "methods": method_paths,
            "vmax": vmax_by_key,
            "global_mse": {k: float(v.mean()) for k, v in heatmaps.items()},
        }
        manifest["pairs"].append(pair_entry)
        print(f"[{pair_idx:02d}/{len(pairs)}] {pair_label} nm -> {panel_path.name}")

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    _save_overview_montage(run_dir, manifest, methods, percentile, cmap)

    print(f"\nDone. {len(pairs)} pair panels saved under:\n  {run_dir}")
    print(f"Manifest: {manifest_path}")
    return run_dir


def _save_overview_montage(
    run_dir: Path,
    manifest: dict,
    methods: list[HeatmapMethod],
    percentile: float,
    cmap: str,
) -> None:
    """One montage per method: all 29 pairs in a grid (fixed method, all pairs)."""
    overview_dir = run_dir / "overview"
    overview_dir.mkdir(exist_ok=True)

    for method in methods:
        n_pairs = len(manifest["pairs"])
        n_cols = 5
        n_rows = int(np.ceil(n_pairs / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        for idx, pair in enumerate(manifest["pairs"]):
            arr = np.load(run_dir / pair["label"] / f"{method.key}.npy")
            vmax = pair["vmax"][method.key]
            ax = axes[idx]
            im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(pair["label"], fontsize=8)
            ax.axis("off")

        for j in range(n_pairs, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"All pairs - {method.title}", fontsize=13)
        plt.tight_layout()
        out = overview_dir / f"all_pairs_{method.key}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate adjacent-band pairwise difference heatmaps from folder band files.",
    )
    parser.add_argument(
        "--stack-dir",
        default=str(DEFAULT_STACK_DIR),
        help="Folder containing wavelength-named band images (e.g. 404.jpeg).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Base output directory.",
    )
    parser.add_argument(
        "--wl-min",
        type=int,
        default=None,
        help="Optional: only include bands with stem >= wl-min (nm).",
    )
    parser.add_argument(
        "--wl-max",
        type=int,
        default=None,
        help="Optional: only include bands with stem <= wl-max (nm).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Optional resize, e.g. --image-size 512 512",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile for heatmap color scale vmax (default 99).",
    )
    parser.add_argument("--cmap", default="jet", help="Matplotlib colormap.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Subset of method keys (default: all).",
    )
    parser.add_argument("--exp-name", default="run")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    all_methods = default_heatmap_methods()
    if args.methods:
        key_set = set(args.methods)
        methods = [m for m in all_methods if m.key in key_set]
        unknown = key_set - {m.key for m in methods}
        if unknown:
            raise SystemExit(f"Unknown method keys: {sorted(unknown)}")
        if not methods:
            raise SystemExit("No valid methods selected.")
    else:
        methods = all_methods

    image_size = tuple(args.image_size) if args.image_size else None
    run_pairwise_heatmap_experiment(
        stack_dir=args.stack_dir,
        output_dir=args.output_dir,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        image_size=image_size,
        percentile=args.percentile,
        cmap=args.cmap,
        methods=methods,
        exp_name=args.exp_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())