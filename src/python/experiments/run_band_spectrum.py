#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize 2D FFT magnitude spectra for every band in an HSI stack folder.

Default sample: data/cut_images_all/2024-06-25_10-12-29-white

Outputs per band:
  - {wl}_spectrum2d.png          2D log-magnitude spectrum (DC at center)
  - {wl}_band_spectrum.png       original band + spectrum side-by-side

Outputs for the whole stack:
  - all_bands_spectrum_compare.png   6x5 grid, shared color scale
  - all_bands_spectrum_strip.png     horizontal strip (404->695 nm)
  - all_bands_spectrum_diff_from_mean.png  deviation from mean spectrum
  - radial_profiles_all_bands.png    1D radial profile overlay
  - manifest.json
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

from band_spectrum import (
    load_bands_from_folder,
    magnitude_spectrum_log,
    radial_mean_profile,
)

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_STACK_DIR = DATA_DIR / "cut_images_all" / "2024-06-25_10-12-29-white"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "band_spectrum"


def _band_to_uint8(band: np.ndarray) -> np.ndarray:
    arr = np.asarray(band, dtype=np.float32)
    if arr.max() <= 1.0 and arr.min() >= 0.0:
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _spectrum_vmax(spec: np.ndarray, percentile: float = 99.5) -> float:
    return max(float(np.percentile(spec, percentile)), 1e-6)


def save_band_spectrum_figures(
    band: np.ndarray,
    spectrum: np.ndarray,
    wavelength,
    out_dir: Path,
    percentile: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    vmax = _spectrum_vmax(spectrum, percentile)
    paths = {}

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(spectrum, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(f"{wavelength} nm - 2D log magnitude spectrum")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, label="log(1 + |F|)")
    p2d = out_dir / f"{wavelength}_spectrum2d.png"
    plt.tight_layout()
    plt.savefig(p2d, dpi=150, bbox_inches="tight")
    plt.close()
    paths["spectrum2d"] = p2d

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(_band_to_uint8(band), cmap="gray")
    axes[0].set_title(f"{wavelength} nm - spatial")
    axes[0].axis("off")
    im = axes[1].imshow(spectrum, cmap="magma", vmin=0, vmax=vmax)
    axes[1].set_title(f"{wavelength} nm - frequency")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, label="log(1 + |F|)")
    p_pair = out_dir / f"{wavelength}_band_spectrum.png"
    plt.tight_layout()
    plt.savefig(p_pair, dpi=150, bbox_inches="tight")
    plt.close()
    paths["band_spectrum"] = p_pair

    np.save(out_dir / f"{wavelength}_spectrum2d.npy", spectrum.astype(np.float32))
    return paths


def _global_spectrum_vmax(spectra: dict, percentile: float = 99.5) -> float:
    stacked = np.concatenate([s.ravel() for s in spectra.values()])
    return max(float(np.percentile(stacked, percentile)), 1e-6)


def save_overview_montage(
    spectra: dict,
    wavelengths: list,
    out_path: Path,
    percentile: float,
    n_cols: int = 6,
    unified_scale: bool = True,
) -> float:
    n = len(wavelengths)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    if unified_scale:
        vmax = _global_spectrum_vmax(spectra, percentile)
    else:
        vmax = None

    im = None
    for idx, wl in enumerate(wavelengths):
        spec = spectra[wl]
        ax = axes[idx]
        local_vmax = vmax if unified_scale else _spectrum_vmax(spec, percentile)
        im = ax.imshow(spec, cmap="magma", vmin=0, vmax=local_vmax)
        ax.set_title(f"{wl} nm", fontsize=9)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    scale_note = "shared color scale" if unified_scale else "per-band color scale"
    fig.suptitle(
        f"2D log-magnitude spectra comparison ({scale_note}, DC at center)",
        fontsize=13,
    )
    if im is not None:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="log(1 + |F|)")
    fig.subplots_adjust(left=0.04, right=0.92, top=0.93, bottom=0.04, wspace=0.08, hspace=0.25)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return vmax if unified_scale else float("nan")


def save_spectrum_diff_from_mean(
    spectra: dict,
    wavelengths: list,
    out_path: Path,
    n_cols: int = 6,
) -> None:
    """Highlight inter-band differences: |spectrum - mean spectrum|."""
    stack = np.stack([spectra[wl] for wl in wavelengths], axis=0)
    mean_spec = stack.mean(axis=0)
    diffs = {wl: np.abs(spectra[wl] - mean_spec) for wl in wavelengths}

    n = len(wavelengths)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    vmax = _global_spectrum_vmax(diffs, percentile=99.5)
    im = None
    for idx, wl in enumerate(wavelengths):
        ax = axes[idx]
        im = ax.imshow(diffs[wl], cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{wl} nm", fontsize=9)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("|spectrum - mean spectrum| across bands", fontsize=13)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="abs diff")
    fig.subplots_adjust(left=0.04, right=0.92, top=0.93, bottom=0.04, wspace=0.08, hspace=0.25)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_spectrum_comparison_strip(
    spectra: dict,
    wavelengths: list,
    out_path: Path,
    percentile: float,
) -> None:
    """All bands in one horizontal strip for wavelength-order comparison."""
    vmax = _global_spectrum_vmax(spectra, percentile)
    n = len(wavelengths)
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 2.4))
    if n == 1:
        axes = [axes]

    im = None
    for ax, wl in zip(axes, wavelengths):
        im = ax.imshow(spectra[wl], cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{wl}", fontsize=7, rotation=90, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("All bands - 2D spectra (shared scale)", fontsize=11, y=1.02)
    fig.subplots_adjust(top=0.82, bottom=0.05, left=0.02, right=0.88, wspace=0.05)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.012, 0.72])
    fig.colorbar(im, cax=cbar_ax, label="log(1+|F|)")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_radial_profiles(
    spectra: dict,
    wavelengths: list,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for wl in wavelengths:
        r, p = radial_mean_profile(spectra[wl])
        ax.plot(r, p, linewidth=1.2, alpha=0.85, label=f"{wl} nm")

    ax.set_xlabel("Normalized radial frequency (0=DC, 1=Nyquist)")
    ax.set_ylabel("Mean log magnitude")
    ax.set_title("Radial mean spectrum across bands")
    ax.grid(True, alpha=0.3)
    if len(wavelengths) <= 12:
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(
            0.02, 0.98,
            f"{len(wavelengths)} bands (legend omitted)",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_band_spectrum_experiment(
    stack_dir,
    output_dir=None,
    wl_min=None,
    wl_max=None,
    image_size=None,
    percentile: float = 99.5,
    exp_name: str = "run",
) -> Path:
    stack_dir = Path(stack_dir)
    bands, wavelengths, band_paths = load_bands_from_folder(
        stack_dir,
        wl_min=wl_min,
        wl_max=wl_max,
        image_size=image_size,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir or DEFAULT_OUTPUT_DIR) / stack_dir.name / f"{exp_name}_{ts}"
    per_band_dir = run_dir / "bands"
    per_band_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Band 2D FFT Spectrum Visualization")
    print(f"Stack: {stack_dir}")
    print(f"Bands ({len(wavelengths)}): {wavelengths}")
    print(f"Output: {run_dir}")
    print("=" * 60)

    spectra = {}
    manifest = {
        "experiment": "band_spectrum",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stack_dir": str(stack_dir.resolve()),
        "wavelengths_nm": [str(w) for w in wavelengths],
        "num_bands": len(wavelengths),
        "image_size": list(image_size) if image_size else None,
        "percentile_vmax": percentile,
        "bands": [],
    }

    for idx, (wl, band, path) in enumerate(zip(wavelengths, bands, band_paths), start=1):
        spec = magnitude_spectrum_log(band)
        spectra[wl] = spec
        band_out = per_band_dir / str(wl)
        saved = save_band_spectrum_figures(
            band, spec, wl, band_out, percentile=percentile,
        )
        r, p = radial_mean_profile(spec)
        np.savez(
            band_out / f"{wl}_radial_profile.npz",
            radius=r,
            profile=p,
        )
        manifest["bands"].append({
            "index": idx,
            "wavelength_nm": str(wl),
            "source_path": str(path),
            "spectrum2d": str(saved["spectrum2d"].relative_to(run_dir)),
            "band_spectrum": str(saved["band_spectrum"].relative_to(run_dir)),
            "shape": list(band.shape),
        })
        print(f"[{idx:02d}/{len(wavelengths)}] {wl} nm -> {saved['band_spectrum'].name}")

    montage_path = run_dir / "all_bands_spectrum_compare.png"
    global_vmax = save_overview_montage(
        spectra, wavelengths, montage_path, percentile, unified_scale=True,
    )

    diff_path = run_dir / "all_bands_spectrum_diff_from_mean.png"
    save_spectrum_diff_from_mean(spectra, wavelengths, diff_path)

    strip_path = run_dir / "all_bands_spectrum_strip.png"
    save_spectrum_comparison_strip(spectra, wavelengths, strip_path, percentile)

    radial_path = run_dir / "radial_profiles_all_bands.png"
    save_radial_profiles(spectra, wavelengths, radial_path)

    manifest["all_bands_spectrum_compare"] = str(montage_path.relative_to(run_dir))
    manifest["all_bands_spectrum_diff_from_mean"] = str(diff_path.relative_to(run_dir))
    manifest["all_bands_spectrum_strip"] = str(strip_path.relative_to(run_dir))
    manifest["global_spectrum_vmax"] = global_vmax
    manifest["radial_profiles_all_bands"] = str(radial_path.relative_to(run_dir))

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nComparison figures:")
    print(f"  {montage_path.name}")
    print(f"  {strip_path.name}")
    print(f"  {diff_path.name}")
    print(f"  {radial_path.name}")
    print(f"\nDone. {len(wavelengths)} band spectra saved under:\n  {run_dir}")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize 2D FFT magnitude spectrum for each band in a stack folder.",
    )
    parser.add_argument("--stack-dir", default=str(DEFAULT_STACK_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--wl-min", type=int, default=None)
    parser.add_argument("--wl-max", type=int, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"), default=None)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--exp-name", default="run")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    image_size = tuple(args.image_size) if args.image_size else None
    run_band_spectrum_experiment(
        stack_dir=args.stack_dir,
        output_dir=args.output_dir,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        image_size=image_size,
        percentile=args.percentile,
        exp_name=args.exp_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
