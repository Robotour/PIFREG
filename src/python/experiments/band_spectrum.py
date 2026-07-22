# -*- coding: utf-8 -*-
"""2D FFT magnitude spectrum utilities for HSI band images."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.fft import fft2, fftshift

from pairwise_heatmap import (
    IMAGE_EXTENSIONS,
    _band_sort_key,
    load_band_image,
    wavelength_from_path,
)


def list_band_files(
    stack_dir,
    wl_min: Optional[int] = None,
    wl_max: Optional[int] = None,
) -> List[Path]:
    folder = Path(stack_dir)
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No band images found in {folder}")

    files = sorted(files, key=_band_sort_key)
    if wl_min is not None or wl_max is not None:
        filtered = []
        for path in files:
            try:
                wl = int(path.stem)
            except ValueError:
                continue
            if wl_min is not None and wl < wl_min:
                continue
            if wl_max is not None and wl > wl_max:
                continue
            filtered.append(path)
        files = filtered

    if not files:
        raise FileNotFoundError(f"No band images left after wavelength filter in {folder}")
    return files


def load_bands_from_folder(
    stack_dir,
    wl_min: Optional[int] = None,
    wl_max: Optional[int] = None,
    image_size: Optional[Tuple[int, int]] = None,
):
    band_paths = list_band_files(stack_dir, wl_min=wl_min, wl_max=wl_max)
    bands = []
    wavelengths = []
    for path in band_paths:
        wavelengths.append(wavelength_from_path(path))
        bands.append(load_band_image(path, image_size=image_size))
    return bands, wavelengths, band_paths


def magnitude_spectrum_log(
    image: np.ndarray,
    log_scale: bool = True,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Shifted 2D magnitude spectrum of a single band.

    DC component is at the image center after fftshift.
    """
    arr = np.asarray(image, dtype=np.float64)
    arr = arr - arr.mean()
    spec = fftshift(fft2(arr))
    mag = np.abs(spec)
    if log_scale:
        return np.log1p(mag).astype(np.float32)
    return mag.astype(np.float32)


def radial_mean_profile(spectrum2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean log-magnitude vs normalized radial frequency radius in [0, 1]."""
    h, w = spectrum2d.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = radius.max()
    if max_r <= 0:
        return np.array([0.0]), np.array([float(spectrum2d.mean())])

    n_bins = min(h, w) // 2
    bins = np.linspace(0, max_r, n_bins + 1)
    radii_norm = []
    profile = []
    for i in range(n_bins):
        mask = (radius >= bins[i]) & (radius < bins[i + 1])
        if not np.any(mask):
            continue
        radii_norm.append(0.5 * (bins[i] + bins[i + 1]) / max_r)
        profile.append(float(spectrum2d[mask].mean()))
    return np.asarray(radii_norm, dtype=np.float32), np.asarray(profile, dtype=np.float32)
