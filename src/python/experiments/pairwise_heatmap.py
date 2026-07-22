"""Pairwise difference / RMSE heatmap utilities for adjacent HSI bands."""
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.exposure import match_histograms


DiffFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
PrepFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def _compute_mi(image1: np.ndarray, image2: np.ndarray, bins: int = 256) -> float:
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py)
    non_zero = pxy > 0
    return float(np.sum(pxy[non_zero] * np.log(pxy[non_zero] / px_py[non_zero])))


@dataclass(frozen=True)
class HeatmapMethod:
    key: str
    title: str
    description: str
    prep: PrepFn
    metric: DiffFn


def _to_float(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)


def _minmax01(img: np.ndarray) -> np.ndarray:
    arr = _to_float(img)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _zscore(img: np.ndarray) -> np.ndarray:
    arr = _to_float(img)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    arr = _to_float(img)
    if arr.max() <= 1.0 and arr.min() >= 0.0:
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _prep_identity(fixed: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _to_float(fixed), _to_float(moving)


def _prep_minmax(fixed: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _minmax01(fixed), _minmax01(moving)


def _prep_hist_eq(fixed: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    f = cv2.equalizeHist(_to_uint8(fixed)).astype(np.float32)
    m = cv2.equalizeHist(_to_uint8(moving)).astype(np.float32)
    return f, m


def _prep_hist_match(fixed: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    f = _minmax01(fixed)
    try:
        m = match_histograms(_minmax01(moving), f, channel_axis=None).astype(np.float32)
    except TypeError:
        m = match_histograms(_minmax01(moving), f, multichannel=False).astype(np.float32)
    return f, m


def _prep_clahe(fixed: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _apply(img: np.ndarray) -> np.ndarray:
        return clahe.apply(_to_uint8(img)).astype(np.float32)

    return _apply(fixed), _apply(moving)


def _prep_zscore(fixed: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _zscore(fixed), _zscore(moving)


def _prep_gradient(fixed: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def _grad(img: np.ndarray) -> np.ndarray:
        arr = _minmax01(img)
        gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy).astype(np.float32)

    return _grad(fixed), _grad(moving)


def _metric_mse_map(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    diff = _to_float(fixed) - _to_float(moving)
    return (diff * diff).astype(np.float32)


def _metric_mae_map(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    return np.abs(_to_float(fixed) - _to_float(moving)).astype(np.float32)


def local_ncc_dissimilarity(
    fixed: np.ndarray,
    moving: np.ndarray,
    win: int = 9,
) -> np.ndarray:
    """1 - local NCC; high values indicate local structure mismatch."""
    a = _minmax01(fixed)
    b = _minmax01(moving)
    size = (win, win)
    ma = uniform_filter(a, size)
    mb = uniform_filter(b, size)
    ma2 = uniform_filter(a * a, size)
    mb2 = uniform_filter(b * b, size)
    mab = uniform_filter(a * b, size)
    va = np.maximum(ma2 - ma * ma, 0.0)
    vb = np.maximum(mb2 - mb * mb, 0.0)
    cov = mab - ma * mb
    ncc = cov / (np.sqrt(va * vb) + 1e-8)
    return (1.0 - np.clip(ncc, -1.0, 1.0)).astype(np.float32)


def local_mi_dissimilarity(
    fixed: np.ndarray,
    moving: np.ndarray,
    patch: int = 32,
    stride: int = 16,
    bins: int = 32,
) -> np.ndarray:
    """
    Patch-wise MI dissimilarity map.
    Low MI -> high dissimilarity; result is normalized to [0, 1] per image pair.
    """
    a = _minmax01(fixed)
    b = _minmax01(moving)
    h, w = a.shape
    mi_sum = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    for y0 in range(0, max(h - patch + 1, 1), stride):
        for x0 in range(0, max(w - patch + 1, 1), stride):
            y1 = min(y0 + patch, h)
            x1 = min(x0 + patch, w)
            pa = a[y0:y1, x0:x1]
            pb = b[y0:y1, x0:x1]
            mi = _compute_mi(pa, pb, bins=bins)
            mi_sum[y0:y1, x0:x1] += mi
            weight[y0:y1, x0:x1] += 1.0

    mi_map = mi_sum / np.maximum(weight, 1.0)
    mi_min = float(mi_map.min())
    mi_max = float(mi_map.max())
    if mi_max - mi_min < 1e-8:
        return np.zeros((h, w), dtype=np.float32)
    norm_mi = (mi_map - mi_min) / (mi_max - mi_min)
    return (1.0 - norm_mi).astype(np.float32)


def default_heatmap_methods() -> List[HeatmapMethod]:
    return [
        HeatmapMethod(
            key="raw_mse",
            title="Raw MSE",
            description="Direct squared difference on original intensity.",
            prep=_prep_identity,
            metric=_metric_mse_map,
        ),
        HeatmapMethod(
            key="minmax_mse",
            title="MinMax MSE",
            description="Per-band min-max normalization, then squared difference.",
            prep=_prep_minmax,
            metric=_metric_mse_map,
        ),
        HeatmapMethod(
            key="hist_eq_mse",
            title="HistEq MSE",
            description="Histogram equalization on each band, then squared difference.",
            prep=_prep_hist_eq,
            metric=_metric_mse_map,
        ),
        HeatmapMethod(
            key="hist_match_mse",
            title="HistMatch MSE",
            description="Histogram-match moving to fixed, then squared difference.",
            prep=_prep_hist_match,
            metric=_metric_mse_map,
        ),
        HeatmapMethod(
            key="clahe_mse",
            title="CLAHE MSE",
            description="CLAHE contrast normalization, then squared difference.",
            prep=_prep_clahe,
            metric=_metric_mse_map,
        ),
        HeatmapMethod(
            key="zscore_mse",
            title="ZScore MSE",
            description="Per-band z-score normalization, then squared difference.",
            prep=_prep_zscore,
            metric=_metric_mse_map,
        ),
        HeatmapMethod(
            key="gradient_mse",
            title="Gradient MSE",
            description="Sobel gradient magnitude after min-max, then squared difference.",
            prep=_prep_gradient,
            metric=_metric_mse_map,
        ),
        HeatmapMethod(
            key="local_ncc",
            title="Local NCC Dissim",
            description="1 - local normalized cross-correlation (structure mismatch).",
            prep=_prep_minmax,
            metric=lambda f, m: local_ncc_dissimilarity(f, m, win=9),
        ),
        HeatmapMethod(
            key="local_mi",
            title="Local MI Dissim",
            description="1 - patch-wise normalized mutual information.",
            prep=_prep_minmax,
            metric=lambda f, m: local_mi_dissimilarity(f, m, patch=32, stride=16, bins=32),
        ),
    ]


def compute_heatmap(
    fixed: np.ndarray,
    moving: np.ndarray,
    method: HeatmapMethod,
) -> np.ndarray:
    fixed_p, moving_p = method.prep(fixed, moving)
    return method.metric(fixed_p, moving_p)


def compute_all_heatmaps(
    fixed: np.ndarray,
    moving: np.ndarray,
    methods: Optional[Sequence[HeatmapMethod]] = None,
) -> Dict[str, np.ndarray]:
    methods = list(methods or default_heatmap_methods())
    return {m.key: compute_heatmap(fixed, moving, m) for m in methods}


def heatmap_display_vmax(
    heatmap: np.ndarray,
    percentile: float = 99.0,
    floor: float = 1e-8,
) -> float:
    arr = np.asarray(heatmap, dtype=np.float64)
    if arr.size == 0:
        return floor
    return max(float(np.percentile(arr, percentile)), floor)


def load_band_image(path, image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = img.astype(np.float32)
    if image_size is not None:
        img = cv2.resize(img, image_size)
    return img


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _band_sort_key(path) -> tuple:
    stem = Path(path).stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def discover_band_files(
    stack_dir,
    wl_min: Optional[int] = None,
    wl_max: Optional[int] = None,
):
    """Scan folder for band images; sort by numeric stem (wavelength nm)."""
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

    if len(files) < 2:
        raise ValueError(
            f"Need at least 2 band images for pairwise heatmaps, found {len(files)} in {folder}"
        )
    return files


def wavelength_from_path(path) -> int | str:
    try:
        return int(Path(path).stem)
    except ValueError:
        return Path(path).stem


def load_wavelength_stack(
    stack_dir,
    wl_min: Optional[int] = None,
    wl_max: Optional[int] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[List[np.ndarray], List, List]:
    """
    Load bands from files present in stack_dir (sorted by wavelength stem).

    By default uses every image in the folder; wl_min/wl_max optionally filter.
    """
    band_paths = discover_band_files(stack_dir, wl_min=wl_min, wl_max=wl_max)
    bands = []
    wavelengths = []
    for path in band_paths:
        wavelengths.append(wavelength_from_path(path))
        bands.append(load_band_image(path, image_size=image_size))
    return bands, wavelengths, band_paths


def adjacent_pairs(wavelengths: Sequence[int]) -> List[Tuple[int, int]]:
    wls = list(wavelengths)
    return [(wls[i], wls[i + 1]) for i in range(len(wls) - 1)]
