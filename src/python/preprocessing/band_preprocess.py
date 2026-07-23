"""Band-level preprocessing shared by registration experiments."""

from __future__ import annotations

import cv2
import numpy as np


def histogram_equalize_band(img: np.ndarray) -> np.ndarray:
    """Per-band histogram equalization (uint8 EQ), returned as float32 in [0, 255]."""
    img = np.asarray(img, dtype=np.float32)
    lo, hi = float(np.min(img)), float(np.max(img))
    if hi > lo:
        u8 = ((img - lo) / (hi - lo) * 255.0).astype(np.uint8)
    else:
        u8 = np.zeros_like(img, dtype=np.uint8)
    return cv2.equalizeHist(u8).astype(np.float32)


def refresh_histogram_equalized(raw_band: np.ndarray) -> np.ndarray:
    """Recompute hist-eq from a (possibly warped) raw band for chain fixed/moving inputs."""
    return histogram_equalize_band(raw_band)
