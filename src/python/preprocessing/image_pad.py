"""Bottom-right zero padding for VoxelMorph (U-Net requires divisibility by 16)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

VOXMORPH_PAD_DIVISOR = 16


def ceil_to_divisor(n: int, divisor: int = VOXMORPH_PAD_DIVISOR) -> int:
    return int(np.ceil(n / divisor) * divisor)


@dataclass(frozen=True)
class PadInfo:
    orig_h: int
    orig_w: int
    canvas_h: int
    canvas_w: int

    @property
    def pad_bottom(self) -> int:
        return self.canvas_h - self.orig_h

    @property
    def pad_right(self) -> int:
        return self.canvas_w - self.orig_w


def pad_info_from_shape(
    orig_h: int,
    orig_w: int,
    canvas_h: Optional[int] = None,
    canvas_w: Optional[int] = None,
    divisor: int = VOXMORPH_PAD_DIVISOR,
) -> PadInfo:
    if canvas_h is None:
        canvas_h = ceil_to_divisor(orig_h, divisor)
    if canvas_w is None:
        canvas_w = ceil_to_divisor(orig_w, divisor)
    return PadInfo(orig_h, orig_w, canvas_h, canvas_w)


def pad_bottom_right(
    img: np.ndarray,
    canvas_h: int,
    canvas_w: int,
    fill: float = 0.0,
) -> np.ndarray:
    """Pad with black border on bottom and right; content stays top-left aligned."""
    img = np.asarray(img)
    h, w = img.shape[:2]
    if h > canvas_h or w > canvas_w:
        raise ValueError(
            f'Image ({h}, {w}) exceeds canvas ({canvas_h}, {canvas_w}); cannot pad.',
        )
    if h == canvas_h and w == canvas_w:
        return np.asarray(img, dtype=np.float32 if img.dtype != np.uint8 else img.dtype)

    out = np.full((canvas_h, canvas_w), fill, dtype=np.float32)
    out[:h, :w] = img.astype(np.float32, copy=False)
    return out


def crop_to_original(img: np.ndarray, pad_info: PadInfo) -> np.ndarray:
    return np.asarray(img)[: pad_info.orig_h, : pad_info.orig_w]


def crop_flow(flow: np.ndarray, pad_info: PadInfo) -> np.ndarray:
    """Remove padding region from a (2, H, W) displacement field."""
    flow = np.asarray(flow, dtype=np.float32)
    return flow[:, : pad_info.orig_h, : pad_info.orig_w].copy()


def scan_image_shape(path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Failed to read image: {path}')
    return img.shape[:2]


def scan_folders_max_shape(folders: Sequence) -> Tuple[int, int]:
    max_h, max_w = 0, 0
    for folder in folders:
        folder = Path(folder)
        for pattern in ('*.jpeg', '*.jpg'):
            for path in folder.glob(pattern):
                if path.stat().st_size <= 0:
                    continue
                try:
                    h, w = scan_image_shape(path)
                except ValueError:
                    continue
                max_h = max(max_h, h)
                max_w = max(max_w, w)
    if max_h <= 0 or max_w <= 0:
        raise ValueError('Could not infer any image shapes from folders.')
    return max_h, max_w


def resolve_voxelmorph_canvas(
    folders: Sequence,
    image_size: Optional[Tuple[int, int]] = None,
    divisor: int = VOXMORPH_PAD_DIVISOR,
) -> Tuple[int, int]:
    """
    Return (canvas_h, canvas_w) for model I/O.

    image_size: OpenCV (width, height); default 512×512 (no native-resolution scan).
    """
    from src.python.experiments.experiment_data import resolve_image_size

    w, h = resolve_image_size(image_size)
    return ceil_to_divisor(h, divisor), ceil_to_divisor(w, divisor)
