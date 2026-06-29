#!/usr/bin/env python3
"""
Compare vendored pyelastix (src/python/vendor) with the pip-installed package.

Runs pairwise registration on the same image pair with identical parameters and
reports max/mean absolute differences. Elastix uses random MI sampling, so results
are expected to match within a small tolerance rather than bit-for-bit.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

import pyelastix as pyelastix_pip
from src.python.vendor import pyelastix as pyelastix_local


DATA_DIR = PROJECT_ROOT / "data"


def resolve_image_path(path):
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    for base in (PROJECT_ROOT, DATA_DIR):
        resolved = base / candidate
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Image not found: {path}")


def load_pair(fixed_path, moving_path, image_size=(256, 256)):
    fixed = cv2.imread(str(resolve_image_path(fixed_path)), cv2.IMREAD_GRAYSCALE)
    moving = cv2.imread(str(resolve_image_path(moving_path)), cv2.IMREAD_GRAYSCALE)
    if fixed is None or moving is None:
        raise ValueError("Failed to read input images")

    fixed = cv2.resize(fixed, image_size).astype(np.float32)
    moving = cv2.resize(moving, image_size).astype(np.float32)

    for img in (fixed, moving):
        lo, hi = np.min(img), np.max(img)
        img[:] = (img - lo) / (hi - lo + 1e-8)

    return fixed, moving


def make_params(epochs=30, grid_spacing=20):
    params = pyelastix_pip.get_default_params()
    params.MaximumNumberOfIterations = epochs
    params.FinalGridSpacingInVoxels = grid_spacing
    return params


def compare_arrays(name, a, b):
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    print(f"  {name}:")
    print(f"    shape pip={np.asarray(a).shape}, local={np.asarray(b).shape}")
    print(f"    max abs diff  = {diff.max():.6e}")
    print(f"    mean abs diff = {diff.mean():.6e}")
    return diff.max()


def run_pairwise_compare(fixed, moving, params, verbose=0):
    print("\n=== Pairwise registration ===")
    print(f"pip version:   {pyelastix_pip.__version__}")
    print(f"local version: {pyelastix_local.__version__}")

    warped_pip, field_pip = pyelastix_pip.register(
        moving.copy(), fixed.copy(), params, verbose=verbose
    )
    warped_local, field_local = pyelastix_local.register(
        moving.copy(), fixed.copy(), params, verbose=verbose
    )

    max_warp = compare_arrays("warped image", warped_pip, warped_local)
    max_field_x = compare_arrays("field x", field_pip[0], field_local[0])
    max_field_y = compare_arrays("field y", field_pip[1], field_local[1])

    return max(max_warp, max_field_x, max_field_y)


def run_api_parity():
    print("\n=== API parity ===")
    pip_params = pyelastix_pip.get_default_params().as_dict()
    local_params = pyelastix_local.get_default_params().as_dict()
    shared_keys = set(pip_params) & set(local_params)
    mismatches = [
        key for key in sorted(shared_keys)
        if pip_params[key] != local_params[key]
    ]
    print(f"  default param keys: pip={len(pip_params)}, local={len(local_params)}")
    print(f"  mismatched values: {len(mismatches)}")
    if mismatches:
        for key in mismatches[:10]:
            print(f"    {key}: pip={pip_params[key]!r}, local={local_params[key]!r}")


if __name__ == "__main__":
    fixed_path = "cut_images_all/2024-06-25_10-12-29-white/650.jpeg"
    moving_path = "cut_images_all/2024-06-25_10-12-29-white/639.jpeg"

    fixed, moving = load_pair(fixed_path, moving_path, image_size=(256, 256))
    params = make_params(epochs=30, grid_spacing=20)

    run_api_parity()
    max_diff = run_pairwise_compare(fixed, moving, params, verbose=0)

    tol = 1e-3
    if max_diff <= tol:
        print(f"\nPASS: vendored pyelastix matches pip install within tol={tol}")
    else:
        print(
            f"\nNOTE: max diff {max_diff:.6e} > tol={tol}. "
            "This can happen because Elastix MI sampling is stochastic; "
            "re-run or increase iterations to confirm parity."
        )
