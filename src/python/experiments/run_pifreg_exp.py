#!/usr/bin/env python3
"""PIFReg pairwise registration experiment (GitHub-maintained entry point)."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.python.metrics import compute_MI, compute_NCC, compute_NTG
from src.python.registration import register_pifreg

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


def load_images(fixed_path, moving_path, image_size=(512, 512)):
    fixed = cv2.imread(str(resolve_image_path(fixed_path)), cv2.IMREAD_GRAYSCALE)
    moving = cv2.imread(str(resolve_image_path(moving_path)), cv2.IMREAD_GRAYSCALE)
    if fixed is None or moving is None:
        raise ValueError("Failed to read input images")

    fixed = fixed.astype(np.float32)
    moving = moving.astype(np.float32)
    if image_size is not None:
        fixed = cv2.resize(fixed, image_size)
        moving = cv2.resize(moving, image_size)

    fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())
    moving = (moving - moving.min()) / (moving.max() - moving.min())
    return fixed, moving


def main():
    parser = argparse.ArgumentParser(description="PIFReg pairwise registration")
    parser.add_argument("--fixed", required=True, help="Fixed image path")
    parser.add_argument("--moving", required=True, help="Moving image path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--no-multiscale", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    fixed, moving = load_images(args.fixed, args.moving)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    warped = register_pifreg(
        fixed, moving,
        device=device,
        epochs=args.epochs,
        multiscale=not args.no_multiscale,
        early_stop=True,
        patience=120,
        lr_schedule="cosine",
    )

    print(f"MI:  {compute_MI(fixed, moving):.4f} -> {compute_MI(fixed, warped):.4f}")
    print(f"NCC: {compute_NCC(fixed, moving):.4f} -> {compute_NCC(fixed, warped):.4f}")
    print(f"NTG: {compute_NTG(fixed, moving):.4f} -> {compute_NTG(fixed, warped):.4f}")

    if not args.no_show:
        plt.figure(figsize=(12, 4))
        for i, (img, title) in enumerate(
            [(fixed, "Fixed"), (moving, "Moving"), (warped, "Warped")], start=1
        ):
            plt.subplot(1, 3, i)
            plt.imshow(img, cmap="gray")
            plt.title(title)
            plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
