#!/usr/bin/env python3
"""PIFReg pairwise registration experiment — 标准实验记录。"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.python.experiments.experiment_data import load_pair_images, pairwise_metrics_dict
from src.python.experiments.experiment_recorder import (
    _band_to_uint8,
    create_run_dir,
    describe_pairwise_architecture,
    record_pairwise_experiment,
)
from src.python.registration import register_pifreg

DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pifreg_pairwise"
EXPERIMENT_ID = "pifreg_pairwise"


def run_experiment(
    fixed_path,
    moving_path,
    output_dir=None,
    exp_name="run",
    image_size=(512, 512),
    device="cuda",
    epochs=3000,
    fast_mode=False,
    multiscale=True,
    no_show=True,
):
    base_output = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    run_dir = create_run_dir(base_output, exp_name=exp_name)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    fixed, moving, fixed_raw, moving_raw = load_pair_images(
        fixed_path, moving_path, PROJECT_ROOT, DATA_DIR, image_size=image_size,
    )

    config = {
        "experiment": EXPERIMENT_ID,
        "exp_name": exp_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fixed_image": str(Path(fixed_path).resolve()),
        "moving_image": str(Path(moving_path).resolve()),
        "image_size": list(image_size),
        "device": str(device),
        "registration": {
            "epochs": epochs,
            "fast_mode": fast_mode,
            "multiscale": multiscale,
        },
    }

    print("=" * 60)
    print("PIFReg Pairwise Experiment")
    print(f"Run folder: {run_dir}")

    t0 = time.perf_counter()
    warped, flow = register_pifreg(
        fixed, moving,
        device=device,
        epochs=epochs,
        multiscale=multiscale,
        early_stop=True,
        patience=120,
        lr_schedule="cosine",
        fast_mode=fast_mode,
        return_flow=True,
    )
    elapsed = time.perf_counter() - t0

    # 位移场作用于原图 moving
    warped_raw = _apply_flow_to_raw(moving_raw, flow, device)

    pm = pairwise_metrics_dict(fixed, moving, warped)
    row_before = {"band_index": 0, **pm["before"]}
    row_after = {"band_index": 0, **pm["after"]}
    metrics_before = {"ref_band_index": 0, "per_band": [row_before], "mean": pm["before"]}
    metrics_after = {"ref_band_index": 0, "per_band": [row_after], "mean": pm["after"]}
    metrics_summary = pm["summary"]

    manifest = record_pairwise_experiment(
        run_dir=run_dir,
        config=config,
        architecture_text=describe_pairwise_architecture(image_size, fast_mode, multiscale),
        fixed_raw=fixed_raw,
        moving_raw=moving_raw,
        warped_raw=warped_raw,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        metrics_summary=metrics_summary,
        elapsed_seconds=elapsed,
        flow=flow,
    )

    for key, vals in metrics_summary.items():
        print(f"{key}: {vals['before']:.4f} -> {vals['after']:.4f} (Δ{vals['delta']:+.4f})")
    print(f"\nExperiment saved to: {run_dir}")

    if not no_show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        for i, (img, title) in enumerate(
            [(fixed_raw, "Fixed"), (moving_raw, "Moving"), (warped_raw, "Warped")], start=1,
        ):
            plt.subplot(1, 3, i)
            plt.imshow(_band_to_uint8(img), cmap="gray")
            plt.title(title)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    return warped_raw, manifest


def _apply_flow_to_raw(moving_raw, flow, device):
    from src.python.voxelmorph.layers import SpatialTransformer
    h, w = moving_raw.shape
    m = torch.tensor(moving_raw, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    flow_t = torch.tensor(flow, dtype=torch.float32, device=device).unsqueeze(0)
    transformer = SpatialTransformer((h, w)).to(device)
    with torch.no_grad():
        warped = transformer(m, flow_t)
    return warped.squeeze().cpu().numpy().astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="PIFReg pairwise registration with experiment logging")
    p.add_argument("--fixed", required=True)
    p.add_argument("--moving", required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--exp-name", type=str, default="run")
    p.add_argument("--image-size", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--fast-mode", action="store_true")
    p.add_argument("--no-multiscale", action="store_true")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        fixed_path=args.fixed,
        moving_path=args.moving,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        image_size=tuple(args.image_size),
        device=args.device,
        epochs=args.epochs,
        fast_mode=args.fast_mode,
        multiscale=not args.no_multiscale,
        no_show=not args.show,
    )
