---
name: voxelmorph-hsi-training
description: >-
  Train and evaluate unsupervised VoxelMorph on HSI tongue stacks (cut_images_all).
  Supports baseline pairwise vs stack_spatial method, isolated run folders, best.pt
  checkpoint, full test eval and RGB visualization. Use when the user asks to train
  VoxelMorph, compare registration methods, run voxelmorph experiments on Linux
  workstation, or visualize test-set fake RGB after training.
---

# VoxelMorph HSI Training Workflow

## Methods (separate functions)

| `--method` | Python function | Description |
|------------|-----------------|-------------|
| `baseline` | `train_voxelmorph_baseline()` | Random adjacent-band pairs, uniform loss |
| `stack_spatial` | `train_voxelmorph_stack_spatial()` | Session sub-chain + edge-high spatial weights |

Code: `src/python/voxelmorph/training.py`

## Run directory layout

Each experiment gets its own folder (never overwrite prior runs):

```
outputs/voxelmorph_runs/{method}/{exp_name}_{timestamp}/
  config.json
  split_manifest.json
  train_history.json
  best_info.json
  test_metrics.json
  run_summary.json
  checkpoints/
    best.pt      # best validation epoch — use this for eval/viz
    final.pt
    0020.pt ...
  visualizations/
    01_{session}/rgb_overview.png
    index.json
```

## One-command pipeline (train + eval + visualize all test sessions)

From repo root on Linux:

### Baseline (comparison)

```bash
python src/python/experiments/train_voxelmorph.py \
  --method baseline \
  --exp-name baseline_v1 \
  --data-dir data/cut_images_all \
  --train-ratio 0.7 \
  --epochs 1000 \
  --steps-per-epoch 80 \
  --val-interval 20 \
  --image-loss ncc \
  --device cuda
```

### Proposed method (stack + spatial weights)

```bash
python src/python/experiments/train_voxelmorph.py \
  --method stack_spatial \
  --exp-name stack_spatial_v1 \
  --data-dir data/cut_images_all \
  --train-ratio 0.7 \
  --epochs 1000 \
  --steps-per-epoch 80 \
  --val-interval 20 \
  --subchain-len 6 \
  --image-loss mse \
  --smooth-flow-sigma 1.5 \
  --device cuda
```

## Re-visualize an existing run (best.pt)

```bash
python src/python/experiments/visualize_voxelmorph_test.py \
  --run-dir outputs/voxelmorph_runs/baseline/baseline_v1_YYYYMMDD_HHMMSS \
  --all-test-sessions \
  --device cuda
```

## Eval-only (skip training)

```bash
python src/python/experiments/train_voxelmorph.py \
  --method baseline \
  --exp-name baseline_v1 \
  --eval-only \
  --run-dir outputs/voxelmorph_runs/baseline/baseline_v1_YYYYMMDD_HHMMSS
```

## Validation metrics

| Method | Val metric | Saved as |
|--------|------------|----------|
| baseline | pairwise `NCC_after` | `checkpoints/best.pt` |
| stack_spatial | stack chain `NCC_after_mean` | `checkpoints/best.pt` |

## Comparing experiments

1. Run baseline and stack_spatial with different `--exp-name` (auto timestamp subfolder).
2. Compare `test_metrics.json` → `stack_eval.summary` (primary for RGB quality).
3. Compare `visualizations/*/rgb_overview.png` side by side.

## Key scripts

| File | Role |
|------|------|
| `src/python/experiments/train_voxelmorph.py` | CLI entry |
| `src/python/voxelmorph/experiment.py` | Run dir + full pipeline |
| `src/python/voxelmorph/training.py` | `train_voxelmorph_baseline`, `train_voxelmorph_stack_spatial` |
| `src/python/experiments/visualize_voxelmorph_test.py` | Fake RGB visualization |

## Environment

```bash
conda activate dxtorch   # or any env with torch, cv2, scipy
cd /path/to/24_hyper_registration
```

## Agent checklist

When user changes hyperparameters:

- [ ] Use new `--exp-name` (timestamp folder is automatic)
- [ ] Keep `--method baseline` vs `stack_spatial` explicit for fair comparison
- [ ] After training, confirm `checkpoints/best.pt` exists
- [ ] Confirm `visualizations/` contains all test sessions when pipeline finished
- [ ] Report `test_metrics.json` stack_eval NCC before/after
