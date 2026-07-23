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

**Default image size: 512×512** (do not downsample to 256 unless debugging).

## Metrics comparison CSV (one file per random seed)

Path: `outputs/metrics_tables/seed_{seed}.csv`

| Row | method | stage | Meaning |
|-----|--------|-------|---------|
| 1 | `unregistered` | `before` | Test set, all C(N,2) band-pair metrics averaged |
| 2+ | `stackreg_chain`, `voxelmorph_baseline`, … | `after` | After registration, same metric definition |

Metrics per row: **MI, NMI, NCC, NTG, MSE** — for each session, compute every unordered band pair (≈435 pairs for 30 bands), average; then average over test sessions.

```bash
# Row 1 only (unregistered baseline)
python src/python/experiments/run_append_metrics_csv.py \
  --init-only --seed 42 --data-dir data/cut_images_all

# Classical / VoxelMorph eval scripts append rows automatically (default CSV path)
# Manual append from an existing run:
python src/python/experiments/run_append_metrics_csv.py \
  --method voxelmorph_stack_spatial \
  --from-run-dir outputs/voxelmorph_runs/stack_spatial/stack_spatial_v1_YYYYMMDD_HHMMSS \
  --seed 42 --overwrite
```

Code: `src/python/experiments/stack_pairwise_metrics.py`, `metrics_csv.py`

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
4. Run classical baselines on the **same test split** (see below).

## Classical baselines on test set (Elastix / StackReg / KEREN)

Script: `src/python/experiments/run_classical_baseline_eval.py`  
Code: `src/python/registration/classical_stack.py`

| `--method` | Description |
|------------|-------------|
| `elastix_groupwise` | Full-stack Elastix BSplineStackTransform |
| `elastix_chain` | Pairwise Elastix along wavelength chain (like VoxelMorph inference) |
| `stackreg_chain` | Pairwise StackReg bilinear chain |
| `keren` | KEREN pyramid LK (translation + rotation) |
| `all` | Run all four sequentially |

Output: `outputs/classical_baselines/{method}/{exp_name}_{timestamp}/test_metrics.json`

**Fair comparison**: reuse VoxelMorph split via `--split-from-run-dir`.

```bash
# StackReg (fast sanity check)
python src/python/experiments/run_classical_baseline_eval.py \
  --method stackreg_chain \
  --exp-name stackreg_v1 \
  --split-from-run-dir outputs/voxelmorph_runs/baseline/baseline_v1_YYYYMMDD_HHMMSS \
  --visualize

# KEREN
python src/python/experiments/run_classical_baseline_eval.py \
  --method keren \
  --exp-name keren_v1 \
  --split-from-run-dir outputs/voxelmorph_runs/baseline/baseline_v1_YYYYMMDD_HHMMSS

# Elastix chain — pairwise chain, same graph as VoxelMorph (very slow; needs elastix.exe)
python src/python/experiments/run_classical_baseline_eval.py \
  --method elastix_chain \
  --exp-name elastix_chain_v1 \
  --split-from-run-dir outputs/voxelmorph_runs/baseline/baseline_v1_YYYYMMDD_HHMMSS \
  --elastix-epochs 20 \
  --visualize

# Elastix groupwise (slow; needs elastix.exe)
python src/python/experiments/run_classical_baseline_eval.py \
  --method elastix_groupwise \
  --exp-name elastix_gw_v1 \
  --split-from-run-dir outputs/voxelmorph_runs/baseline/baseline_v1_YYYYMMDD_HHMMSS \
  --elastix-epochs 80

# All classical methods
python src/python/experiments/run_classical_baseline_eval.py \
  --method all \
  --exp-name classical_compare_v1 \
  --split-from-run-dir outputs/voxelmorph_runs/baseline/baseline_v1_YYYYMMDD_HHMMSS
```

Debug with fewer sessions: `--max-sessions 5`

Compare metrics: `test_metrics.json` → `stack_eval.summary.NCC_after_mean` (same field as VoxelMorph).

## Key scripts

| File | Role |
|------|------|
| `src/python/experiments/train_voxelmorph.py` | VoxelMorph CLI entry |
| `src/python/experiments/run_classical_baseline_eval.py` | Classical baseline test eval |
| `src/python/voxelmorph/experiment.py` | Run dir + full pipeline |
| `src/python/voxelmorph/training.py` | `train_voxelmorph_baseline`, `train_voxelmorph_stack_spatial` |
| `src/python/registration/classical_stack.py` | Whole-stack Elastix/StackReg/KEREN |
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
