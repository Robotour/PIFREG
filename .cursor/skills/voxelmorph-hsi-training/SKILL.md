---
name: voxelmorph-hsi-training
description: >-
  Train and evaluate unsupervised VoxelMorph on HSI tongue stacks (cut_images_all).
  Supports baseline pairwise vs stack_spatial method, isolated run folders, best.pt
  checkpoint, full test eval, per-band/flow exports, and RGB visualization. Use when
  the user asks to train VoxelMorph, compare registration methods, run voxelmorph
  experiments on Linux workstation, or visualize test-set fake RGB after training.
---

# VoxelMorph HSI Training Workflow

**Default image size: 512×512** (OpenCV `--image-size 512 512`). All bands are resized before training, registration, metrics, and export. Constant `DEFAULT_IMAGE_SIZE` in `src/python/experiments/experiment_data.py`.

**Default preprocessing: per-band histogram equalization** for registration optimization only.
Displacement / global transforms are applied to **raw grayscale at 512×512** (`bands_raw`), not to hist-eq images.
Chain methods refresh hist-eq from warped raw before the next pairwise step.

Utility: `src/python/preprocessing/band_preprocess.py` (`histogram_equalize_band`, `refresh_histogram_equalized`).

## Environment

```bash
conda activate dxtorch
cd /path/to/24_hyper_registration
```

## Metrics comparison CSV + unregistered baseline

Path: `outputs/metrics_tables/seed_{seed}.csv`

| Row | method | stage | Meaning |
|-----|--------|-------|---------|
| 1 | `unregistered` | `before` | Test set, all C(N,2) band-pair metrics averaged |
| 2+ | `stackreg_chain`, `voxelmorph_baseline`, … | `after` | After registration, same metric definition |

Metrics per row: **MI, NMI, NCC, NTG, MSE** — for each session, compute every unordered band pair (≈435 pairs for 30 bands), average; then average over test sessions.

**Unregistered JSON report** (written automatically when running eval scripts):

```
outputs/metrics_tables/seed_{seed}_unregistered.json
```

Both VoxelMorph and classical eval scripts print `Before (unregistered)` / `After (registered)` / `Delta` to the terminal.

```bash
# Row 1 only (unregistered baseline, optional standalone init)
python src/python/experiments/run_append_metrics_csv.py \
  --init-only --seed 42 --data-dir data/cut_images_all
```

Code: `src/python/experiments/stack_pairwise_metrics.py`, `metrics_csv.py`, `session_outputs.py`

## Per-session exports (bands + displacement fields)

Each test session saves **raw grayscale bands before/after** and **flow visualizations**:

```
# VoxelMorph (under visualizations/)
visualizations/01_{session}/
  bands/before/{wavelength}.jpeg
  bands/after/{wavelength}.jpeg
  flows/flow_stack.npy
  flows/color/{wavelength}_flow.png
  flows/magnitude/{wavelength}_magnitude.png
  session_outputs.json
  rgb_overview.png
  images/rgb_*.png

# Classical (under session_exports/)
session_exports/01_{session}/
  bands/before/*.jpeg
  bands/after/*.jpeg
  flows/...          # dense flow (Elastix / StackReg chain / VoxelMorph-style)
  transforms.json    # StackReg / KEREN (rigid, no dense flow)
  session_outputs.json
```

Classical: `--save-outputs` is **on by default**; use `--no-save-outputs` to skip.
VoxelMorph: bands/flows saved automatically during `train_voxelmorph.py` pipeline and `visualize_voxelmorph_test.py`.

## Methods (separate functions)

| `--method` | Python function | Description |
|------------|-----------------|-------------|
| `baseline` | `train_voxelmorph_baseline()` | Random adjacent-band pairs, uniform loss |
| `stack_spatial` | `train_voxelmorph_stack_spatial()` | Session sub-chain + edge-high spatial weights |

Code: `src/python/voxelmorph/training.py`

## Run directory layout

```
outputs/voxelmorph_runs/{method}/{exp_name}_{timestamp}/
  config.json
  split_manifest.json
  train_history.json
  best_info.json
  test_metrics.json          # includes all_pairs_eval summary_before/after
  run_summary.json
  checkpoints/best.pt
  visualizations/01_{session}/bands|flows|rgb_*
  visualizations/index.json
```

## Recommended command sequence (fair comparison)

Replace `YYYYMMDD_HHMMSS` with your actual run folder timestamp.

### Step 0 — optional: init CSV row 1 only

```bash
python src/python/experiments/run_append_metrics_csv.py \
  --init-only --seed 42 --data-dir data/cut_images_all
```

### Step 1 — VoxelMorph baseline (train + eval + bands/flows/RGB)

```bash
python src/python/experiments/train_voxelmorph.py \
  --method baseline \
  --exp-name baseline_v1 \
  --data-dir data/cut_images_all \
  --train-ratio 0.7 \
  --seed 42 \
  --epochs 1000 \
  --steps-per-epoch 80 \
  --val-interval 20 \
  --image-loss ncc \
  --device cuda
```

Outputs:
- `outputs/metrics_tables/seed_42_unregistered.json` (before metrics)
- `outputs/metrics_tables/seed_42.csv` (row 1 unregistered + row voxelmorph_baseline)
- `outputs/voxelmorph_runs/baseline/baseline_v1_*/visualizations/*/bands|flows`

### Step 2 — VoxelMorph stack_spatial (your method)

```bash
python src/python/experiments/train_voxelmorph.py \
  --method stack_spatial \
  --exp-name stack_spatial_v1 \
  --data-dir data/cut_images_all \
  --train-ratio 0.7 \
  --seed 42 \
  --epochs 1000 \
  --steps-per-epoch 80 \
  --val-interval 20 \
  --subchain-len 6 \
  --image-loss mse \
  --smooth-flow-sigma 1.5 \
  --device cuda
```

### Step 3 — Classical baselines (same test split as Step 1)

```bash
VM_RUN=outputs/voxelmorph_runs/baseline/baseline_v1_YYYYMMDD_HHMMSS

# StackReg chain (fast sanity check)
python src/python/experiments/run_classical_baseline_eval.py \
  --method stackreg_chain \
  --exp-name stackreg_v1 \
  --split-from-run-dir "$VM_RUN" \
  --seed 42 \
  --visualize

# KEREN
python src/python/experiments/run_classical_baseline_eval.py \
  --method keren \
  --exp-name keren_v1 \
  --split-from-run-dir "$VM_RUN" \
  --seed 42

# Elastix pairwise chain (slow; needs elastix.exe)
python src/python/experiments/run_classical_baseline_eval.py \
  --method elastix_chain \
  --exp-name elastix_chain_v1 \
  --split-from-run-dir "$VM_RUN" \
  --elastix-epochs 20 \
  --visualize

# Elastix groupwise (slow)
python src/python/experiments/run_classical_baseline_eval.py \
  --method elastix_groupwise \
  --exp-name elastix_gw_v1 \
  --split-from-run-dir "$VM_RUN" \
  --elastix-epochs 80

# All four classical methods in one go
python src/python/experiments/run_classical_baseline_eval.py \
  --method all \
  --exp-name classical_compare_v1 \
  --split-from-run-dir "$VM_RUN" \
  --seed 42
```

Classical outputs:
- `outputs/classical_baselines/{method}/{exp_name}_*/session_exports/*/bands|flows`
- Terminal prints before/after/delta; appends row to `seed_42.csv`

Debug with fewer sessions: `--max-sessions 5`

Skip band/flow export: `--no-save-outputs`

### Step 4 — Re-export VoxelMorph test viz (optional, from existing run)

```bash
python src/python/experiments/visualize_voxelmorph_test.py \
  --run-dir outputs/voxelmorph_runs/baseline/baseline_v1_YYYYMMDD_HHMMSS \
  --all-test-sessions \
  --device cuda
```

### Eval-only (skip training, re-run metrics + viz)

```bash
python src/python/experiments/train_voxelmorph.py \
  --method baseline \
  --exp-name baseline_v1 \
  --eval-only \
  --run-dir outputs/voxelmorph_runs/baseline/baseline_v1_YYYYMMDD_HHMMSS \
  --seed 42 \
  --device cuda
```

Skip CSV: add `--no-metrics-csv`

## Validation metrics

| Method | Val metric | Saved as |
|--------|------------|----------|
| baseline | pairwise `NCC_after` | `checkpoints/best.pt` |
| stack_spatial | stack chain `NCC_after_mean` | `checkpoints/best.pt` |

## Key scripts

| File | Role |
|------|------|
| `src/python/experiments/train_voxelmorph.py` | VoxelMorph CLI entry |
| `src/python/experiments/run_classical_baseline_eval.py` | Classical baseline test eval + exports |
| `src/python/experiments/session_outputs.py` | Save bands/flows per session |
| `src/python/voxelmorph/experiment.py` | Run dir + full pipeline |
| `src/python/voxelmorph/training.py` | `train_voxelmorph_baseline`, `train_voxelmorph_stack_spatial` |
| `src/python/registration/classical_stack.py` | Whole-stack Elastix/StackReg/KEREN |
| `src/python/experiments/visualize_voxelmorph_test.py` | Fake RGB + bands/flows on test set |

## Agent checklist

When user changes hyperparameters:

- [ ] Use new `--exp-name` (timestamp folder is automatic)
- [ ] Keep `--method baseline` vs `stack_spatial` explicit for fair comparison
- [ ] Use same `--seed` and `--split-from-run-dir` for classical vs DL
- [ ] After training, confirm `checkpoints/best.pt` exists
- [ ] Confirm `visualizations/*/bands/` and `flows/` exist
- [ ] Report `test_metrics.json` → `all_pairs_eval.summary_before/after`
- [ ] Confirm `outputs/metrics_tables/seed_{seed}_unregistered.json` exists
