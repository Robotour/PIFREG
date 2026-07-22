"""HSI band-pair data generator and VoxelMorph pre-training utilities."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch


def _normalize_band(img):
    img = img.astype(np.float32)
    vmin, vmax = np.min(img), np.max(img)
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    return img


def load_band_image(path, image_size=None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Failed to read image: {path}')
    if image_size is not None:
        img = cv2.resize(img, image_size)
    return _normalize_band(img)


def discover_band_folders(data_roots, min_bands=2):
    """Find session folders containing wavelength-named jpeg bands."""
    folders = set()
    for root in data_roots:
        root = Path(root)
        if not root.exists():
            continue
        for pattern in ('**/*.jpeg', '**/*.jpg'):
            for path in root.glob(pattern):
                folders.add(path.parent)

    valid = []
    for folder in sorted(folders):
        files = list(folder.glob('*.jpeg')) + list(folder.glob('*.jpg'))
        if len(files) >= min_bands:
            valid.append(folder)
    return valid


def split_folders_train_test(folders, train_ratio=0.7, seed=42):
    """Split session folders; keep all pairs from one session in the same split."""
    folders = list(folders)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(folders))
    rng.shuffle(indices)
    n_train = max(1, int(round(len(folders) * train_ratio)))
    if n_train >= len(folders):
        n_train = max(1, len(folders) - 1)
    train_idx = set(indices[:n_train].tolist())
    train_folders = [folders[i] for i in range(len(folders)) if i in train_idx]
    test_folders = [folders[i] for i in range(len(folders)) if i not in train_idx]
    return train_folders, test_folders


def build_adjacent_band_pairs(folders, image_size=None):
    """Build (moving, fixed) pairs from adjacent bands in each folder."""
    pairs = []
    for folder in folders:
        files = sorted(
            list(folder.glob('*.jpeg')) + list(folder.glob('*.jpg')),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )
        if len(files) < 2:
            continue

        for i in range(len(files) - 1):
            moving = load_band_image(files[i], image_size=image_size)
            fixed = load_band_image(files[i + 1], image_size=image_size)
            pairs.append((moving, fixed, str(files[i]), str(files[i + 1])))

    if not pairs:
        raise ValueError('No adjacent band pairs found in provided folders.')
    return pairs


def save_split_manifest(path, train_folders, test_folders, train_ratio, seed):
    payload = {
        'train_ratio': train_ratio,
        'seed': seed,
        'num_train_sessions': len(train_folders),
        'num_test_sessions': len(test_folders),
        'train_sessions': [str(p) for p in train_folders],
        'test_sessions': [str(p) for p in test_folders],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def scan_to_scan_generator(pairs, batch_size=1):
    """Yield random adjacent-band (moving, fixed) batches."""
    while True:
        indices = np.random.randint(len(pairs), size=batch_size)
        moving_batch = []
        fixed_batch = []
        for idx in indices:
            moving, fixed, _, _ = pairs[idx]
            moving_batch.append(moving)
            fixed_batch.append(fixed)

        moving_arr = np.stack(moving_batch, axis=0)[:, np.newaxis, ...]
        fixed_arr = np.stack(fixed_batch, axis=0)[:, np.newaxis, ...]
        yield moving_arr, fixed_arr


def _pair_metrics(fixed, moving, warped):
    from src.python.metrics.evaluation import compute_MI, compute_NCC, compute_NMI, compute_NTG

    return {
        'MSE_before': float(np.mean((fixed - moving) ** 2)),
        'MSE_after': float(np.mean((fixed - warped) ** 2)),
        'MI_before': float(compute_MI(fixed, moving)),
        'MI_after': float(compute_MI(fixed, warped)),
        'NMI_before': float(compute_NMI(fixed, moving)),
        'NMI_after': float(compute_NMI(fixed, warped)),
        'NCC_before': float(compute_NCC(fixed, moving)),
        'NCC_after': float(compute_NCC(fixed, warped)),
        'NTG_before': float(compute_NTG(fixed, moving)),
        'NTG_after': float(compute_NTG(fixed, warped)),
    }


def evaluate_voxelmorph_pairs(
    model,
    pairs,
    device='cuda',
    max_pairs=None,
    verbose=True,
):
    """Run inference on band pairs and aggregate before/after metrics."""
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model.eval()
    model = model.to(device)
    if max_pairs is not None and max_pairs < len(pairs):
        indices = np.random.default_rng(0).choice(len(pairs), size=max_pairs, replace=False)
        eval_pairs = [pairs[i] for i in indices]
    else:
        eval_pairs = pairs

    rows = []
    with torch.no_grad():
        for idx, (moving, fixed, moving_path, fixed_path) in enumerate(eval_pairs):
            moving_t = torch.from_numpy(moving).float().unsqueeze(0).unsqueeze(0).to(device)
            fixed_t = torch.from_numpy(fixed).float().unsqueeze(0).unsqueeze(0).to(device)
            warped_t, _ = model(moving_t, fixed_t, registration=True)
            warped = warped_t.squeeze().detach().cpu().numpy().astype(np.float32)
            row = _pair_metrics(fixed, moving, warped)
            row['moving_path'] = moving_path
            row['fixed_path'] = fixed_path
            rows.append(row)
            if verbose and (idx + 1) % max(len(eval_pairs) // 10, 1) == 0:
                print(f'  evaluated {idx + 1}/{len(eval_pairs)} pairs', flush=True)

    summary = _summarize_metric_rows(rows)
    return {'summary': summary, 'per_pair': rows, 'num_pairs': len(rows)}


def _summarize_metric_rows(rows):
    if not rows:
        return {}
    keys = [k for k in rows[0] if k.endswith('_before') or k.endswith('_after')]
    summary = {}
    for key in keys:
        vals = [r[key] for r in rows]
        summary[key] = float(np.mean(vals))
    for metric in ('MSE', 'MI', 'NMI', 'NCC', 'NTG'):
        b, a = f'{metric}_before', f'{metric}_after'
        if b in summary and a in summary:
            summary[f'{metric}_delta'] = summary[a] - summary[b]
    return summary


def train_voxelmorph(
    pairs,
    model_dir,
    inshape,
    device='cuda',
    epochs=1500,
    steps_per_epoch=100,
    lr=1e-4,
    image_loss='ncc',
    lamda=0.01,
    int_steps=7,
    int_downsize=2,
    load_model=None,
    enc_nf=None,
    dec_nf=None,
    val_pairs=None,
    val_steps=20,
    val_interval=20,
):
    """
    Pre-train official VxmDense on HSI adjacent-band pairs (P3 workflow).

    Returns the trained model and final checkpoint path.
    """
    from .config import default_unet_features
    from .losses import Grad, MSE, NCC
    from .networks import VxmDense

    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    enc_nf = enc_nf or default_unet_features()[0]
    dec_nf = dec_nf or default_unet_features()[1]

    if load_model:
        model = VxmDense.load(load_model, device)
    else:
        model = VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            int_steps=int_steps,
            int_downsize=int_downsize,
        )

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if image_loss == 'ncc':
        image_loss_func = NCC().loss
    elif image_loss == 'mse':
        image_loss_func = MSE().loss
    else:
        raise ValueError(f'Unsupported image_loss: {image_loss}')

    grad_loss_func = Grad('l2', loss_mult=int_downsize).loss
    generator = scan_to_scan_generator(pairs, batch_size=1)
    history = []

    for epoch in range(epochs):
        epoch_loss = []
        for _ in range(steps_per_epoch):
            moving_np, fixed_np = next(generator)
            moving = torch.from_numpy(moving_np).to(device).float()
            fixed = torch.from_numpy(fixed_np).to(device).float()

            warped, preint_flow = model(moving, fixed)
            loss = image_loss_func(fixed, warped) + lamda * grad_loss_func(None, preint_flow)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        mean_loss = float(np.mean(epoch_loss))
        record = {'epoch': epoch + 1, 'train_loss': mean_loss}

        if val_pairs and ((epoch + 1) % val_interval == 0 or epoch == epochs - 1):
            val_result = evaluate_voxelmorph_pairs(
                model, val_pairs, device=device, max_pairs=val_steps, verbose=False,
            )
            record['val'] = val_result['summary']
            print(
                f'Epoch {epoch + 1}/{epochs}  train_loss={mean_loss:.4e}  '
                f'val_NCC {record["val"]["NCC_before"]:.4f}->{record["val"]["NCC_after"]:.4f}',
                flush=True,
            )
        elif epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f'Epoch {epoch + 1}/{epochs}  train_loss={mean_loss:.4e}',
                flush=True,
            )

        history.append(record)

        if epoch % 20 == 0 or epoch == epochs - 1:
            ckpt = model_dir / f'{epoch + 1:04d}.pt'
            model.save(str(ckpt))

    final_path = model_dir / 'final.pt'
    model.save(str(final_path))

    history_path = model_dir / 'train_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    return model, str(final_path)
