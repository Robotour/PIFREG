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
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f'Failed to read image (empty/missing): {path}')
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Failed to read image: {path}')
    if image_size is not None:
        img = cv2.resize(img, image_size)
    return _normalize_band(img)


def _list_readable_band_files(folder):
    """List non-empty jpeg/jpg bands that OpenCV can read, sorted by wavelength."""
    candidates = sorted(
        list(folder.glob('*.jpeg')) + list(folder.glob('*.jpg')),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    readable = []
    for path in candidates:
        if path.stat().st_size <= 0:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        readable.append(path)
    return readable


def discover_band_folders(data_roots, min_bands=2):
    """Find session folders containing wavelength-named jpeg bands."""
    folders = set()
    for root in data_roots:
        root = Path(root)
        if not root.exists():
            continue
        for pattern in ('**/*.jpeg', '**/*.jpg'):
            for path in root.glob(pattern):
                if path.stat().st_size > 0:
                    folders.add(path.parent)

    valid = []
    for folder in sorted(folders):
        files = _list_readable_band_files(folder)
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
    skipped_empty = 0
    for folder in folders:
        files = _list_readable_band_files(folder)
        skipped_empty += (
            len(list(folder.glob('*.jpeg')) + list(folder.glob('*.jpg'))) - len(files)
        )
        if len(files) < 2:
            continue

        for i in range(len(files) - 1):
            moving = load_band_image(files[i], image_size=image_size)
            fixed = load_band_image(files[i + 1], image_size=image_size)
            pairs.append((moving, fixed, str(files[i]), str(files[i + 1])))

    if skipped_empty:
        print(f'Skipped {skipped_empty} empty/unreadable band images.')
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


def select_best_checkpoint(model_dir, metric='NCC_after', history_name='train_history.json'):
    """
    Prefer checkpoints/best.pt saved during training; fall back to history scan.
    """
    model_dir = Path(model_dir)
    best_info_path = model_dir / 'best_info.json'
    if best_info_path.is_file():
        info = json.loads(best_info_path.read_text(encoding='utf-8'))
        ckpt = Path(info['checkpoint'])
        if ckpt.is_file():
            info.setdefault('note', 'best.pt from training')
            return info

    for candidate in (model_dir / 'checkpoints' / 'best.pt', model_dir / 'best.pt'):
        if candidate.is_file():
            return {
                'epoch': None,
                'metric': metric,
                'metric_value': None,
                'checkpoint': str(candidate.resolve()),
                'val': None,
                'note': 'best.pt on disk',
            }

    history_path = model_dir / history_name
    if not history_path.is_file():
        final_path = model_dir / 'checkpoints' / 'final.pt'
        if not final_path.is_file():
            final_path = model_dir / 'final.pt'
        if final_path.is_file():
            return {
                'epoch': None,
                'metric': metric,
                'metric_value': None,
                'checkpoint': str(final_path),
                'val': None,
                'note': 'no train_history.json; using final.pt',
            }
        raise FileNotFoundError(f'No history or final.pt under {model_dir}')

    history = json.loads(history_path.read_text(encoding='utf-8'))
    best = None
    for row in history:
        val = row.get('val_stack') or row.get('val')
        if not val or metric not in val:
            continue
        score = float(val[metric])
        if best is None or score > best['metric_value']:
            best = {
                'epoch': int(row['epoch']),
                'metric': metric,
                'metric_value': score,
                'val': val,
            }
    if best is None:
        final_path = model_dir / 'checkpoints' / 'final.pt'
        if not final_path.is_file():
            final_path = model_dir / 'final.pt'
        if not final_path.is_file():
            raise FileNotFoundError(f'No val entries in history and no final.pt in {model_dir}')
        return {
            'epoch': None,
            'metric': metric,
            'metric_value': None,
            'checkpoint': str(final_path),
            'val': None,
            'note': 'no val metrics; using final.pt',
        }

    numbered = []
    for search_dir in (model_dir / 'checkpoints', model_dir):
        if not search_dir.is_dir():
            continue
        for path in search_dir.glob('*.pt'):
            if path.stem.isdigit():
                numbered.append((int(path.stem), path))
    numbered.sort(key=lambda x: x[0])

    if numbered:
        target = best['epoch']
        nearest_epoch, nearest_path = min(numbered, key=lambda x: abs(x[0] - target))
        best['checkpoint'] = str(nearest_path)
        best['checkpoint_epoch'] = nearest_epoch
        if nearest_epoch != target:
            best['note'] = (
                f'best val epoch={target}, nearest saved checkpoint={nearest_path.name}'
            )
    else:
        final_path = model_dir / 'checkpoints' / 'final.pt'
        if not final_path.is_file():
            final_path = model_dir / 'final.pt'
        best['checkpoint'] = str(final_path)
        best['checkpoint_epoch'] = None
        best['note'] = 'no numbered checkpoints; using final.pt'
    return best


def register_stack_with_voxelmorph_chain(
    model,
    bands,
    device='cuda',
    descending=True,
    smooth_flow_sigma=0.0,
):
    """
    用相邻波段 VoxelMorph 沿波长链配准整栈。

    训练配对约定：moving=较短波长, fixed=较长波长（升序相邻）。
    descending=True：从长波锚点向短波逐对配准（与 PIFReg chain 一致）。

    smooth_flow_sigma>0 时对每步 flow 做高斯平滑，减轻链式毛刺。

    Returns:
        registered bands (list of float32 arrays), list of per-step info
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model.eval()
    model = model.to(device)
    registered = [np.asarray(b, dtype=np.float32).copy() for b in bands]
    n = len(registered)
    if n < 2:
        return registered, []

    steps = []
    with torch.no_grad():
        pair_indices = _chain_pair_indices(n, descending=descending)

        for step, (fixed_idx, moving_idx) in enumerate(pair_indices, start=1):
            moving = registered[moving_idx]
            fixed = registered[fixed_idx]
            moving_t = torch.from_numpy(moving).float().unsqueeze(0).unsqueeze(0).to(device)
            fixed_t = torch.from_numpy(fixed).float().unsqueeze(0).unsqueeze(0).to(device)
            warped_t, flow_t = model(moving_t, fixed_t, registration=True)
            flow = flow_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if smooth_flow_sigma and smooth_flow_sigma > 0:
                flow = smooth_flow_2d(flow, sigma=float(smooth_flow_sigma))
                warped = warp_band_with_flow(moving, flow, device=device)
            else:
                warped = warped_t.squeeze().detach().cpu().numpy().astype(np.float32)
            registered[moving_idx] = warped
            steps.append({
                'step': step,
                'fixed_idx': fixed_idx,
                'moving_idx': moving_idx,
                'flow': flow,
            })
    return registered, steps


def warp_band_with_flow(band, flow, device='cpu'):
    """Apply a (2,H,W) flow to a single band (any intensity range)."""
    from .layers import SpatialTransformer

    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    band = np.asarray(band, dtype=np.float32)
    h, w = band.shape
    transformer = SpatialTransformer((h, w)).to(device)
    src = torch.from_numpy(band).float().unsqueeze(0).unsqueeze(0).to(device)
    flow_t = torch.from_numpy(np.asarray(flow, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = transformer(src, flow_t)
    return out.squeeze().detach().cpu().numpy().astype(np.float32)


def register_raw_stack_with_chain_flows(bands_raw, chain_steps, device='cpu'):
    """Apply chain flows (from register_stack_with_voxelmorph_chain) to raw-intensity bands."""
    registered = [np.asarray(b, dtype=np.float32).copy() for b in bands_raw]
    for step in chain_steps:
        moving_idx = step['moving_idx']
        registered[moving_idx] = warp_band_with_flow(
            registered[moving_idx], step['flow'], device=device,
        )
    return registered


def smooth_flow_2d(flow, sigma=1.5):
    """Gaussian-smooth a (2,H,W) displacement field to reduce chain artifacts."""
    from scipy.ndimage import gaussian_filter

    flow = np.asarray(flow, dtype=np.float32)
    out = np.zeros_like(flow)
    for c in range(flow.shape[0]):
        out[c] = gaussian_filter(flow[c], sigma=sigma)
    return out


def compute_spatial_weight_map(
    fixed: np.ndarray,
    center_floor: float = 0.35,
    edge_gain: float = 1.0,
) -> np.ndarray:
    """
    Tongue-oriented spatial weights: higher on edges / high-gradient regions,
    lower in the relatively stable tongue body center.

    Returns float32 (H,W) with spatial mean ~= 1.
    """
    img = np.asarray(fixed, dtype=np.float32)
    if img.max() > 1.0:
        img = (img - img.min()) / max(float(img.max() - img.min()), 1e-8)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = grad / (float(grad.max()) + 1e-8)

    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    dist = dist / (float(dist.max()) + 1e-8)

    edge_score = 0.65 * grad + 0.35 * dist
    edge_score = edge_score / (float(edge_score.max()) + 1e-8)
    weight = center_floor + edge_gain * (1.0 - center_floor) * edge_score
    weight = weight.astype(np.float32)
    weight = weight / (float(weight.mean()) + 1e-8)
    return weight


def load_session_bands(folder, image_size=None):
    """Load normalized bands for one session, sorted by wavelength."""
    files = _list_readable_band_files(Path(folder))
    bands = [load_band_image(p, image_size=image_size) for p in files]
    return bands, [str(p) for p in files]


def stack_subchain_generator(folders, subchain_len=6, image_size=None):
    """
    Yield contiguous sub-chains from random sessions for stack-aware training.

    Each sample: list of normalized bands length K (K=subchain_len).
    """
    folders = list(folders)
    while True:
        folder = folders[np.random.randint(len(folders))]
        bands, _ = load_session_bands(folder, image_size=image_size)
        if len(bands) < 2:
            continue
        k = min(subchain_len, len(bands))
        if k < 2:
            continue
        start = np.random.randint(0, len(bands) - k + 1)
        yield bands[start : start + k]


def _chain_pair_indices(n, descending=True):
    if descending:
        return [(i + 1, i) for i in range(n - 2, -1, -1)]
    return [(i - 1, i) for i in range(1, n)]


def evaluate_voxelmorph_sessions(
    model,
    folders,
    device='cuda',
    image_size=None,
    descending=True,
    smooth_flow_sigma=0.0,
    max_sessions=None,
    verbose=True,
):
    """Session-level chain registration metrics (whole 30-band stack)."""
    from src.python.metrics.evaluation import compute_NCC

    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

    eval_folders = list(folders)
    if max_sessions is not None and max_sessions < len(eval_folders):
        eval_folders = list(np.random.default_rng(0).choice(eval_folders, size=max_sessions, replace=False))

    rows = []
    for idx, folder in enumerate(eval_folders):
        bands, _ = load_session_bands(folder, image_size=image_size)
        registered, _ = register_stack_with_voxelmorph_chain(
            model,
            bands,
            device=device,
            descending=descending,
            smooth_flow_sigma=smooth_flow_sigma,
        )
        anchor = len(bands) - 1 if descending else 0
        ref = registered[anchor]
        ncc_before = []
        ncc_after = []
        for i, (b, r) in enumerate(zip(bands, registered)):
            if i == anchor:
                continue
            ncc_before.append(float(compute_NCC(ref, b)))
            ncc_after.append(float(compute_NCC(ref, r)))
        rows.append({
            'session': str(folder),
            'NCC_before_mean': float(np.mean(ncc_before)),
            'NCC_after_mean': float(np.mean(ncc_after)),
            'NCC_delta_mean': float(np.mean(ncc_after) - np.mean(ncc_before)),
        })
        if verbose:
            print(
                f'  session {idx + 1}/{len(eval_folders)}  '
                f'NCC {rows[-1]["NCC_before_mean"]:.4f}->{rows[-1]["NCC_after_mean"]:.4f}',
                flush=True,
            )

    summary = {
        'NCC_before_mean': float(np.mean([r['NCC_before_mean'] for r in rows])),
        'NCC_after_mean': float(np.mean([r['NCC_after_mean'] for r in rows])),
        'NCC_delta_mean': float(np.mean([r['NCC_delta_mean'] for r in rows])),
    }
    return {'summary': summary, 'per_session': rows, 'num_sessions': len(rows)}


def _apply_similarity_loss(fixed, warped, image_loss, image_loss_func, weight_t=None):
    from .losses import weighted_mse_loss, weighted_ncc_loss

    if weight_t is None:
        return image_loss_func(fixed, warped)
    if image_loss == 'mse':
        return weighted_mse_loss(fixed, warped, weight_t)
    return weighted_ncc_loss(fixed, warped, weight_t)


def _weight_tensor_from_fixed(fixed_t, center_floor, edge_gain, device):
    fixed_np = fixed_t.squeeze().detach().cpu().numpy()
    w = compute_spatial_weight_map(fixed_np, center_floor=center_floor, edge_gain=edge_gain)
    return torch.from_numpy(w).to(device).unsqueeze(0).unsqueeze(0)


def _train_step_stack_chain(
    model,
    bands,
    device,
    image_loss,
    image_loss_func,
    grad_loss_func,
    lamda,
    descending=True,
    spatial_weights=False,
    center_floor=0.35,
    edge_gain=1.0,
):
    """One stack sub-chain forward: accumulate chain losses (differentiable)."""
    registered = [
        torch.from_numpy(np.asarray(b, dtype=np.float32)).float().unsqueeze(0).unsqueeze(0).to(device)
        for b in bands
    ]
    pair_indices = _chain_pair_indices(len(bands), descending=descending)
    total_loss = 0.0
    for fixed_idx, moving_idx in pair_indices:
        moving = registered[moving_idx]
        fixed = registered[fixed_idx]
        warped, preint_flow = model(moving, fixed)
        weight_t = (
            _weight_tensor_from_fixed(fixed, center_floor, edge_gain, device)
            if spatial_weights else None
        )
        sim = _apply_similarity_loss(fixed, warped, image_loss, image_loss_func, weight_t)
        total_loss = total_loss + sim + lamda * grad_loss_func(None, preint_flow)
        registered[moving_idx] = warped
    return total_loss / max(len(pair_indices), 1)


def _validation_score(record, method='baseline'):
    if method == 'stack_spatial' or record.get('val_stack'):
        return float(record['val_stack']['NCC_after_mean'])
    return float(record['val']['NCC_after'])


def _train_voxelmorph_core(
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
    training_mode='pair',
    train_folders=None,
    val_folders=None,
    subchain_len=6,
    spatial_weights=False,
    center_floor=0.35,
    edge_gain=1.0,
    chain_descending=True,
    val_session_steps=10,
    method='baseline',
):
    """
    Internal trainer. Saves checkpoints/best.pt (best val) and checkpoints/final.pt.
    """
    from .config import default_unet_features
    from .losses import Grad, MSE, NCC
    from .networks import VxmDense

    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model_dir = Path(model_dir)
    ckpt_dir = model_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
    pair_generator = scan_to_scan_generator(pairs, batch_size=1) if pairs else None
    stack_generator = (
        stack_subchain_generator(train_folders, subchain_len=subchain_len, image_size=inshape[::-1])
        if train_folders else None
    )
    history = []
    best_score = float('-inf')
    best_epoch = None
    best_path = ckpt_dir / 'best.pt'

    for epoch in range(epochs):
        epoch_loss = []
        for _ in range(steps_per_epoch):
            if training_mode == 'stack':
                if stack_generator is None:
                    raise ValueError('train_folders required for stack training_mode')
                subchain = next(stack_generator)
                loss = _train_step_stack_chain(
                    model,
                    subchain,
                    device,
                    image_loss,
                    image_loss_func,
                    grad_loss_func,
                    lamda,
                    descending=chain_descending,
                    spatial_weights=spatial_weights,
                    center_floor=center_floor,
                    edge_gain=edge_gain,
                )
            else:
                moving_np, fixed_np = next(pair_generator)
                moving = torch.from_numpy(moving_np).to(device).float()
                fixed = torch.from_numpy(fixed_np).to(device).float()
                warped, preint_flow = model(moving, fixed)
                weight_t = (
                    _weight_tensor_from_fixed(fixed, center_floor, edge_gain, device)
                    if spatial_weights else None
                )
                sim = _apply_similarity_loss(fixed, warped, image_loss, image_loss_func, weight_t)
                loss = sim + lamda * grad_loss_func(None, preint_flow)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_loss))
        record = {
            'epoch': epoch + 1,
            'train_loss': mean_loss,
            'training_mode': training_mode,
            'method': method,
        }
        did_validate = False

        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            if training_mode == 'stack' and val_folders:
                val_result = evaluate_voxelmorph_sessions(
                    model,
                    val_folders,
                    device=device,
                    image_size=inshape[::-1],
                    descending=chain_descending,
                    max_sessions=val_session_steps,
                    verbose=False,
                )
                record['val_stack'] = val_result['summary']
                did_validate = True
                print(
                    f'Epoch {epoch + 1}/{epochs}  train_loss={mean_loss:.4e}  '
                    f'val_stack_NCC {record["val_stack"]["NCC_before_mean"]:.4f}'
                    f'->{record["val_stack"]["NCC_after_mean"]:.4f}',
                    flush=True,
                )
            elif val_pairs:
                val_result = evaluate_voxelmorph_pairs(
                    model, val_pairs, device=device, max_pairs=val_steps, verbose=False,
                )
                record['val'] = val_result['summary']
                did_validate = True
                print(
                    f'Epoch {epoch + 1}/{epochs}  train_loss={mean_loss:.4e}  '
                    f'val_NCC {record["val"]["NCC_before"]:.4f}->{record["val"]["NCC_after"]:.4f}',
                    flush=True,
                )
            elif epoch % 20 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch + 1}/{epochs}  train_loss={mean_loss:.4e}', flush=True)
        elif epoch % 20 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch + 1}/{epochs}  train_loss={mean_loss:.4e}', flush=True)

        if did_validate:
            score = _validation_score(record, method=method)
            record['val_score'] = score
            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                model.save(str(best_path))
                record['is_best'] = True
            else:
                record['is_best'] = False

        history.append(record)

        if epoch % 20 == 0 or epoch == epochs - 1:
            model.save(str(ckpt_dir / f'{epoch + 1:04d}.pt'))

    final_path = ckpt_dir / 'final.pt'
    model.save(str(final_path))

    if not best_path.is_file():
        model.save(str(best_path))
        best_epoch = epochs
        best_score = None

    best_info = {
        'method': method,
        'best_epoch': best_epoch,
        'best_score': best_score,
        'metric': 'NCC_after_mean' if method == 'stack_spatial' else 'NCC_after',
        'checkpoint': str(best_path.resolve()),
        'final_checkpoint': str(final_path.resolve()),
    }
    with open(model_dir / 'best_info.json', 'w', encoding='utf-8') as f:
        json.dump(best_info, f, indent=2)

    history_path = model_dir / 'train_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    return model, str(best_path), best_info


def train_voxelmorph_baseline(
    pairs,
    model_dir,
    inshape,
    val_pairs=None,
    device='cuda',
    epochs=1500,
    steps_per_epoch=100,
    lr=1e-4,
    image_loss='ncc',
    lamda=0.01,
    int_steps=7,
    int_downsize=2,
    load_model=None,
    val_steps=20,
    val_interval=20,
    **kwargs,
):
    """
    Baseline VoxelMorph: random adjacent-band pairs, uniform spatial loss.
    Validation metric: pairwise NCC_after on held-out pairs.
    """
    return _train_voxelmorph_core(
        pairs=pairs,
        model_dir=model_dir,
        inshape=inshape,
        device=device,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
        image_loss=image_loss,
        lamda=lamda,
        int_steps=int_steps,
        int_downsize=int_downsize,
        load_model=load_model,
        val_pairs=val_pairs,
        val_steps=val_steps,
        val_interval=val_interval,
        training_mode='pair',
        spatial_weights=False,
        method='baseline',
        **kwargs,
    )


def train_voxelmorph_stack_spatial(
    pairs,
    model_dir,
    inshape,
    train_folders,
    val_folders,
    device='cuda',
    epochs=1500,
    steps_per_epoch=100,
    lr=1e-4,
    image_loss='mse',
    lamda=0.01,
    int_steps=7,
    int_downsize=2,
    load_model=None,
    val_interval=20,
    subchain_len=6,
    center_floor=0.35,
    edge_gain=1.0,
    chain_descending=True,
    val_session_steps=15,
    **kwargs,
):
    """
    Proposed method: session sub-chain training + edge-high / center-low spatial weights.
    Validation metric: whole-stack chain NCC_after_mean on held-out sessions.
    """
    return _train_voxelmorph_core(
        pairs=pairs,
        model_dir=model_dir,
        inshape=inshape,
        device=device,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
        image_loss=image_loss,
        lamda=lamda,
        int_steps=int_steps,
        int_downsize=int_downsize,
        load_model=load_model,
        val_pairs=None,
        val_interval=val_interval,
        training_mode='stack',
        train_folders=train_folders,
        val_folders=val_folders,
        subchain_len=subchain_len,
        spatial_weights=True,
        center_floor=center_floor,
        edge_gain=edge_gain,
        chain_descending=chain_descending,
        val_session_steps=val_session_steps,
        method='stack_spatial',
        **kwargs,
    )


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
    training_mode='pair',
    train_folders=None,
    val_folders=None,
    subchain_len=6,
    spatial_weights=False,
    center_floor=0.35,
    edge_gain=1.0,
    chain_descending=True,
    val_session_steps=10,
):
    """Backward-compatible wrapper. Prefer train_voxelmorph_baseline / train_voxelmorph_stack_spatial."""
    method = 'stack_spatial' if training_mode == 'stack' and spatial_weights else 'baseline'
    model, best_path, _ = _train_voxelmorph_core(
        pairs=pairs,
        model_dir=model_dir,
        inshape=inshape,
        device=device,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
        image_loss=image_loss,
        lamda=lamda,
        int_steps=int_steps,
        int_downsize=int_downsize,
        load_model=load_model,
        enc_nf=enc_nf,
        dec_nf=dec_nf,
        val_pairs=val_pairs,
        val_steps=val_steps,
        val_interval=val_interval,
        training_mode=training_mode,
        train_folders=train_folders,
        val_folders=val_folders,
        subchain_len=subchain_len,
        spatial_weights=spatial_weights,
        center_floor=center_floor,
        edge_gain=edge_gain,
        chain_descending=chain_descending,
        val_session_steps=val_session_steps,
        method=method,
    )
    return model, best_path
