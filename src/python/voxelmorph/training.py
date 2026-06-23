"""HSI band-pair data generator and VoxelMorph pre-training utilities."""

from __future__ import annotations

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


def discover_band_folders(data_roots):
    """Find folders containing wavelength-named jpeg bands."""
    folders = set()
    for root in data_roots:
        root = Path(root)
        if not root.exists():
            continue
        for pattern in ('**/*.jpeg', '**/*.jpg'):
            for path in root.glob(pattern):
                folders.add(path.parent)
    return sorted(folders)


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

        if epoch % 20 == 0 or epoch == epochs - 1:
            ckpt = model_dir / f'{epoch:04d}.pt'
            model.save(str(ckpt))
            print(
                f'Epoch {epoch + 1}/{epochs}  '
                f'loss={np.mean(epoch_loss):.4e}  saved={ckpt.name}',
                flush=True,
            )

    final_path = model_dir / f'{epochs:04d}.pt'
    model.save(str(final_path))
    return model, str(final_path)
