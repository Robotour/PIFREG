# PIFReg: Pyramid Instance Flow Registration
# 金字塔实例流配准 —— 逐对无监督优化 + 多尺度位移场传递（非 VoxelMorph 论文流程）

import copy

import cv2
import numpy as np
import torch
from pystackreg import StackReg
from skimage.exposure import match_histograms

from ..losses.registration_losses import NCC, Grad
from ..voxelmorph import VxmDense
from ..voxelmorph.config import compact_unet_features, default_unet_features
from ..voxelmorph.layers import SpatialTransformer
from ..voxelmorph.losses import MSE

METHOD_NAME = 'PIFReg'
METHOD_FULL_NAME = 'Pyramid Instance Flow Registration'


def _resolve_device(device):
    if isinstance(device, str):
        return torch.device(device if torch.cuda.is_available() else 'cpu')
    return device


def _downsample_image(img, width, height):
    """下采样仅用于该尺度的配准优化，使用 INTER_AREA 减少混叠。"""
    if img.shape[0] == height and img.shape[1] == width:
        return img.astype(np.float32)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)


def _upsample_flow(flow, target_h, target_w):
    """将位移场上采样到目标分辨率，并按像素比例缩放位移分量。"""
    _, _, h, w = flow.shape
    if h == target_h and w == target_w:
        return flow

    scale_y = target_h / h
    scale_x = target_w / w
    flow_scaled = flow.clone()
    flow_scaled[:, 0, ...] *= scale_y
    flow_scaled[:, 1, ...] *= scale_x
    return torch.nn.functional.interpolate(
        flow_scaled, size=(target_h, target_w), mode='bilinear', align_corners=True
    )


def _compose_flows(base_flow, delta_flow, shape_hw, device):
    """组合位移场：先施加 base，再叠加经 base warp 后的 delta（残差组合）。"""
    transformer = SpatialTransformer(shape_hw).to(device)
    warped_delta = transformer(delta_flow, base_flow)
    return base_flow + warped_delta


def _apply_flow_to_image(moving_image, flow, device):
    """用位移场对原分辨率图像做一次空间变换。"""
    h, w = moving_image.shape
    m = torch.tensor(moving_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    transformer = SpatialTransformer((h, w)).to(device)
    with torch.no_grad():
        warped = transformer(m, flow)
    return warped.squeeze().cpu().numpy().astype(np.float32)


def _build_flow_model(
    inshape, device, model_path=None, int_steps=7, int_downsize=2, nb_unet_features=None,
):
    """创建或加载 U-Net 位移场预测模型（VxmDense 仅作网络 backbone）。"""
    if model_path:
        model = VxmDense.load(model_path, device)
    else:
        enc_nf, dec_nf = nb_unet_features or default_unet_features()
        model = VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            int_steps=int_steps,
            int_downsize=int_downsize,
        )
    return model.to(device)


def _create_lr_scheduler(optimizer, schedule, max_epochs, lr_gamma=0.5, lr_min=1e-6):
    schedule = (schedule or 'none').lower()
    if schedule == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(max_epochs // 3, 1), gamma=lr_gamma
        ), 'step'
    if schedule == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr_min
        ), 'cosine'
    if schedule == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_gamma, patience=max(max_epochs // 10, 20),
            min_lr=lr_min, verbose=False
        ), 'plateau'
    return None, 'none'


def _preprocess_pair(fixed_image, moving_image, histogram_match=False, affine_init=False):
    """单对配准前的预处理。"""
    fixed = fixed_image.astype(np.float32)
    moving = moving_image.astype(np.float32)

    if histogram_match:
        try:
            moving = match_histograms(moving, fixed, channel_axis=None).astype(np.float32)
        except TypeError:
            moving = match_histograms(moving, fixed, multichannel=False).astype(np.float32)

    if affine_init:
        sr = StackReg(StackReg.BILINEAR)
        sr.register(fixed, moving)
        moving = sr.transform(moving).astype(np.float32)

    return fixed, moving


def _train_at_scale(
    fixed_image,
    moving_image,
    device,
    max_epochs,
    lr,
    lamda,
    int_steps,
    int_downsize,
    image_loss,
    model_path=None,
    nb_unet_features=None,
    early_stop=True,
    patience=100,
    min_delta=1e-5,
    lr_schedule='cosine',
    lr_gamma=0.5,
    lr_min=1e-6,
    verbose=True,
):
    """在固定尺度上对单对图像做无监督优化，支持早停与学习率调度。"""
    inshape = fixed_image.shape
    f = torch.tensor(fixed_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    m = torch.tensor(moving_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    model = _build_flow_model(
        inshape, device, model_path=model_path,
        int_steps=int_steps, int_downsize=int_downsize,
        nb_unet_features=nb_unet_features,
    )

    if image_loss == 'ncc':
        image_loss_fn = NCC().loss
    elif image_loss == 'mse':
        image_loss_fn = MSE().loss
    else:
        raise ValueError(f'Unsupported image_loss: {image_loss}')

    grad_loss_fn = Grad('l2', loss_mult=int_downsize).loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler, schedule_type = _create_lr_scheduler(
        optimizer, lr_schedule, max_epochs, lr_gamma=lr_gamma, lr_min=lr_min
    )

    best_loss = float('inf')
    best_flow = None
    best_state = None
    stale_epochs = 0
    log_every = max(max_epochs // 15, 1)

    for epoch in range(max_epochs):
        model.train()
        warped, preint_flow = model(m, f)
        loss = image_loss_fn(f, warped) + lamda * grad_loss_fn(None, preint_flow)
        current_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if schedule_type == 'plateau':
            scheduler.step(current_loss)
        elif scheduler is not None:
            scheduler.step()

        improved = current_loss < best_loss - min_delta
        if improved:
            best_loss = current_loss
            stale_epochs = 0
            model.eval()
            with torch.no_grad():
                _, best_flow = model(m, f, registration=True)
                best_state = copy.deepcopy(model.state_dict())
            model.train()
        else:
            stale_epochs += 1

        if verbose and (epoch % log_every == 0 or epoch == max_epochs - 1):
            cur_lr = optimizer.param_groups[0]['lr']
            print(
                f'  epoch {epoch + 1}/{max_epochs}: loss={current_loss:.4f} '
                f'best={best_loss:.4f} lr={cur_lr:.2e}'
            )

        if early_stop and stale_epochs >= patience:
            if verbose:
                print(
                    f'  early stop at epoch {epoch + 1} '
                    f'(patience={patience}, best_loss={best_loss:.4f})'
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        flow = best_flow
    else:
        model.eval()
        with torch.no_grad():
            _, flow = model(m, f, registration=True)

    return model, flow


def _register_multiscale(
    fixed_original,
    moving_original,
    device,
    max_epochs_per_scale,
    lr,
    lamda,
    int_steps,
    int_downsize,
    image_loss,
    scales,
    early_stop,
    patience,
    min_delta,
    lr_schedule,
    lr_gamma,
    lr_min,
    save_model_path=None,
    nb_unet_features=None,
):
    """多尺度配准：各尺度下采样优化，位移场上采样组合，最终对原图 warp 一次。"""
    h, w = fixed_original.shape
    active_scales = [s for s in scales if int(h * s) >= 32 and int(w * s) >= 32]
    if not active_scales:
        active_scales = [1.0]
    if active_scales[-1] != 1.0:
        active_scales.append(1.0)

    print(
        f'{METHOD_NAME}: multiscale {active_scales}, '
        f'up to {max_epochs_per_scale} epochs/scale, early_stop={early_stop}'
    )

    flow_at_scale = None
    model = None

    for scale in active_scales:
        sh, sw = int(h * scale), int(w * scale)
        f_s = _downsample_image(fixed_original, sw, sh)
        m_s = _downsample_image(moving_original, sw, sh)

        if flow_at_scale is not None:
            flow_on_scale = _upsample_flow(flow_at_scale, sh, sw)
            m_s = _apply_flow_to_image(m_s, flow_on_scale, device)

        print(f'Scale {scale:.2f} ({sh}x{sw})')
        model, flow_delta = _train_at_scale(
            f_s, m_s, device, max_epochs_per_scale, lr, lamda,
            int_steps, int_downsize, image_loss,
            nb_unet_features=nb_unet_features,
            early_stop=early_stop, patience=patience, min_delta=min_delta,
            lr_schedule=lr_schedule, lr_gamma=lr_gamma, lr_min=lr_min,
            verbose=True,
        )

        if flow_at_scale is None:
            flow_at_scale = flow_delta
        else:
            flow_prev = _upsample_flow(flow_at_scale, sh, sw)
            flow_at_scale = _compose_flows(flow_prev, flow_delta, (sh, sw), device)

    flow_full = _upsample_flow(flow_at_scale, h, w)
    warped = _apply_flow_to_image(moving_original, flow_full, device)

    if save_model_path and model is not None:
        model.save(save_model_path)
    return warped


def register_pifreg(
    fixed_image,
    moving_image,
    lr=1e-4,
    epochs=3000,
    device='cuda',
    lamda=0.01,
    model_path=None,
    int_steps=7,
    int_downsize=2,
    nb_unet_features=None,
    image_loss='ncc',
    save_model_path=None,
    affine_init=True,
    histogram_match=True,
    multiscale=True,
    scales=(0.25, 0.5, 1.0),
    early_stop=True,
    patience=100,
    min_delta=1e-5,
    lr_schedule='cosine',
    lr_gamma=0.5,
    lr_min=1e-6,
    fast_mode=False,
):
    """
    PIFReg（Pyramid Instance Flow Registration）金字塔实例流配准。

    逐对无监督优化：StackReg 仿射初始化 + 直方图匹配 + 多尺度位移场金字塔；
    与 VoxelMorph 论文（大数据预训练 + 单次前向）不同，本方法为 test-time optimization。

    参数:
        fixed_image / moving_image: (H, W) float 图像，建议 [0, 1]
        lr: Adam 初始学习率
        epochs: 单尺度最大 epoch；多尺度时为**每个金字塔层级**的最大 epoch
        lamda: 位移场平滑正则权重
        nb_unet_features: U-Net 通道配置 [enc_nf, dec_nf]；None 用默认 [16,32,...]
        fast_mode: 加速模式 — 轻量 U-Net、更高 lr、更低 lamda、更少积分步数、
                   两尺度金字塔 (0.5, 1.0)。与 fast_mode 同传的 lr/lamda/int_steps 等
                   仍会被 fast_mode 覆盖；需细调时请设 fast_mode=False 并手动传参
        early_stop: 是否启用早停（推荐 True，可设大 epochs 让模型充分收敛）
        patience: 早停耐心值（连续多少 epoch 无改善则停止该尺度）
        min_delta: 判定 loss 改善的最小幅度
        lr_schedule: 学习率策略 'cosine' | 'step' | 'plateau' | 'none'
        lr_gamma: StepLR / Plateau 的衰减因子
        lr_min: 学习率下限
        affine_init / histogram_match / multiscale / scales: 预处理与金字塔配置

    返回:
        warped_image: 配准后的移动图像（原分辨率，仅一次 warp）
    """
    if fast_mode:
        nb_unet_features = nb_unet_features or compact_unet_features()
        int_steps = 3
        int_downsize = 2
        lr = 2e-4
        lamda = 0.005
        scales = (0.25, 0.5, 1.0)
        patience = 80
        print(
            f'{METHOD_NAME}: fast_mode — compact U-Net, int_steps=3, '
            f'lr={lr}, lamda={lamda}, scales={scales}'
        )

    device = _resolve_device(device)
    h, w = fixed_image.shape

    fixed, moving = _preprocess_pair(
        fixed_image, moving_image,
        histogram_match=histogram_match,
        affine_init=affine_init,
    )
    moving_original = moving.copy()
    fixed_original = fixed.copy()

    train_kwargs = dict(
        early_stop=early_stop,
        patience=patience,
        min_delta=min_delta,
        lr_schedule=lr_schedule,
        lr_gamma=lr_gamma,
        lr_min=lr_min,
    )

    if multiscale and model_path is None:
        return _register_multiscale(
            fixed_original, moving_original, device, epochs, lr, lamda,
            int_steps, int_downsize, image_loss, scales, save_model_path=save_model_path,
            nb_unet_features=nb_unet_features,
            **train_kwargs,
        )

    print(f'{METHOD_NAME}: single scale {h}x{w}, up to {epochs} epochs')
    model, flow = _train_at_scale(
        fixed, moving, device, epochs, lr, lamda,
        int_steps, int_downsize, image_loss,
        model_path=model_path, nb_unet_features=nb_unet_features,
        verbose=True, **train_kwargs,
    )
    warped = _apply_flow_to_image(moving_original, flow, device)
    if save_model_path:
        model.save(save_model_path)
    return warped
