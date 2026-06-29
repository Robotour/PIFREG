# PIFReg StackFlow Groupwise — 一次性预测 N-1 个 per-band 位移场 + 粗→细金字塔
#
# 在 pif_groupwise_joint.py（共享单位移场）基础上扩展：
#   - 锚点波段（600 nm），其余 N-1 个波段各有一个 2D 位移场
#   - 形状 (N-1, 2, H, W)，例如 30 波段 → 29 个位移场 @ 512×512
#   - 金字塔 128→256→512，逐层上采样/组合位移场
#   - 损失 = 相邻波段 NCC 均值（29 对 warped[i-1] vs warped[i]）+ 平滑正则

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from ..losses.registration_losses import Grad, NCC
from ..voxelmorph.config import compact_unet_features
from ..voxelmorph.layers import ResizeTransform, SpatialTransformer, VecInt
from ..voxelmorph.networks import Unet

METHOD_NAME = 'PIFReg-StackFlow'
METHOD_FULL_NAME = 'PIFReg Per-Band Stack Flow Groupwise Registration'

StackInput = Union[Sequence[np.ndarray], np.ndarray]

DEFAULT_PYRAMID_SIZES = (128, 256, 512)
DEFAULT_EPOCHS_PER_LEVEL = (900, 1500, 2500)
DEFAULT_PATIENCE_PER_LEVEL = (100, 120, 150)

FEATURE_MODE_MEAN_ANCHOR = 'mean_anchor'
FEATURE_MODE_SPECTRAL_ENCODER = 'spectral_encoder'
DEFAULT_SPECTRAL_ENC_CHANNELS = 4
DEFAULT_SPECTRAL_ENC_KERNEL = 3


def _resolve_device(device):
    if isinstance(device, str):
        return torch.device(device if torch.cuda.is_available() else 'cpu')
    return device


def _as_band_list(stack: StackInput) -> List[np.ndarray]:
    if isinstance(stack, list):
        return [np.asarray(b, dtype=np.float32) for b in stack]
    arr = np.asarray(stack)
    if arr.ndim == 2:
        return [arr.astype(np.float32)]
    if arr.ndim == 3:
        return [arr[i].astype(np.float32) for i in range(arr.shape[0])]
    raise ValueError(f'Expected stack shape (N,H,W) or list of (H,W), got {arr.shape}')


def _bands_to_tensor(bands: List[np.ndarray], device) -> torch.Tensor:
    stack = np.stack(bands, axis=0).astype(np.float32)
    return torch.tensor(stack, dtype=torch.float32, device=device).unsqueeze(0)


def _tensor_to_bands(stack_t: torch.Tensor) -> List[np.ndarray]:
    stack = stack_t.squeeze(0).detach().cpu().numpy()
    return [stack[i].astype(np.float32) for i in range(stack.shape[0])]


def _downsample_band(img, width, height):
    if img.shape[0] == height and img.shape[1] == width:
        return img.astype(np.float32)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)


def _downsample_stack(bands, width, height):
    return [_downsample_band(b, width, height) for b in bands]


def _build_pyramid_levels(h: int, w: int, sizes=DEFAULT_PYRAMID_SIZES):
    """生成不超过原图尺寸的方形金字塔层级 (sh, sw)。"""
    side = min(h, w)
    levels = []
    for s in sizes:
        if s <= side:
            levels.append((s, s))
    if not levels or levels[-1] != (side, side):
        levels.append((h, w))
    else:
        levels[-1] = (h, w)
    return sorted(set(levels), key=lambda x: x[0])


def _upsample_flow(flow: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, h, w = flow.shape
    if h == target_h and w == target_w:
        return flow
    scale_y = target_h / h
    scale_x = target_w / w
    out = flow.clone()
    out[:, 0, ...] *= scale_y
    out[:, 1, ...] *= scale_x
    return F.interpolate(out, size=(target_h, target_w), mode='bilinear', align_corners=True)


def _upsample_flow_stack(flow_stack: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """(1, M, 2, h, w) -> (1, M, 2, H, W)"""
    b, m, c, h, w = flow_stack.shape
    flat = flow_stack.reshape(b * m, c, h, w)
    flat_up = _upsample_flow(flat, target_h, target_w)
    return flat_up.reshape(b, m, c, target_h, target_w)


def _compose_flows(base_flow, delta_flow, shape_hw, device):
    transformer = SpatialTransformer(shape_hw).to(device)
    warped_delta = transformer(delta_flow, base_flow)
    return base_flow + warped_delta


def _compose_flow_stacks(base_stack, delta_stack, shape_hw, device):
    """逐波段 compose：(1,M,2,H,W)。"""
    m = base_stack.shape[1]
    composed = []
    for i in range(m):
        bi = base_stack[:, i, :, :, :]
        di = delta_stack[:, i, :, :, :]
        composed.append(_compose_flows(bi, di, shape_hw, device).unsqueeze(1))
    return torch.cat(composed, dim=1)


def _moving_band_indices(num_bands: int, anchor_idx: int) -> List[int]:
    return [i for i in range(num_bands) if i != anchor_idx]


def flow_stack_to_numpy(flow_stack: torch.Tensor) -> np.ndarray:
    """(1, M, 2, H, W) torch tensor → (M, 2, H, W) numpy float32."""
    return flow_stack.squeeze(0).detach().cpu().numpy().astype(np.float32)


def warp_bands_with_flow_stack(
    bands: StackInput,
    flow_stack: np.ndarray,
    anchor_band_idx: int = -1,
    device: str = 'cpu',
) -> List[np.ndarray]:
    """将位移场作用于任意强度图像（如原图），不改变像素动态范围。"""
    device = _resolve_device(device)
    band_list = _as_band_list(bands)
    h, w = band_list[0].shape
    flow_t = torch.tensor(flow_stack, dtype=torch.float32, device=device).unsqueeze(0)
    stack_t = _bands_to_tensor(band_list, device)
    warped_t = _warp_stack_perband(stack_t, flow_t, anchor_band_idx, (h, w))
    return _tensor_to_bands(warped_t)


def _flow_index_for_band(band_idx: int, anchor_idx: int) -> int:
    """moving band index -> flow_stack 通道索引。"""
    if band_idx == anchor_idx:
        raise ValueError('anchor band has no flow')
    return band_idx if band_idx < anchor_idx else band_idx - 1


def _warp_stack_perband(
    stack_t: torch.Tensor,
    flow_stack: torch.Tensor,
    anchor_idx: int,
    shape_hw,
) -> torch.Tensor:
    """
    stack_t: (1,N,H,W), flow_stack: (1,N-1,2,H,W)
    锚点波段不变形，其余各用对应位移场。
    """
    device = stack_t.device
    transformer = SpatialTransformer(shape_hw).to(device)
    n = stack_t.shape[1]
    warped = []
    for i in range(n):
        if i == anchor_idx:
            warped.append(stack_t[:, i : i + 1])
        else:
            fi = _flow_index_for_band(i, anchor_idx)
            flow_i = flow_stack[:, fi, :, :, :]
            warped.append(transformer(stack_t[:, i : i + 1], flow_i))
    return torch.cat(warped, dim=1)


def _create_lr_scheduler(optimizer, schedule, max_epochs, lr_min=1e-6):
    schedule = (schedule or 'none').lower()
    if schedule == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr_min
        ), 'cosine'
    if schedule == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=max(max_epochs // 10, 15),
            min_lr=lr_min, verbose=False
        ), 'plateau'
    return None, 'none'


def sequential_pairwise_ncc_loss(warped_stack: torch.Tensor, ncc_fn) -> torch.Tensor:
    """
    依次相邻波段 NCC 的平均值：NCC(warped[i-1], warped[i])，i=1..N-1。
    30 波段 → 29 项 NCC 损失再取平均（与链式参考一致，但位移场仍联合优化）。
    """
    n_bands = warped_stack.shape[1]
    if n_bands <= 1:
        return warped_stack.sum() * 0.0
    total = sum(
        ncc_fn(warped_stack[:, i - 1 : i], warped_stack[:, i : i + 1])
        for i in range(1, n_bands)
    )
    return total / (n_bands - 1)


def stack_grad_loss(preint_flow_stack: torch.Tensor, grad_fn) -> torch.Tensor:
    """对 (1,M,2,h,w) 各位移场求平滑正则并平均。"""
    m = preint_flow_stack.shape[1]
    total = sum(grad_fn(None, preint_flow_stack[:, i, :, :, :]) for i in range(m))
    return total / m


class SpectralEncoder1D(nn.Module):
    """
    沿光谱维对每个像素做 1D 卷积：(B, N, H, W) → (B, K, H, W)。
    比栈均值更能保留光谱结构，计算量仍是一次前向。
    """

    def __init__(self, num_bands: int, out_channels: int = DEFAULT_SPECTRAL_ENC_CHANNELS,
                 kernel_size: int = DEFAULT_SPECTRAL_ENC_KERNEL):
        super().__init__()
        pad = kernel_size // 2
        mid = max(out_channels * 2, 8)
        self.net = nn.Sequential(
            nn.Conv1d(num_bands, mid, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, out_channels, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
        )

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        b, n, h, w = stack.shape
        x = stack.reshape(b, n, h * w)
        x = self.net(x)
        return x.reshape(b, -1, h, w)


class PerBandStackFlowNet(nn.Module):
    """
    一次性输出 N-1 个 2D 位移场 (1, N-1, 2, H, W)。

    feature_mode:
      - mean_anchor: U-Net 输入 [栈均值, 锚点波段]
      - spectral_encoder: U-Net 输入 [SpectralEncoder1D(全栈), 锚点波段]
    """

    def __init__(
        self,
        inshape,
        num_bands,
        num_moving_bands,
        anchor_idx=-1,
        nb_unet_features=None,
        int_steps=3,
        int_downsize=2,
        feature_mode=FEATURE_MODE_MEAN_ANCHOR,
        spectral_enc_channels=DEFAULT_SPECTRAL_ENC_CHANNELS,
        spectral_enc_kernel=DEFAULT_SPECTRAL_ENC_KERNEL,
    ):
        super().__init__()
        ndims = len(inshape)
        self.num_bands = int(num_bands)
        self.num_moving = int(num_moving_bands)
        self.anchor_idx = int(anchor_idx)
        self.feature_mode = feature_mode
        self.spectral_enc_channels = int(spectral_enc_channels)
        enc_nf, dec_nf = nb_unet_features or compact_unet_features()

        if feature_mode == FEATURE_MODE_SPECTRAL_ENCODER:
            self.spectral_encoder = SpectralEncoder1D(
                self.num_bands, spectral_enc_channels, spectral_enc_kernel
            )
            unet_infeats = spectral_enc_channels + 1
        elif feature_mode == FEATURE_MODE_MEAN_ANCHOR:
            self.spectral_encoder = None
            unet_infeats = 2
        else:
            raise ValueError(
                f'Unknown feature_mode={feature_mode!r}, '
                f'expected {FEATURE_MODE_MEAN_ANCHOR!r} or {FEATURE_MODE_SPECTRAL_ENCODER!r}'
            )

        self.unet = Unet(inshape, infeats=unet_infeats, nb_features=[enc_nf, dec_nf])
        out_ch = self.num_moving * ndims
        self.flow = nn.Conv2d(self.unet.final_nf, out_ch, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        min_side = min(inshape)
        eff_steps = int_steps
        if int_downsize > 1 and min_side >= 32:
            self.resize = ResizeTransform(int_downsize, ndims)
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
            down_shape = [int(d / int_downsize) for d in inshape]
        else:
            self.resize = None
            self.fullsize = None
            down_shape = list(inshape)
            if min_side < 32:
                eff_steps = min(int_steps, 2)

        self.integrate = VecInt(down_shape, eff_steps) if eff_steps > 0 else None
        self.transformer = SpatialTransformer(inshape)

    def _stack_features(self, stack):
        ref = stack[:, self.anchor_idx : self.anchor_idx + 1]
        if self.feature_mode == FEATURE_MODE_SPECTRAL_ENCODER:
            spec = self.spectral_encoder(stack)
            return torch.cat([spec, ref], dim=1)
        mean = stack.mean(dim=1, keepdim=True)
        return torch.cat([mean, ref], dim=1)

    def _reshape_flow(self, flow_field, h, w):
        b = flow_field.shape[0]
        return flow_field.view(b, self.num_moving, 2, h, w)

    def _integrate_flow_stack(self, flow_stack):
        if self.integrate is None:
            return flow_stack
        b, m, _, h, w = flow_stack.shape
        flat = flow_stack.reshape(b * m, 2, h, w)
        flat = self.integrate(flat)
        return flat.reshape(b, m, 2, h, w)

    def _resize_flow_stack(self, flow_stack):
        if self.resize is None:
            return flow_stack
        b, m, _, h, w = flow_stack.shape
        flat = flow_stack.reshape(b * m, 2, h, w)
        flat = self.resize(flat)
        nh, nw = flat.shape[2], flat.shape[3]
        return flat.reshape(b, m, 2, nh, nw)

    def _fullsize_flow_stack(self, flow_stack):
        if self.fullsize is None:
            return flow_stack
        b, m, _, h, w = flow_stack.shape
        flat = flow_stack.reshape(b * m, 2, h, w)
        flat = self.fullsize(flat)
        nh, nw = flat.shape[2], flat.shape[3]
        return flat.reshape(b, m, 2, nh, nw)

    def predict_flow_stack(self, stack, registration=False):
        x = self._stack_features(stack)
        x = self.unet(x)
        raw = self.flow(x)
        _, _, h, w = raw.shape

        pos = self._reshape_flow(raw, h, w)
        pos = self._resize_flow_stack(pos)
        preint = pos
        pos = self._integrate_flow_stack(pos)
        pos = self._fullsize_flow_stack(pos)

        if registration:
            return pos, preint
        sh, sw = pos.shape[-2], pos.shape[-1]
        warped = _warp_stack_perband(stack, pos, self.anchor_idx, (sh, sw))
        return warped, preint


def _level_train_params(level_side: int, base_int_steps=3, base_int_downsize=2):
    if level_side <= 32:
        return min(base_int_steps, 2), 1
    return base_int_steps, base_int_downsize


def _train_stackflow_at_level(
    bands,
    device,
    anchor_idx,
    max_epochs,
    patience,
    lr,
    lamda,
    ncc_weight,
    int_steps,
    int_downsize,
    nb_unet_features,
    early_stop,
    min_delta,
    lr_schedule,
    lr_min,
    verbose,
    feature_mode=FEATURE_MODE_MEAN_ANCHOR,
    spectral_enc_channels=DEFAULT_SPECTRAL_ENC_CHANNELS,
    spectral_enc_kernel=DEFAULT_SPECTRAL_ENC_KERNEL,
):
    h, w = bands[0].shape
    n = len(bands)
    num_moving = n - 1
    stack_t = _bands_to_tensor(bands, device)

    int_steps, int_downsize = _level_train_params(min(h, w), int_steps, int_downsize)

    model = PerBandStackFlowNet(
        inshape=(h, w),
        num_bands=n,
        num_moving_bands=num_moving,
        anchor_idx=anchor_idx,
        nb_unet_features=nb_unet_features,
        int_steps=int_steps,
        int_downsize=int_downsize,
        feature_mode=feature_mode,
        spectral_enc_channels=spectral_enc_channels,
        spectral_enc_kernel=spectral_enc_kernel,
    ).to(device)

    grad_fn = Grad('l2', loss_mult=int_downsize).loss
    ncc_fn = NCC().loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler, schedule_type = _create_lr_scheduler(optimizer, lr_schedule, max_epochs, lr_min=lr_min)

    best_loss = float('inf')
    best_flow_stack = None
    best_state = None
    stale_epochs = 0
    log_every = max(max_epochs // 10, 1)

    for epoch in range(max_epochs):
        model.train()
        warped, preint = model.predict_flow_stack(stack_t, registration=False)
        loss = (
            ncc_weight * sequential_pairwise_ncc_loss(warped, ncc_fn)
            + lamda * stack_grad_loss(preint, grad_fn)
        )
        current_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if schedule_type == 'plateau':
            scheduler.step(current_loss)
        elif scheduler is not None:
            scheduler.step()

        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            stale_epochs = 0
            model.eval()
            with torch.no_grad():
                best_flow_stack, _ = model.predict_flow_stack(stack_t, registration=True)
                best_state = copy.deepcopy(model.state_dict())
            model.train()
        else:
            stale_epochs += 1

        if verbose and (epoch % log_every == 0 or epoch == max_epochs - 1):
            lr_now = optimizer.param_groups[0]['lr']
            print(
                f'    epoch {epoch + 1}/{max_epochs}: loss={current_loss:.4f} '
                f'best={best_loss:.4f} lr={lr_now:.2e}'
            )

        if early_stop and stale_epochs >= patience:
            if verbose:
                print(f'    early stop @ epoch {epoch + 1} (best={best_loss:.4f})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        flow_stack = best_flow_stack
    else:
        model.eval()
        with torch.no_grad():
            flow_stack, _ = model.predict_flow_stack(stack_t, registration=True)

    return model, flow_stack


def register_pifreg_groupwise_stackflow(
    img_list: StackInput,
    device: str = 'cuda',
    anchor_band_idx: int = -1,
    pyramid_sizes: Tuple[int, ...] = DEFAULT_PYRAMID_SIZES,
    epochs_per_level: Optional[Sequence[int]] = None,
    patience_per_level: Optional[Sequence[int]] = None,
    lr: float = 2e-4,
    lamda: float = 0.005,
    ncc_weight: float = 1.0,
    int_steps: int = 3,
    int_downsize: int = 2,
    nb_unet_features=None,
    early_stop: bool = True,
    min_delta: float = 1e-5,
    lr_schedule: str = 'cosine',
    lr_min: float = 1e-6,
    fast_mode: bool = True,
    feature_mode: str = FEATURE_MODE_MEAN_ANCHOR,
    spectral_enc_channels: int = DEFAULT_SPECTRAL_ENC_CHANNELS,
    spectral_enc_kernel: int = DEFAULT_SPECTRAL_ENC_KERNEL,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any], np.ndarray]:
    """
    Per-band 栈位移场配准：一次性联合优化 N-1 个位移场，粗→细金字塔传递。

    例：30 波段 @ 512² → 29 个位移场；金字塔默认 32→512。

    参数:
        anchor_band_idx: 锚点波段（默认 -1，即最后一波段，不动）
        feature_mode: 'mean_anchor' | 'spectral_encoder'（方案6 轻量光谱编码）
        spectral_enc_channels: 光谱编码输出通道数 K（默认 4）
        pyramid_sizes: 金字塔各层边长，默认 (32,64,128,256,512)
        ncc_weight: 相邻波段 NCC 均值项权重
        epochs_per_level / patience_per_level: 各层 epoch 与早停耐心
    """
    device = _resolve_device(device)
    bands = _as_band_list(img_list)
    n = len(bands)
    if n <= 1:
        return bands, {'mode': 'stackflow', 'num_bands': n}, np.zeros((0, 2, 0, 0), dtype=np.float32)

    feature_mode = (feature_mode or FEATURE_MODE_MEAN_ANCHOR).lower()
    if feature_mode not in (FEATURE_MODE_MEAN_ANCHOR, FEATURE_MODE_SPECTRAL_ENCODER):
        raise ValueError(
            f'Unknown feature_mode={feature_mode!r}, '
            f'expected {FEATURE_MODE_MEAN_ANCHOR!r} or {FEATURE_MODE_SPECTRAL_ENCODER!r}'
        )

    if fast_mode:
        nb_unet_features = nb_unet_features or compact_unet_features()
        lr = 2e-4
        lamda = 0.005

    h, w = bands[0].shape
    anchor_band_idx = int(anchor_band_idx) % n
    levels = _build_pyramid_levels(h, w, pyramid_sizes)

    ep_levels = list(epochs_per_level or DEFAULT_EPOCHS_PER_LEVEL)
    pat_levels = list(patience_per_level or DEFAULT_PATIENCE_PER_LEVEL)
    while len(ep_levels) < len(levels):
        ep_levels.append(ep_levels[-1])
    while len(pat_levels) < len(levels):
        pat_levels.append(pat_levels[-1])

    if verbose:
        feat_desc = (
            f'spectral_enc(K={spectral_enc_channels})+anchor'
            if feature_mode == FEATURE_MODE_SPECTRAL_ENCODER
            else 'mean+anchor'
        )
        print(
            f'{METHOD_NAME}: {n} bands, {n - 1} flow fields, anchor={anchor_band_idx}, '
            f'features={feat_desc}, pyramid={[f"{a}x{b}" for a, b in levels]}'
        )

    working = [b.copy() for b in bands]
    flow_stack_full = None

    for li, (sh, sw) in enumerate(levels):
        bands_s = _downsample_stack(working, sw, sh)

        if flow_stack_full is not None:
            flow_on_level = _upsample_flow_stack(flow_stack_full, sh, sw)
            stack_t = _bands_to_tensor(bands_s, device)
            stack_t = _warp_stack_perband(stack_t, flow_on_level, anchor_band_idx, (sh, sw))
            bands_s = _tensor_to_bands(stack_t)

        ep = ep_levels[li]
        pat = pat_levels[li]
        if verbose:
            print(
                f'Level {li + 1}/{len(levels)}: {sh}x{sw}, '
                f'{n - 1} flows, max_epochs={ep}, patience={pat}'
            )

        _, flow_delta = _train_stackflow_at_level(
            bands_s, device, anchor_band_idx, ep, pat, lr, lamda,
            ncc_weight, int_steps, int_downsize, nb_unet_features,
            early_stop, min_delta, lr_schedule, lr_min, verbose,
            feature_mode=feature_mode,
            spectral_enc_channels=spectral_enc_channels,
            spectral_enc_kernel=spectral_enc_kernel,
        )

        if flow_stack_full is None:
            flow_stack_full = flow_delta
        else:
            flow_prev = _upsample_flow_stack(flow_stack_full, sh, sw)
            flow_stack_full = _compose_flow_stacks(flow_prev, flow_delta, (sh, sw), device)

    flow_stack_full = _upsample_flow_stack(flow_stack_full, h, w)
    stack_orig = _bands_to_tensor(working, device)
    warped_t = _warp_stack_perband(stack_orig, flow_stack_full, anchor_band_idx, (h, w))
    registered = _tensor_to_bands(warped_t)

    unet_feats = nb_unet_features or compact_unet_features()
    flow_np = flow_stack_to_numpy(flow_stack_full)
    info = {
        'mode': 'stackflow',
        'method': METHOD_FULL_NAME,
        'num_bands': n,
        'num_flow_fields': n - 1,
        'anchor_band_idx': anchor_band_idx,
        'moving_band_indices': _moving_band_indices(n, anchor_band_idx),
        'flow_stack_shape': list(flow_np.shape),
        'pyramid_sizes': list(pyramid_sizes),
        'pyramid_levels': [list(lv) for lv in levels],
        'epochs_per_level': ep_levels[: len(levels)],
        'patience_per_level': pat_levels[: len(levels)],
        'lr': lr,
        'lamda': lamda,
        'ncc_weight': ncc_weight,
        'int_steps': int_steps,
        'int_downsize': int_downsize,
        'nb_unet_features': [list(unet_feats[0]), list(unet_feats[1])],
        'early_stop': early_stop,
        'min_delta': min_delta,
        'lr_schedule': lr_schedule,
        'lr_min': lr_min,
        'shared_flow': False,
        'loss': 'sequential_pairwise_ncc_mean + per_flow_grad',
        'fast_mode': fast_mode,
        'feature_mode': feature_mode,
        'spectral_enc_channels': spectral_enc_channels,
        'spectral_enc_kernel': spectral_enc_kernel,
        'device': str(device),
    }
    return registered, info, flow_np
