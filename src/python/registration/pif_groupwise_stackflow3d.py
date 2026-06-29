# PIFReg StackFlow3D — 方案 A：3D U-Net 全栈输入 + per-band 2D flow warp
#
# 输入：(1, 1, N, H, W) 全光谱立方体（不做 mean 压缩）
# 3D U-Net：仅在 H×W 上 pool，光谱维 N 保持完整
# 输出：(1, N, 2, H, W) 各 band 的 xy 位移（锚点 band flow 恒为 0）
# Warp / Loss：与 2D StackFlow 相同（2D SpatialTransformer + 链式 NCC）

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from ..losses.registration_losses import Grad, NCC
from ..voxelmorph.config import compact_unet_features
from ..voxelmorph.layers import ResizeTransform, SpatialTransformer, VecInt

from .pif_groupwise_stackflow import (
    DEFAULT_EPOCHS_PER_LEVEL,
    DEFAULT_PATIENCE_PER_LEVEL,
    DEFAULT_PYRAMID_SIZES,
    StackInput,
    _as_band_list,
    _bands_to_tensor,
    _build_pyramid_levels,
    _compose_flows,
    _create_lr_scheduler,
    _downsample_stack,
    _resolve_device,
    _tensor_to_bands,
    _upsample_flow,
    sequential_pairwise_ncc_loss,
)

METHOD_NAME = 'PIFReg-StackFlow3D'
METHOD_FULL_NAME = 'PIFReg 3D Stack Flow Groupwise Registration (Scheme A)'


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.conv(x))


class SpectralStackUnet3d(nn.Module):
    """
    各向异性 3D U-Net：输入 (B, C, N, H, W)，MaxPool 仅 (1,2,2)，保留光谱维 N。
    """

    def __init__(self, in_channels=1, nb_features=None):
        super().__init__()
        enc_nf, dec_nf = nb_features or compact_unet_features()
        tail_nf = dec_nf[len(enc_nf) :]
        dec_nf = dec_nf[: len(enc_nf)]

        self.downs = nn.ModuleList()
        ch = in_channels
        skip_channels: List[int] = []
        for nf in enc_nf:
            self.downs.append(ConvBlock3d(ch, nf))
            skip_channels.append(nf)
            ch = nf

        self.bottom = ConvBlock3d(ch, ch)

        self.ups = nn.ModuleList()
        for i, nf in enumerate(reversed(dec_nf)):
            skip_ch = skip_channels[-(i + 1)]
            self.ups.append(ConvBlock3d(ch + skip_ch, nf))
            ch = nf

        tail_layers = []
        for nf in tail_nf:
            tail_layers.append(ConvBlock3d(ch, nf))
            ch = nf
        self.tail = nn.Sequential(*tail_layers)
        self.final_nf = ch

    @staticmethod
    def _pool(x):
        return F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')

    @staticmethod
    def _crop_to(x, ref):
        _, _, nd, nh, nw = ref.shape
        return x[:, :, :nd, :nh, :nw]

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self._pool(x)

        x = self.bottom(x)

        for i, up in enumerate(self.ups):
            x = self._up(x)
            skip = skips[-(i + 1)]
            x = self._crop_to(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = up(x)

        return self.tail(x)


def _upsample_flow_volume(flow_vol: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """(1, N, 2, h, w) -> (1, N, 2, H, W)"""
    b, n, c, h, w = flow_vol.shape
    flat = flow_vol.reshape(b * n, c, h, w)
    flat_up = _upsample_flow(flat, target_h, target_w)
    return flat_up.reshape(b, n, c, target_h, target_w)


def _compose_flow_volumes(base_vol, delta_vol, shape_hw, anchor_idx, device):
    """(1, N, 2, H, W) 逐 band compose，锚点保持零位移。"""
    n = base_vol.shape[1]
    composed = []
    for i in range(n):
        if i == anchor_idx:
            composed.append(torch.zeros_like(base_vol[:, i : i + 1]))
            continue
        bi = base_vol[:, i, :, :, :]
        di = delta_vol[:, i, :, :, :]
        composed.append(_compose_flows(bi, di, shape_hw, device).unsqueeze(1))
    return torch.cat(composed, dim=1)


def _warp_stack_band_flows(stack_t, flow_vol, anchor_idx, shape_hw):
    """stack (1,N,H,W), flow_vol (1,N,2,H,W)"""
    device = stack_t.device
    transformer = SpatialTransformer(shape_hw).to(device)
    n = stack_t.shape[1]
    warped = []
    for i in range(n):
        if i == anchor_idx:
            warped.append(stack_t[:, i : i + 1])
        else:
            flow_i = flow_vol[:, i, :, :, :]
            warped.append(transformer(stack_t[:, i : i + 1], flow_i))
    return torch.cat(warped, dim=1)


def _volume_grad_loss(preint_vol, grad_fn):
    """preint_vol: (1, N-1, 2, h, w) 仅 moving bands，在积分分辨率上正则。"""
    m = preint_vol.shape[1]
    if m == 0:
        return preint_vol.sum() * 0.0
    total = sum(grad_fn(None, preint_vol[:, i, :, :, :]) for i in range(m))
    return total / m


class StackFlowNet3d(nn.Module):
    """
    方案 A：3D U-Net 读全栈，输出 (1, N, 2, H, W)；各 slice 独立 2D diffeomorphic 积分。
    """

    def __init__(
        self,
        num_bands,
        spatial_shape,
        anchor_idx=0,
        nb_unet_features=None,
        int_steps=3,
        int_downsize=2,
    ):
        super().__init__()
        self.num_bands = int(num_bands)
        self.anchor_idx = int(anchor_idx)
        h, w = spatial_shape

        self.unet3d = SpectralStackUnet3d(in_channels=1, nb_features=nb_unet_features)
        self.flow = nn.Conv3d(self.unet3d.final_nf, 2, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        min_side = min(h, w)
        eff_steps = int_steps
        if int_downsize > 1 and min_side >= 32:
            self.resize = ResizeTransform(int_downsize, 2)
            self.fullsize = ResizeTransform(1 / int_downsize, 2)
            down_shape = [int(h / int_downsize), int(w / int_downsize)]
        else:
            self.resize = None
            self.fullsize = None
            down_shape = [h, w]
            if min_side < 32:
                eff_steps = min(int_steps, 2)

        self.integrate = VecInt(down_shape, eff_steps) if eff_steps > 0 else None
        self.int_downsize = int_downsize if self.resize is not None else 1

    def _stack_to_volume(self, stack):
        return stack.unsqueeze(1)

    def _raw_to_band_flows(self, raw):
        """(1, 2, N, H, W) -> (1, N, 2, H, W)"""
        return raw.permute(0, 2, 1, 3, 4).contiguous()

    def _process_band_flows(self, flow_vol):
        b, n, _, h, w = flow_vol.shape
        out_slices = []
        preint_slices = []

        for i in range(n):
            if i == self.anchor_idx:
                z = torch.zeros(b, 2, h, w, device=flow_vol.device, dtype=flow_vol.dtype)
                out_slices.append(z)
                continue
            fi = flow_vol[:, i, :, :, :]
            pos = fi
            if self.resize is not None:
                pos = self.resize(pos)
            pre = pos
            if self.integrate is not None:
                pos = self.integrate(pos)
            if self.fullsize is not None:
                pos = self.fullsize(pos)
            out_slices.append(pos)
            preint_slices.append(pre)

        pos_vol = torch.stack(out_slices, dim=1)
        if preint_slices:
            pre_vol = torch.stack(preint_slices, dim=1)
        else:
            pre_vol = pos_vol.new_zeros(b, 0, 2, h, w)
        return pos_vol, pre_vol

    def predict_flow_volume(self, stack, registration=False):
        vol = self._stack_to_volume(stack)
        feat = self.unet3d(vol)
        raw = self.flow(feat)
        band_raw = self._raw_to_band_flows(raw)
        pos, preint = self._process_band_flows(band_raw)

        if registration:
            return pos, preint
        sh, sw = pos.shape[-2], pos.shape[-1]
        warped = _warp_stack_band_flows(stack, pos, self.anchor_idx, (sh, sw))
        return warped, preint


def _level_train_params(level_side: int, base_int_steps=3, base_int_downsize=2):
    if level_side <= 32:
        return min(base_int_steps, 2), 1
    return base_int_steps, base_int_downsize


def _train_stackflow3d_at_level(
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
):
    h, w = bands[0].shape
    n = len(bands)
    stack_t = _bands_to_tensor(bands, device)

    int_steps, int_downsize = _level_train_params(min(h, w), int_steps, int_downsize)

    model = StackFlowNet3d(
        num_bands=n,
        spatial_shape=(h, w),
        anchor_idx=anchor_idx,
        nb_unet_features=nb_unet_features,
        int_steps=int_steps,
        int_downsize=int_downsize,
    ).to(device)

    grad_fn = Grad('l2', loss_mult=model.int_downsize).loss
    ncc_fn = NCC().loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler, schedule_type = _create_lr_scheduler(optimizer, lr_schedule, max_epochs, lr_min=lr_min)

    best_loss = float('inf')
    best_flow_vol = None
    best_state = None
    stale_epochs = 0
    log_every = max(max_epochs // 10, 1)

    for epoch in range(max_epochs):
        model.train()
        warped, preint = model.predict_flow_volume(stack_t, registration=False)
        loss = (
            ncc_weight * sequential_pairwise_ncc_loss(warped, ncc_fn)
            + lamda * _volume_grad_loss(preint, grad_fn)
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
                best_flow_vol, _ = model.predict_flow_volume(stack_t, registration=True)
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
        flow_vol = best_flow_vol
    else:
        model.eval()
        with torch.no_grad():
            flow_vol, _ = model.predict_flow_volume(stack_t, registration=True)

    return model, flow_vol


def register_pifreg_groupwise_stackflow3d(
    img_list: StackInput,
    device: str = 'cuda',
    anchor_band_idx: int = 0,
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
    verbose: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    方案 A：3D U-Net 全栈联合配准。

    输入栈 (N,H,W) 作为 (1,1,N,H,W) 立方体；输出 per-band 2D flow；warp 仍为 2D slice。
    """
    device = _resolve_device(device)
    bands = _as_band_list(img_list)
    n = len(bands)
    if n <= 1:
        return bands, {'mode': 'stackflow3d', 'num_bands': n}, np.zeros((0, 2, 0, 0), dtype=np.float32)

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
        print(
            f'{METHOD_NAME}: {n} bands, 3D U-Net input (1,1,{n},H,W), anchor={anchor_band_idx}, '
            f'pyramid={[f"{a}x{b}" for a, b in levels]}'
        )

    working = [b.copy() for b in bands]
    flow_vol_full = None

    for li, (sh, sw) in enumerate(levels):
        bands_s = _downsample_stack(working, sw, sh)

        if flow_vol_full is not None:
            flow_on_level = _upsample_flow_volume(flow_vol_full, sh, sw)
            stack_t = _bands_to_tensor(bands_s, device)
            stack_t = _warp_stack_band_flows(stack_t, flow_on_level, anchor_band_idx, (sh, sw))
            bands_s = _tensor_to_bands(stack_t)

        ep = ep_levels[li]
        pat = pat_levels[li]
        if verbose:
            print(
                f'Level {li + 1}/{len(levels)}: {sh}x{sw} x {n} bands, '
                f'max_epochs={ep}, patience={pat}'
            )

        _, flow_delta = _train_stackflow3d_at_level(
            bands_s, device, anchor_band_idx, ep, pat, lr, lamda,
            ncc_weight, int_steps, int_downsize, nb_unet_features,
            early_stop, min_delta, lr_schedule, lr_min, verbose,
        )

        if flow_vol_full is None:
            flow_vol_full = flow_delta
        else:
            flow_prev = _upsample_flow_volume(flow_vol_full, sh, sw)
            flow_vol_full = _compose_flow_volumes(
                flow_prev, flow_delta, (sh, sw), anchor_band_idx, device
            )

    flow_vol_full = _upsample_flow_volume(flow_vol_full, h, w)
    stack_orig = _bands_to_tensor(working, device)
    warped_t = _warp_stack_band_flows(stack_orig, flow_vol_full, anchor_band_idx, (h, w))
    registered = _tensor_to_bands(warped_t)

    flow_np = flow_vol_full.squeeze(0).detach().cpu().numpy().astype(np.float32)
    moving_idx = [i for i in range(n) if i != anchor_band_idx]
    flow_stack_np = flow_np[moving_idx]
    info = {
        'mode': 'stackflow3d',
        'scheme': 'A',
        'method': METHOD_FULL_NAME,
        'num_bands': n,
        'num_flow_fields': n - 1,
        'anchor_band_idx': anchor_band_idx,
        'moving_band_indices': moving_idx,
        'flow_stack_shape': list(flow_stack_np.shape),
        'pyramid_levels': [list(lv) for lv in levels],
        'epochs_per_level': ep_levels[: len(levels)],
        'network': 'SpectralStackUnet3d (anisotropic pool 1x2x2)',
        'input_shape': f'(1,1,{n},{h},{w})',
        'flow_shape': f'(1,{n},2,{h},{w})',
        'loss': 'sequential_pairwise_ncc_mean + per_flow_grad',
        'ncc_weight': ncc_weight,
        'fast_mode': fast_mode,
    }
    return registered, info, flow_stack_np
