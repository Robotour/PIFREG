# PIFReg Joint Groupwise — 将 N 个波段视为 (N,H,W) 栈，单次优化共享位移场
#
# 对应 Elastix groupwise 思想（VarianceOverLastDimensionMetric + 共同形变空间）：
#   - 一个 U-Net 预测整张 2D 位移场
#   - 同一位移场同时 warp 全部波段
#   - 损失 = 光谱维方差（SubtractMean 后）+ 可选 NCC(栈均值) + 平滑正则
#
# 与 pairwise 栈配准 (chain/mean) 相比：仅 1 次（×金字塔层数）优化，而非 N 次 PIFReg。

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from ..losses.registration_losses import Grad, NCC
from ..voxelmorph.config import compact_unet_features, default_unet_features
from ..voxelmorph.layers import ResizeTransform, SpatialTransformer, VecInt
from ..voxelmorph.networks import Unet

METHOD_NAME = 'PIFReg-Joint'
METHOD_FULL_NAME = 'PIFReg Joint Stack Groupwise Registration'

StackInput = Union[Sequence[np.ndarray], np.ndarray]


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
    """(N,H,W) -> (1,N,H,W)"""
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


def _upsample_flow(flow, target_h, target_w):
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
    transformer = SpatialTransformer(shape_hw).to(device)
    warped_delta = transformer(delta_flow, base_flow)
    return base_flow + warped_delta


def _warp_stack_with_flow(stack_t: torch.Tensor, flow: torch.Tensor, shape_hw) -> torch.Tensor:
    """对 (1,N,H,W) 栈施加同一 (1,2,H,W) 位移场。"""
    device = stack_t.device
    transformer = SpatialTransformer(shape_hw).to(device)
    n = stack_t.shape[1]
    warped = [transformer(stack_t[:, i : i + 1], flow) for i in range(n)]
    return torch.cat(warped, dim=1)


def _create_lr_scheduler(optimizer, schedule, max_epochs, lr_gamma=0.5, lr_min=1e-6):
    schedule = (schedule or 'none').lower()
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


class JointStackFlowNet(nn.Module):
    """
    共享 2D 位移场网络：输入栈的 [均值, 参考波段]，输出作用于全部波段的形变。
    """

    def __init__(self, inshape, nb_unet_features=None, int_steps=3, int_downsize=2, ref_band_idx=0):
        super().__init__()
        ndims = len(inshape)
        enc_nf, dec_nf = nb_unet_features or compact_unet_features()
        self.ref_band_idx = int(ref_band_idx)

        self.unet = Unet(
            inshape,
            infeats=2,
            nb_features=[enc_nf, dec_nf],
        )
        self.flow = nn.Conv2d(self.unet.final_nf, ndims, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        if int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.resize = None
            self.fullsize = None

        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None
        self.transformer = SpatialTransformer(inshape)

    def _stack_features(self, stack):
        """stack: (1,N,H,W) -> U-Net input (1,2,H,W)"""
        mean = stack.mean(dim=1, keepdim=True)
        ref_idx = min(self.ref_band_idx, stack.shape[1] - 1)
        ref = stack[:, ref_idx : ref_idx + 1]
        return torch.cat([mean, ref], dim=1)

    def predict_flow(self, stack, registration=False):
        x = self._stack_features(stack)
        x = self.unet(x)
        flow_field = self.flow(x)

        pos_flow = flow_field
        if self.resize is not None:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow
        if self.integrate is not None:
            pos_flow = self.integrate(pos_flow)
            if self.fullsize is not None:
                pos_flow = self.fullsize(pos_flow)

        if registration:
            return pos_flow, preint_flow
        warped = self.warp_stack(stack, pos_flow)
        return warped, preint_flow

    def warp_stack(self, stack, flow):
        n = stack.shape[1]
        warped = [self.transformer(stack[:, i : i + 1], flow) for i in range(n)]
        return torch.cat(warped, dim=1)


def spectral_variance_loss(warped_stack: torch.Tensor) -> torch.Tensor:
    """
    模拟 Elastix VarianceOverLastDimensionMetric + SubtractMean：
    各波段 warp 后减去栈均值，再最小化光谱维方差。
    """
    mean = warped_stack.mean(dim=1, keepdim=True)
    centered = warped_stack - mean
    return (centered ** 2).mean()


def mean_ncc_loss(warped_stack: torch.Tensor, ncc_fn) -> torch.Tensor:
    """各波段与 warp 后栈均值的 NCC 相似性。"""
    mean = warped_stack.mean(dim=1, keepdim=True)
    n_bands = warped_stack.shape[1]
    total = sum(ncc_fn(mean, warped_stack[:, i : i + 1]) for i in range(n_bands))
    return total / n_bands


def _train_joint_at_scale(
    bands,
    device,
    max_epochs,
    lr,
    lamda,
    var_weight,
    ncc_weight,
    int_steps,
    int_downsize,
    nb_unet_features,
    ref_band_idx,
    early_stop,
    patience,
    min_delta,
    lr_schedule,
    lr_min,
    verbose,
):
    h, w = bands[0].shape
    stack_t = _bands_to_tensor(bands, device)

    model = JointStackFlowNet(
        inshape=(h, w),
        nb_unet_features=nb_unet_features,
        int_steps=int_steps,
        int_downsize=int_downsize,
        ref_band_idx=ref_band_idx,
    ).to(device)

    grad_fn = Grad('l2', loss_mult=int_downsize).loss
    ncc_fn = NCC().loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler, schedule_type = _create_lr_scheduler(optimizer, lr_schedule, max_epochs, lr_min=lr_min)

    best_loss = float('inf')
    best_flow = None
    best_state = None
    stale_epochs = 0
    log_every = max(max_epochs // 12, 1)

    for epoch in range(max_epochs):
        model.train()
        warped, preint_flow = model.predict_flow(stack_t, registration=False)
        loss = (
            var_weight * spectral_variance_loss(warped)
            + ncc_weight * mean_ncc_loss(warped, ncc_fn)
            + lamda * grad_fn(None, preint_flow)
        )
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
                best_flow, _ = model.predict_flow(stack_t, registration=True)
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
                print(f'  early stop at epoch {epoch + 1} (best_loss={best_loss:.4f})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        flow = best_flow
    else:
        model.eval()
        with torch.no_grad():
            flow, _ = model.predict_flow(stack_t, registration=True)

    return model, flow


def register_pifreg_groupwise_joint(
    img_list: StackInput,
    device: str = 'cuda',
    epochs: int = 2000,
    lr: float = 2e-4,
    lamda: float = 0.005,
    var_weight: float = 1.0,
    ncc_weight: float = 0.5,
    int_steps: int = 3,
    int_downsize: int = 2,
    nb_unet_features=None,
    multiscale: bool = True,
    scales: Tuple[float, ...] = (0.25, 0.5, 1.0),
    ref_band_idx: Optional[int] = None,
    early_stop: bool = True,
    patience: int = 60,
    min_delta: float = 1e-5,
    lr_schedule: str = 'cosine',
    lr_min: float = 1e-6,
    fast_mode: bool = True,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    联合栈配准：一次（多尺度）优化，共享位移场对齐全部波段。

    参数:
        img_list: 按波长升序的 (H,W) 列表或 (N,H,W) 数组
        var_weight: 光谱方差项权重（Elastix groupwise 主损失 analog）
        ncc_weight: 与 warp 后栈均值的 NCC 项权重
        ref_band_idx: U-Net 输入中的参考波段；None 则用中间波段
        fast_mode: 轻量 U-Net + 默认 int_steps=3

    返回:
        registered_list: 配准后各波段
        info: 元数据（共享 flow，无 per-band 位移场列表）
    """
    device = _resolve_device(device)
    bands = _as_band_list(img_list)
    n = len(bands)
    if n <= 1:
        return bands, {'mode': 'joint', 'num_bands': n}

    if fast_mode:
        nb_unet_features = nb_unet_features or compact_unet_features()
        int_steps = 3
        lr = 2e-4
        lamda = 0.005
        scales = (0.25, 0.5, 1.0)
        patience = min(patience, 60)

    if ref_band_idx is None:
        ref_band_idx = n // 2

    h, w = bands[0].shape
    working = [b.copy() for b in bands]

    if verbose:
        print(
            f'{METHOD_NAME}: joint stack optimization, bands={n}, '
            f'scales={scales if multiscale else (1.0,)}, ref_band={ref_band_idx}'
        )

    flow_full = None

    if multiscale:
        active = [s for s in scales if int(h * s) >= 32 and int(w * s) >= 32]
        if not active or active[-1] != 1.0:
            active = list(active) + [1.0]
        active = sorted(set(active))
    else:
        active = [1.0]

    for scale in active:
        sh, sw = int(h * scale), int(w * scale)
        bands_s = _downsample_stack(working, sw, sh)

        if flow_full is not None:
            flow_on_scale = _upsample_flow(flow_full, sh, sw)
            stack_t = _bands_to_tensor(bands_s, device)
            stack_t = _warp_stack_with_flow(stack_t, flow_on_scale, (sh, sw))
            bands_s = _tensor_to_bands(stack_t)

        if verbose:
            print(f'Scale {scale:.2f} ({sh}x{sw}) — single joint optimization')

        _, flow_delta = _train_joint_at_scale(
            bands_s, device, epochs, lr, lamda, var_weight, ncc_weight,
            int_steps, int_downsize, nb_unet_features, ref_band_idx,
            early_stop, patience, min_delta, lr_schedule, lr_min, verbose,
        )

        if flow_full is None:
            flow_full = flow_delta
        else:
            flow_prev = _upsample_flow(flow_full, sh, sw)
            flow_full = _compose_flows(flow_prev, flow_delta, (sh, sw), device)

    flow_full = _upsample_flow(flow_full, h, w)
    stack_orig = _bands_to_tensor(working, device)
    warped_t = _warp_stack_with_flow(stack_orig, flow_full, (h, w))
    registered = _tensor_to_bands(warped_t)

    info = {
        'mode': 'joint',
        'method': METHOD_FULL_NAME,
        'num_bands': n,
        'ref_band_idx': ref_band_idx,
        'scales': list(active),
        'shared_flow': True,
        'loss': 'spectral_variance + mean_ncc + grad',
        'var_weight': var_weight,
        'ncc_weight': ncc_weight,
        'fast_mode': fast_mode,
    }
    return registered, info
