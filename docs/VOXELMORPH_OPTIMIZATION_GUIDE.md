# VoxelMorph 配准分析与优化指南

> 针对本项目 `src/python/` 中 VoxelMorph 实现的效果分析与改进方案。  
> 适用场景：高光谱舌诊图像的**相邻波段**配准。

---

## 1. 当前实现框架概览

### 1.1 整体数据流

```
固定图像 fixed (H×W)          移动图像 moving (H×W)
        │                              │
        └──────────┬───────────────────┘
                   ▼
         concat → [B, 2, H, W]  （通道0=moving, 通道1=fixed）
                   ▼
         RegistrationUNet（U-Net）
                   ▼
         位移场 flow [B, 2, H, W]   （dx, dy，单位：像素）
                   ▼
         SpatialTransformer（双线性 grid_sample）
                   ▼
         warped_moving [B, 1, H, W]
                   ▼
    Loss = LNCC(fixed, warped) + λ · Grad(flow)     （λ 默认 10）
                   ▼
         Adam 反向传播，更新 U-Net 权重
```

**关键代码位置：**

| 模块 | 文件 | 作用 |
|------|------|------|
| 训练入口 | `registration/methods.py` → `register_voxelmorph()` | 单对图像、从零训练 |
| 网络 | `networks/registration_networks.py` → `RegistrationUNet` | 预测位移场 |
| 空间变换 | `networks/registration_networks.py` → `SpatialTransformer` | 根据 flow 重采样 |
| 相似性损失 | `losses/registration_losses.py` → `NCC` | 局部归一化互相关 |
| 正则项 | `losses/registration_losses.py` → `Grad` | 位移场平滑（L2 梯度惩罚） |
| 判别器（未使用） | `networks/registration_networks.py` → `Discriminator` | 已定义，训练流程中未接入 |

### 1.2 RegistrationUNet 结构

```
输入 [moving, fixed]  2通道
    │
Encoder（5层，stride=2 下采样）
    2 → 16 → 32 → 32 → 32 → 32
    │    │    │    │    └── skip L4
    │    │    │    └─────── skip L3
    │    │    └──────────── skip L2
    │    └───────────────── skip L1
    └────────────────────── skip L0
    │
    最深层特征 x = L4
    │
Decoder（上采样 + skip 拼接，部分层无 skip）
    up → cat(L3) → conv
    up → cat(L2) → conv
    up → cat(L1) → conv
    up → cat(L0) → conv
    up → conv → conv → conv   ← 最后 4 层未再拼接 skip
    │
输出 2 通道位移场 (dx, dy)
```

**与标准 VoxelMorph 论文实现的差异：**

| 项目 | 论文/官方 VoxelMorph | 本项目 |
|------|---------------------|--------|
| 训练方式 | 在大规模数据集上**预训练**，推理时一次前向 | 每对图像**随机初始化、现场训练 300 epoch** |
| 相似性度量 | 多为 MSE（同模态）或 LNCC | LNCC，窗口 25×25 |
| 多尺度 | 通常有多分辨率策略 | 单尺度 512×512 |
| 仿射初始化 | 常见 coarse-to-fine | 无，从零位移开始 |
| 网络容量 | 通常更深（如 16→32→32→32→32→32） | 较浅，末层 skip 连接不完整 |

### 1.3 当前训练超参数（默认值）

```python
# registration/methods.py
lr = 1e-4
epochs = 300
lamda = 10          # 平滑正则权重
NCC win = [25, 25]
optimizer = Adam(betas=(0.9, 0.999))
```

实验脚本 `run_registration_exp.py` 中图像会被 resize 到 **512×512** 并做 **各自独立的 Min-Max 归一化**。

---

## 2. 为什么效果远不如 Elastix / StackReg？

这不是"VoxelMorph 方法本身一定更差"，而是**当前用法与任务特性叠加**导致处于明显劣势。

### 2.1 根本原因对比

| 维度 | Elastix | StackReg | 本项目 VoxelMorph |
|------|---------|----------|-------------------|
| 优化目标 | Mattes MI（对强度变化更鲁棒） | 像素/特征相关 + 全局仿射/双线性 | 局部 NCC + 强平滑 |
| 初始化 | 多分辨率金字塔，由粗到细 | 全局变换模型（平移/旋转/仿射） | 零位移，随机权重 |
| 是否预训练 | 不需要（经典优化） | 不需要 | **需要大量数据预训练才有优势，当前未做** |
| 位移表达能力 | B-spline 自由形变，网格间距可控 | 全局 bilinear（6+ 自由度） | 逐像素位移，但被 λ=10 强烈压制 |
| 对跨波段强度差异 | MI 专门处理"模态差异" | 仿射+双线性对局部强度不敏感 | NCC 假设局部线性强度关系，跨波段易失效 |

### 2.2 针对高光谱波段配准的具体问题

**① 相邻波段外观不一致（pseudo-multimodal）**

639 nm 与 650 nm 波段对比度、亮度分布不同。Elastix 的 MI 直接最大化统计依赖；LNCC 假设局部 `moving ≈ a·fixed + b`，跨波段时该假设变弱，网络容易陷入**小位移局部最优**。

**② 平滑正则过强（λ=10）**

`Grad` 惩罚位移场空间变化，等价于鼓励"位移尽量均匀、尽量小"。  
波段间可能存在**几像素到十几像素**的平移/轻微旋转，强平滑会把大位移"抹平"，网络只学到近似恒等映射。

**③ 无粗配准初始化**

StackReg 一步估计全局 bilinear 变换；KEREN 用 5 层金字塔。  
VoxelMorph 从随机 U-Net 开始，300 epoch 内可能来不及找到正确的全局对齐，后续只在小范围内微调。

**④ 单尺度 + 固定 512 resize**

小位移在 resize 后可能被缩放；纹理细节损失使 NCC 梯度变弱。Elastix 内置多分辨率；StackReg 在原始分辨率上估计全局变换。

**⑤ 每对图像重新训练，无知识迁移**

标准 VoxelMorph 的价值在于：**一次训练，百万对推理**。  
当前实现相当于"用一个小网络做 300 步的数值优化"，参数量与搜索能力均弱于 Elastix 的成熟优化器。

**⑥ 架构与实现细节**

- `Discriminator` 已写好但未使用，GAN 约束缺失。
- Decoder 最后几层未使用 skip connection，浅层细节利用不足。
- NCC 窗口 25×25 在 512 图像上约占 5%，对舌象这种**局部结构丰富、全局对比度低**的图像可能过度平滑相似性度量。

---

## 3. 优化路线图（按优先级）

建议按 **P0 → P1 → P2** 顺序尝试，每步固定其他变量、记录 MI/NCC/NTG 变化。

```
P0  快速调参（不改架构，1 小时内）
P1  流程改进（仿射初始化 + 多尺度，1–2 天）
P2  损失与训练策略升级（3–7 天）
P3  预训练 / 换用官方 VoxelMorph（长期）
```

---

## 4. P0：快速调参（最先做）

### 4.1 降低平滑权重 λ

**问题**：λ=10 很可能过大，位移场被"锁死"。

**建议**：在 `register_voxelmorph` 中扫描：

```python
for lamda in [0.1, 0.5, 1.0, 3.0, 10.0]:
    warped = register_voxelmorph(fixed, moving, lamda=lamda, epochs=500)
```

| λ | 预期效果 |
|---|----------|
| 0.1–1.0 | 允许更大位移，适合波段错位明显的对 |
| 3.0–10.0 | 位移更平滑，适合噪声大但错位小的对 |

**经验起点**：高光谱相邻波段先试 **`lamda=1.0`** 或 **`0.5`**。

### 4.2 增加训练轮数 + 学习率衰减

300 epoch 对随机初始化可能不够。

```python
# 建议起步配置
epochs = 1000
lr = 1e-4

# 在训练循环中加入（methods.py 内）
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
# 每个 epoch 末尾: scheduler.step()
```

观察 `loss` 是否在第 300–600 epoch 仍在下降；若下降，继续加长训练。

### 4.3 缩小 NCC 窗口

```python
# 当前
ncc_loss = NCC(win=[25, 25])

# 建议尝试
ncc_loss = NCC(win=[9, 9])   # 更局部，对结构对齐更敏感
# 或
ncc_loss = NCC(win=[15, 15])
```

窗口越小 → 对局部对齐越敏感，但对噪声也更敏感。

### 4.4 不要过度 resize

若原始图像接近 512×512，可改为保留原始尺寸或 256 多尺度训练：

```python
# run_registration_exp.py
fixed, moving = load_images(fixed_path, moving_path, image_size=None)  # 需改 load_images 支持
```

---

## 5. P1：流程改进（效果显著）

### 5.1 StackReg / KEREN 仿射预配准（强烈推荐）

思路：**先全局、后形变**。用 StackReg 估计 bilinear/affine，把 moving 粗对齐后再跑 VoxelMorph 做残差形变。

```python
import cv2
import numpy as np
from pystackreg import StackReg
from src.python.registration import register_voxelmorph

def register_voxelmorph_with_affine_init(fixed, moving, **kwargs):
    # Step 1: StackReg 粗配准
    sr = StackReg(StackReg.BILINEAR)
    sr.register(fixed, moving)
    moving_coarse = sr.transform(moving)

    # Step 2: VoxelMorph 在粗对齐结果上估计残差位移
    warped = register_voxelmorph(fixed, moving_coarse, lamda=1.0, epochs=800, **kwargs)
    return warped
```

**原理**：StackReg 擅长的全局变换由传统方法完成；VoxelMorph 只学**局部非刚性残差**，搜索空间大幅缩小。

### 5.2 多尺度（金字塔）训练

模仿 Elastix / KEREN 的 coarse-to-fine：

```python
def register_voxelmorph_multiscale(fixed, moving, scales=(0.25, 0.5, 1.0), epochs_per_scale=400, **kwargs):
    flow_accum = None
    moving_warped = moving.copy()

    for scale in scales:
        h, w = fixed.shape
        sh, sw = int(h * scale), int(w * scale)

        f_s = cv2.resize(fixed, (sw, sh))
        m_s = cv2.resize(moving_warped, (sw, sh))

        # 在当前尺度训练（可将上一尺度的 flow 上采样作为 init，需改网络接口）
        warped_s = register_voxelmorph(f_s, m_s, epochs=epochs_per_scale, **kwargs)

        # 上采样回原分辨率，更新 moving_warped
        moving_warped = cv2.resize(warped_s, (w, h), interpolation=cv2.INTER_LINEAR)

    return moving_warped
```

尺度建议：`0.25 → 0.5 → 1.0`，每级 300–500 epoch，**λ 从大到小**（粗尺度 λ=5，细尺度 λ=1）。

### 5.3 直方图匹配预处理（对齐 Elastix+Histogram 思路）

跨波段强度分布不一致时，先拉齐 histogram 再算 NCC：

```python
from skimage.exposure import match_histograms

moving_matched = match_histograms(moving, fixed, channel_axis=None)
warped = register_voxelmorph(fixed, moving_matched, lamda=1.0, epochs=800)
```

---

## 6. P2：损失函数优化

### 6.1 组合损失（推荐公式）

```
L = α · LNCC(f, warp(m)) 
  + β · (1 - MI_diff(f, warp(m)))   # 或 MIND
  + γ · Grad(flow)
  + δ · ||flow||²                   # 可选：软约束位移幅度而非只惩罚梯度
```

**起步权重**：`α=1, β=0.5, γ=1（λ=1）, δ=0.01`

### 6.2 可微互信息损失（替代 / 补充 NCC）

NCC 对跨波段弱，MI 与 Elastix 一致。可用软直方图近似：

```python
import torch
import torch.nn.functional as F

class SoftMI(torch.nn.Module):
    """可微 MI 近似，用于跨波段配准"""
    def __init__(self, num_bins=32, sigma=0.01):
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.bins = torch.linspace(0, 1, num_bins)

    def _soft_histogram(self, x):
        # x: [B,1,H,W] 归一化到 [0,1]
        x = x.flatten(1)
        bins = self.bins.to(x.device)
        # Gaussian kernel soft binning
        dist = (x.unsqueeze(-1) - bins) / self.sigma
        return F.softmax(-dist ** 2, dim=-1).mean(0)  # [num_bins]

    def forward(self, fixed, warped):
        p_x = self._soft_histogram(fixed)
        p_y = self._soft_histogram(warped)
        # 联合分布可用 outer product 近似（简化版）
        p_xy = torch.outer(p_x, p_y)
        p_xy = p_xy / (p_xy.sum() + 1e-8)
        px = p_xy.sum(1)
        py = p_xy.sum(0)
        mi = (p_xy * (torch.log(p_xy + 1e-8) 
              - torch.log(px.unsqueeze(1) + 1e-8) 
              - torch.log(py.unsqueeze(0) + 1e-8))).sum()
        return -mi  # 最大化 MI → 最小化 -MI
```

在 `register_voxelmorph` 中：

```python
mi_loss = SoftMI(num_bins=32)
loss = loss_ncc + 0.5 * mi_loss(f, warped_m) + lamda * loss_grad
```

### 6.3 区间 / 波段感知损失（针对高光谱）

相邻波段（如 639 vs 650 nm）比远波段更相似，可引入**波长距离权重**：

```python
def band_weighted_loss(fixed, warped, flow, lambda_nm=11):
    """
    lambda_nm: 两波段中心波长差（nm），如 |650-639|=11
    波长差越大，越降低 NCC 权重、提高 MI/边缘损失权重
    """
    loss_ncc = ncc_loss.loss(fixed, warped)
    loss_grad = grad_loss.loss(None, flow)

    # 相邻波段 (<20nm): 信任 NCC
    # 远波段 (>50nm): 更依赖 MI 或梯度结构
    w_ncc = max(0.2, 1.0 - lambda_nm / 100.0)
    w_mi  = 1.0 - w_ncc

    loss_mi = mi_loss(fixed, warped)  # 需实现 6.2
    return w_ncc * loss_ncc + w_mi * loss_mi + lamda * loss_grad
```

**波段区间策略（批处理时）**：

| 波段间隔 | 建议损失 |
|----------|----------|
| ≤ 15 nm | LNCC + 小 λ |
| 15–40 nm | LNCC + MI 混合 |
| > 40 nm | MI / MIND + 边缘损失（参考 `register_elastix_edge`） |

### 6.4 梯度结构损失（NTG 思想）

评价指标 NTG 关注**梯度差异**，可加入训练：

```python
def gradient_loss(img1, img2):
    def grad(x):
        gx = x[:, :, 1:, :] - x[:, :, :-1, :]
        gy = x[:, :, :, 1:] - x[:, :, :, :-1]
        return gx, gy
    g1x, g1y = grad(img1)
    g2x, g2y = grad(img2)
    return (F.l1_loss(g1x, g2x) + F.l1_loss(g1y, g2y))
```

对跨波段纹理对齐有帮助，可与 NCC 加权组合。

### 6.5 位移幅度相关改进

**问题**：仅惩罚 `Grad(flow)` 不等于允许大位移，而是禁止**空间变化剧烈**。

若需要**允许整体大平移**：

```python
# 方案 A：降低 λ（首选）

# 方案 B：改用 L2 位移幅度软约束（只惩罚过大像素位移）
loss_mag = torch.mean(flow ** 2)   # 权重很小，如 0.001

# 方案 C：输出 velocity field 并积分（微分同胚，大位移更稳定）
# flow = model(m, f)
# 需改为多步 scaling-and-squaring，参考官方 VoxelMorph diffeomorphic 版本
```

**方案 C 简述**：网络输出速度场 `v`，通过 `φ = exp(v)` 积分得到位移，大变形更平滑、可逆性更好——适合形变较大的场景，实现成本较高。

### 6.6 启用 GAN 判别器（代码已有，未接入）

`Discriminator` 已在 `registration_networks.py` 定义，可加入对抗损失约束 warped 与 fixed 的**整体分布相似**：

```python
# 伪代码
d_loss = BCE(discriminator(f, warped.detach()), real_label)
g_loss = BCE(discriminator(f, warped), fake_as_real_label)
loss = loss_ncc + lamda * loss_grad + 0.01 * g_loss
```

对跨波段"看起来不像"的情况有一定帮助，但训练不稳定，建议在 P0/P1 完成后再试。

---

## 7. P3：长期提升方向

### 7.1 数据集预训练（正统 VoxelMorph 用法）

1. 从 `data/Test dataset/` 收集所有相邻波段对 `(band_i, band_{i+1})`；
2. 随机划分 train/val；
3. **一个共享 U-Net** 训练 50k–200k iter；
4. 推理时对单对图像 **fine-tune 50–200 epoch** 或直接前向。

这是让 VoxelMorph 超过传统方法的**最可靠路径**。

### 7.2 使用官方 voxelmorph 库

```bash
pip install voxelmorph
```

官方实现含：预训练权重、多尺度、diffeomorphic、更好的 U-Net。可对比本项目轻量实现作为 baseline。

### 7.3 网络结构改进

| 改动 | 说明 |
|------|------|
| 加深通道 16→32→64→64 | 提高表达能力 |
| 补全 Decoder skip | 最后上采样层拼接 L0 特征 |
| 输入差分图 `[m, f, m-f, \|m-f\|]` | 4 通道，突出错位区域 |
| 引入 SpatialTransformer 的 `padding_mode='zeros'` | 边界外推更保守，有时比 `border` 好 |

---

## 8. 推荐实验记录表

每次实验记录如下，便于对比：

| 实验 ID | 预配准 | 损失 | λ | NCC win | epochs | MI_after | NCC_after | NTG_after | 备注 |
|---------|--------|------|---|---------|--------|----------|-----------|-----------|------|
| baseline | 无 | LNCC+Grad | 10 | 25 | 300 | | | | 当前默认 |
| exp-01 | 无 | LNCC+Grad | 1 | 25 | 1000 | | | | 降 λ |
| exp-02 | StackReg | LNCC+Grad | 1 | 9 | 800 | | | | **最推荐** |
| exp-03 | StackReg | LNCC+MI+Grad | 1 | 9 | 800 | | | | 跨波段 |
| exp-04 | 多尺度 | LNCC+Grad | 1→0.5 | 9 | 3×400 | | | | |

---

## 9. 最小改动示例（复制即用）

在 `registration/methods.py` 的 `register_voxelmorph` 中，先做这三处修改，通常即可看到明显提升：

```python
def register_voxelmorph(fixed_image, moving_image, lr=1e-4, epochs=1000, device='cuda', lamda=1.0):
    # ... 前面不变 ...

    ncc_loss = NCC(win=[9, 9])          # 改：窗口 25→9
    grad_loss = Grad(penalty='l2')

    # 改：λ 10→1，epochs 300→1000

    # 可选：在 methods.py 顶部增加 affine 预配准
    from pystackreg import StackReg
    sr = StackReg(StackReg.BILINEAR)
    sr.register(fixed_image, moving_image)
    moving_image = sr.transform(moving_image)

    # ... 训练循环不变 ...
```

---

## 10. 总结

| 问题 | 原因 | 优先解决方案 |
|------|------|--------------|
| 位移太小 | λ=10 平滑过强 | **λ 降到 0.5–1.0** |
| 全局对不齐 | 无初始化 | **StackReg 预配准** |
| 跨波段 NCC 失效 | 强度分布不同 | **直方图匹配 + MI 损失** |
| 收敛不足 | 300 epoch + 随机初始化 | **1000 epoch + LR decay** |
| 单尺度 | 无金字塔 | **0.25/0.5/1.0 多尺度** |
| 方法范式 | 未预训练 | **全数据集预训练 U-Net** |

**务实建议**：在本项目当前架构下，**StackReg 初始化 + 降低 λ + 多尺度 + 增加 epoch** 是最快、最稳的提升组合；若仍不满足，再引入 MI 损失与数据集预训练。

---

## 附录：与 Elastix / StackReg 选型建议

| 场景 | 推荐方法 |
|------|----------|
| 相邻波段（Δλ < 20nm）、快速批量 | StackReg 或 KEREN |
| 跨波段、强度差异大 | Elastix（MI）或 Elastix+Histogram |
| 需要逐像素非刚性形变 | Elastix B-spline 或 **预训练 VoxelMorph** |
| 当前未预训练的 VoxelMorph | 仅作研究对比，**不建议作生产默认** |

---

*文档版本：2026-06-22 · 对应代码路径 `src/python/registration/methods.py`*
