# PIFReg 项目文档

> **PIFReg** = **P**yramid **I**nstance **F**low **Reg**istration（金字塔实例流配准）  
> 供新 Agent / 开发者快速接手本项目深度学习配准模块。

---

## 1. 方法定位

### 1.1 是什么

PIFReg 是一种**逐对、无监督、test-time optimization** 的 2D 图像配准方法，面向**高光谱相邻波段**等场景：

- 每来一对 `(fixed, moving)`，**随机初始化** U-Net，用梯度下降优化该对的位移场；
- **不需要**预训练数据库（无 PIFReg 专用预训练权重时仍可用）；
- 使用**显式图像金字塔**：各尺度在下采样图像上优化，**只传递/组合位移场**，最终对**原分辨率 moving** 做一次 warp。

### 1.2 不是什么

| 对比项 | VoxelMorph 论文 (TMI 2019) | PIFReg |
|--------|---------------------------|--------|
| 训练 | 大数据集预训练共享网络 | 每对图像现场训练 |
| 推理 | 一次前向，无迭代 | 多 epoch + 多尺度迭代 |
| 金字塔 | 无显式图像金字塔（仅 U-Net 内部层次） | 有 `(0.25, 0.5, 1.0)` 等显式尺度 |
| 输出 | 预训练模型 warp | 原图 + 组合位移场 warp 一次 |

旧名 `register_voxelmorph()` 已弃用，调用会触发 `DeprecationWarning`，请用 `register_pifreg()`。

### 1.3 与 Elastix / StackReg 的关系

| 方法 | 速度 | 特点 |
|------|------|------|
| **Elastix** | 快（~100 iter × 4 层金字塔，B 样条低维优化） | MI 度量，工程成熟 |
| **StackReg** | 很快 | 全局仿射/双线性 |
| **PIFReg** | 慢（每尺度可达数千 epoch，U-Net 全图反向传播） | **Dense flow**，跨波段非刚性上限更高 |

推荐组合：**Elastix/StackReg 粗配准 + PIFReg 精修**（尚未默认实现，可作为加速方向）。

---

## 2. 算法流程

```
输入: fixed (H×W), moving (H×W)，值域建议 [0, 1]
    │
    ├─ 预处理（可选）
    │     histogram_match: moving 直方图匹配到 fixed
    │     affine_init: StackReg BILINEAR 全局仿射
    │
    ├─ 保留 moving_original（全分辨率，用于最终 warp）
    │
    └─ 多尺度循环 scale ∈ scales（默认 0.25 → 0.5 → 1.0）
          │
          ├─ f_s, m_s = 下采样 fixed/moving（INTER_AREA，仅用于优化）
          ├─ 若已有粗尺度 flow：上采样 flow → warp m_s（残差配准）
          ├─ 随机初始化 VxmDense，Adam 优化 max_epochs 次
          │     Loss = NCC(f_s, warp(m_s)) + λ · Grad(preint_flow)
          │     支持早停、余弦/阶梯/Plateau 学习率
          ├─ 得到 flow_delta，与 flow_at_scale 组合（compose）
          │
          └─ 循环结束
                flow_full = 上采样 flow 到 (H, W)
                output = warp(moving_original, flow_full)   ← 只 warp 一次，不上采样 warped 图
```

### 2.1 关键设计（勿改坏）

1. **位移场传递，图像不传递**：尺度间只上采样/组合 `flow`，禁止上采样 warped 图像（会导致模糊）。
2. **最终输出**：对预处理后的 `moving_original` 施加 `flow_full`，保证全分辨率质量。
3. **每尺度重新初始化网络**：当前实现每个金字塔层级新建 U-Net（未做 cross-scale weight warm-start）。

---

## 3. 代码结构

```
24_hyper_registration/
├── docs/
│   └── PIFREG.md                    ← 本文档
├── requirements.txt
├── data/                            # 数据集（Test dataset, cut_images_all 等）
├── src/python/
│   ├── registration/
│   │   ├── pif_registration.py      ★ PIFReg 核心实现（register_pifreg）
│   │   ├── methods.py               # Elastix / StackReg / KEREN + 弃用别名
│   │   └── __init__.py
│   ├── voxelmorph/                  # 官方 VxmDense 网络 backbone（vendor，勿与 PIFReg 混淆）
│   │   ├── networks.py              # VxmDense, Unet
│   │   ├── layers.py                # SpatialTransformer, VecInt, ResizeTransform
│   │   ├── losses.py
│   │   └── training.py              # 可选：HSI 波段对预训练（非 PIFReg 必需）
│   ├── losses/registration_losses.py
│   ├── metrics/evaluation.py        # MI, NMI, NCC, NTG
│   ├── experiments/
│   │   ├── run_registration_exp.py  ★ pairwise 实验入口
│   │   └── run_groupwise_registration_exp.py  # Elastix 群组配准（非 PIFReg）
│   └── vendor/pyelastix.py          # Elastix Python 封装
└── All code/                        # 历史代码，一般不要改
```

---

## 4. API 参考

### 4.1 主入口

```python
from src.python.registration import register_pifreg

warped = register_pifreg(
    fixed_image,          # np.ndarray (H, W), float32, [0, 1]
    moving_image,
    device='cuda',
    epochs=3000,          # 多尺度时 = 每个尺度的最大 epoch
    lr=1e-4,
    lamda=0.01,           # 位移场平滑正则（官方 VxmDense 默认量级）
    image_loss='ncc',     # 'ncc' | 'mse'，跨波段推荐 ncc
    int_steps=7,          # diffeomorphic 积分步数
    int_downsize=2,
    # 预处理
    affine_init=True,
    histogram_match=True,
    # 金字塔
    multiscale=True,
    scales=(0.25, 0.5, 1.0),
    # 训练 trick
    early_stop=True,
    patience=100,
    min_delta=1e-5,
    lr_schedule='cosine', # 'cosine' | 'step' | 'plateau' | 'none'
    lr_gamma=0.5,
    lr_min=1e-6,
    save_model_path=None, # 可选：保存最后一尺度模型 .pt
    model_path=None,      # 可选：加载预训练 VxmDense（一般不用）
    fast_mode=False,      # True: 轻量 U-Net + 加速位移场更新
)
```

### 4.2 实验脚本

```powershell
cd F:\PhD\PhD4\24_hyper_registration
conda activate dxtorch

# 需先把项目根目录加入 PYTHONPATH，或从根目录运行：
python src/python/experiments/run_registration_exp.py
```

`run_registration_exp.py` 中：

- `run_pifreg_experiment(fixed_path, moving_path)` — PIFReg
- `run_elastix_experiment(...)` — 对比基线
- `load_images(..., image_size=(512, 512))` — **会 resize**，可能影响清晰度；原图更大时可考虑 `image_size=None`（需自行扩展 loader）

路径解析：`resolve_image_path()` 会在项目根、`data/` 下查找。

---

## 5. 网络与损失

### 5.1 Backbone：VxmDense

位置：`src/python/voxelmorph/networks.py`

- U-Net：`encoder [16,32,32,32]`，`decoder [32,32,32,32,32,16,16]`
- 输出 2 通道位移场 → `VecInt`（7 步 scaling-and-squaring）→ `SpatialTransformer`
- **PIFReg 只把它当可微分形变参数化器**，训练方式与 VoxelMorph 论文不同

### 5.2 损失

```
L = L_sim(fixed, warped) + λ · L_grad(preint_flow)
```

| 项 | 实现 | 说明 |
|----|------|------|
| `L_sim` | `NCC` 或 `MSE` | 跨波段用 NCC |
| `L_grad` | `Grad('l2', loss_mult=int_downsize)` | 平滑正则，λ 默认 0.01 |
| 优化器 | Adam | lr 默认 1e-4 |

早停：跟踪 `best_loss`，保存最优 `flow` 与 `state_dict`；`patience` 轮无改善则停止该尺度。

---

## 6. 默认超参数与调参建议

### 6.1 当前实验默认值（`run_pifreg_experiment`）

```python
epochs=10000
lamda=0.01
affine_init=True
histogram_match=True
multiscale=True
early_stop=True
patience=120
lr_schedule='cosine'
```

### 6.2 调参方向

**`fast_mode=True` 预设（2026-06 新增）**

| 项 | 默认 | fast_mode |
|----|------|-----------|
| U-Net 通道 | `[16,32,32,32]` / ~109k 参数 | `[8,16,16,16]` / ~28k 参数 |
| `int_steps` | 7 | 3 |
| `lr` | 1e-4 | 2e-4 |
| `lamda` | 0.01 | 0.005 |
| `scales` | (0.25, 0.5, 1.0) | (0.5, 1.0) |
| `patience` | 100 | 80 |

目的：更少参数、更高学习率、更低平滑约束 → 每 epoch 更快、位移场更新更激进。需与默认模式对比 MI/NCC/NTG。

| 目标 | 建议 |
|------|------|
| 更高精度 | 增大 `epochs`，增大 `patience`，`multiscale=True`，`affine_init=True` |
| 更快速度 | `fast_mode=True`（推荐先试）；或 `scales=(0.5, 1.0)`；`multiscale=False` + Elastix 预对齐；`int_steps=3`；`epochs=1000, patience=60` |
| 跨波段 | `histogram_match=True`，`image_loss='ncc'` |
| 学习率 | 收敛后期慢 → 试 `lr_schedule='plateau'`；或略增 `lr=2e-4` |
| 位移幅度 | 减小 `lamda`（如 0.005）允许更大形变；过大则不稳定 |

### 6.3 为何比 Elastix 慢（摘要）

1. 优化 **10⁵+ 网络参数** vs B 样条 **~10³ 控制点**
2. 每 epoch：U-Net + 7 步 diffeomorphic 积分 + 全图反向传播
3. 3 个尺度 × 每尺度数千 epoch vs Elastix ~400 轻量迭代
4. 每尺度 **随机重新初始化** 网络

---

## 7. 环境依赖

`requirements.txt`：

```
numpy, torch, opencv-python, scikit-image, pystackreg,
SimpleITK, matplotlib, pandas, openpyxl
```

- **GPU**：强烈推荐；`device='cuda'`
- **Conda 环境**：项目中使用 `dxtorch`
- **Elastix**：通过 `src/python/vendor/pyelastix.py` 调用，需系统安装 Elastix 可执行文件（见 `docs/PYELASTIX_GUIDE.md`）

### 7.1 运行前路径

脚本内已处理：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
```

导入方式：`from src.python.registration import register_pifreg`

---

## 8. 数据说明

高光谱波段图像通常按波长命名：`639.jpeg`, `650.jpeg`, ...

常见目录：

```
data/Test dataset/<session_id>/
data/cut_images_all/<session_id>/
All code/cut_images_all/<session_id>/
```

加载后应 **Min-Max 归一化到 [0, 1]**（`load_images` 已做）。

---

## 9. 评估指标

```python
from src.python.metrics import compute_MI, compute_NMI, compute_NCC, compute_NTG

# 配准后 MI/NCC 应升高，NTG 应降低
```

---

## 10. 已知限制与待改进

| 项 | 状态 |
|----|------|
| 每尺度重新初始化 U-Net | 可改为 cross-scale weight warm-start 加速 |
| 无 Elastix flow 初始化 | 可接 Elastix 位移场作为 PIFReg 初值 |
| `load_images` 强制 512×512 | 可能损失分辨率 |
| `register_voxelmorph` 别名 | 保留兼容，应迁移到 `register_pifreg` |
| `train_voxelmorph.py` | 可选预训练脚本，与 PIFReg 日常用法无关 |
| `VOXELMORPH_OPTIMIZATION_GUIDE.md` | 部分内容已过时（仍写 register_voxelmorph），以本文档为准 |

---

## 11. 快速检查清单（新 Agent）

- [ ] 读 `src/python/registration/pif_registration.py`（唯一核心）
- [ ] 确认输入图像 `[0,1]` float32，形状 `(H,W)` 一致
- [ ] 跑 `run_pifreg_experiment` 与 `run_elastix_experiment` 对比 MI/NCC/NTG
- [ ] 多尺度逻辑：**只传递 flow，最终只 warp 原图一次**
- [ ] 勿把 PIFReg 称为 VoxelMorph 论文方法
- [ ] 改网络结构 → `src/python/voxelmorph/`；改配准流程 → `pif_registration.py`

---

## 12. 最小可运行示例

```python
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r"F:\PhD\PhD4\24_hyper_registration")
sys.path.insert(0, str(ROOT))

from src.python.registration import register_pifreg

def load_band(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

fixed = load_band(ROOT / "data/cut_images_all/2024-06-25_10-12-29-white/650.jpeg")
moving = load_band(ROOT / "data/cut_images_all/2024-06-25_10-12-29-white/639.jpeg")

warped = register_pifreg(
    fixed, moving,
    device="cuda",
    epochs=3000,
    multiscale=True,
    scales=(0.25, 0.5, 1.0),
    early_stop=True,
    patience=120,
)
```

---

## 13. 版本记录

| 日期 | 变更 |
|------|------|
| 2026-06 | 从 `register_voxelmorph` 重命名为 **PIFReg** |
| 2026-06 | 多尺度改为位移场传递，禁止上采样 warped 图 |
| 2026-06 | 加入早停、cosine/plateau/step 学习率调度 |
| 2026-06 | 核心代码独立为 `pif_registration.py` |

---

*文档路径：`docs/PIFREG.md` · 项目根：`24_hyper_registration`*
