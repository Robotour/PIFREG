# PIFREG — 高光谱波段配准

面向高光谱图像（HSI）**波段间对齐**的 Python / MATLAB 工具包。核心方法为 **PIFReg**（Pyramid Instance Flow Registration，金字塔实例流配准）及其群组扩展；同时集成 Elastix、StackReg、KEREN 等传统基线，便于对比实验。

> 仓库：https://github.com/Robotour/PIFREG  
> **实验数据不在本仓库**，请在本机 `data/` 或自定义路径放置图像后运行脚本。

---

## 方法概览

| 类别 | 方法 | 入口函数 / 脚本 | 说明 |
|------|------|-----------------|------|
| **逐对** | PIFReg | `register_pifreg()` | 多尺度位移场金字塔 + U-Net test-time 优化 |
| **群组** | StackFlow | `register_pifreg_groupwise_stackflow()` | 联合预测 N−1 个 2D 位移场（Per-Band StackFlow） |
| **群组** | StackFlow 3D | `register_pifreg_groupwise_stackflow3d()` | 3D U-Net 在 (N,H,W) 栈上预测位移 |
| **群组** | Chain | `register_pifreg_chain()` | 相邻波段链式逐对 PIFReg |
| **群组** | Joint | `register_pifreg_groupwise_joint()` | 联合 StackFlow 变体 |
| **传统** | Elastix | `register_elastix()` 等 | B 样条非刚性（需安装 elastix.exe） |
| **传统** | StackReg | `register_stackreg()` | 全局仿射 / 双线性 |
| **传统** | KEREN | `register_keren()` | 金字塔光流（Python + MATLAB） |

详细算法说明见 [docs/PIFREG_GUIDE.md](docs/PIFREG_GUIDE.md)；StackFlow 架构见 [docs/PIFREG_STACKFLOW_ARCHITECTURE.md](docs/PIFREG_STACKFLOW_ARCHITECTURE.md)。

---

## 目录结构

```
PIFREG/
├── README.md
├── requirements.txt
├── docs/                              # 文档
│   ├── PIFREG_GUIDE.md
│   ├── PIFREG_STACKFLOW_ARCHITECTURE.md
│   ├── PYELASTIX_GUIDE.md
│   └── VOXELMORPH_OPTIMIZATION_GUIDE.md
├── PROJECT_DOCUMENTATION.md           # 项目总览（历史文档）
├── src/
│   ├── python/
│   │   ├── registration/            # 配准核心
│   │   │   ├── pif_registration.py          # 逐对 PIFReg
│   │   │   ├── pif_groupwise_stackflow.py   # StackFlow 群组
│   │   │   ├── pif_groupwise_stackflow3d.py
│   │   │   ├── pif_groupwise_chain.py
│   │   │   ├── pif_groupwise_joint.py
│   │   │   └── methods.py                   # Elastix / StackReg / KEREN
│   │   ├── voxelmorph/              # U-Net backbone（VoxelMorph PyTorch 移植）
│   │   ├── networks/                # 网络兼容层
│   │   ├── losses/                  # NCC、梯度损失
│   │   ├── metrics/                 # MI、NCC、NTG 等评价指标
│   │   ├── preprocessing/           # HSI → RGB
│   │   ├── utils/                   # 图像变换工具
│   │   ├── vendor/                  # pyelastix 封装
│   │   └── experiments/             # 实验脚本（见下表）
│   └── matlab/                      # MATLAB 参考实现
│       ├── core/registration/       # keren.m, keren2.m
│       ├── core/preprocessing/      # HSI2RGB.m, pca_function.m
│       └── demos/                   # cutandregistration_demo.m
├── data/          ← 本地数据（gitignore，不上传）
├── models/        ← 本地权重（gitignore）
└── outputs/       ← 实验输出（gitignore）
```

---

## 安装

```bash
git clone git@github.com:Robotour/PIFREG.git
cd PIFREG
pip install -r requirements.txt
```

**依赖说明**

- **PyTorch**：建议 GPU；PIFReg / StackFlow 为 test-time 优化，计算量较大。
- **pystackreg**：PIFReg 可选仿射预配准（`affine_init=True`）。
- **SimpleITK + elastix.exe**：Elastix 方法；见 [docs/PYELASTIX_GUIDE.md](docs/PYELASTIX_GUIDE.md)。

---

## 快速使用

### 逐对 PIFReg

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import cv2
from src.python.registration import register_pifreg

fixed = cv2.imread("fixed.png", cv2.IMREAD_GRAYSCALE).astype("float32")
moving = cv2.imread("moving.png", cv2.IMREAD_GRAYSCALE).astype("float32")
fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())
moving = (moving - moving.min()) / (moving.max() - moving.min())

warped = register_pifreg(fixed, moving, device="cuda")
```

### 群组 StackFlow

```python
import numpy as np
from src.python.registration.pif_groupwise_stackflow import register_pifreg_groupwise_stackflow

# stack: (N, H, W)，N 个波段，值域 [0, 1]
stack = np.stack([band0, band1, ...], axis=0)
warped_stack, flow_stack = register_pifreg_groupwise_stackflow(stack, device="cuda")
```

---

## 实验脚本

| 脚本 | 用途 |
|------|------|
| `run_pifreg_exp.py` | 逐对 PIFReg，指定 `--fixed` / `--moving` |
| `run_pifreg_groupwise_stackflow_exp.py` | Per-Band StackFlow 群组实验 + 自动记录 |
| `run_pifreg_groupwise_stackflow3d_exp.py` | 3D StackFlow 群组实验 |
| `run_pifreg_groupwise_chain_exp.py` | 链式群组 PIFReg |
| `run_pifreg_groupwise_joint_exp.py` | Joint StackFlow 实验 |
| `run_registration_exp.py` | 多方法对比（PIFReg / Elastix / StackReg / KEREN） |
| `run_groupwise_registration_exp.py` | 群组传统方法对比 |
| `compare_pyelastix.py` | pyelastix 封装测试 |
| `train_voxelmorph.py` | VoxelMorph 预训练（可选，非 PIFReg 默认流程） |
| `experiment_recorder.py` | StackFlow 实验记录与可视化 |

示例（路径指向本机数据）：

```bash
python src/python/experiments/run_pifreg_exp.py \
  --fixed path/to/fixed.jpeg \
  --moving path/to/moving.jpeg

python src/python/experiments/run_pifreg_groupwise_stackflow_exp.py
```

---

## 数据说明

本仓库**不包含**高光谱原始数据、配准结果或模型权重。本地常用目录（已在 `.gitignore` 中排除）：

- `data/` — 测试集与切图
- `outputs/` — 实验输出（StackFlow 脚本会自动写入 `outputs/.../runs/`）
- `models/` — 可选预训练权重

克隆仓库后，将图像放到上述目录或修改实验脚本中的路径常量即可。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [PIFREG_GUIDE.md](docs/PIFREG_GUIDE.md) | PIFReg 算法、参数、代码地图 |
| [PIFREG_STACKFLOW_ARCHITECTURE.md](docs/PIFREG_STACKFLOW_ARCHITECTURE.md) | StackFlow 网络与损失 |
| [PYELASTIX_GUIDE.md](docs/PYELASTIX_GUIDE.md) | Elastix Python 封装说明 |
| [VOXELMORPH_OPTIMIZATION_GUIDE.md](docs/VOXELMORPH_OPTIMIZATION_GUIDE.md) | VoxelMorph 分析与优化笔记 |
| [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) | 项目早期总览 |

---

## 许可与致谢

- 研究用途。
- U-Net backbone 参考 [VoxelMorph](https://github.com/voxelmorph/voxelmorph) PyTorch 实现（`src/python/voxelmorph/`）。
- `pyelastix.py` 遵循 MIT 协议（`src/python/vendor/`）。
