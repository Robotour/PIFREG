# 高光谱图像波段间配准项目

## 项目概述

本项目实现了一个完整的高光谱图像（Hyperspectral Image, HSI）波段间配准工具包，支持多种传统配准算法和基于深度学习的配准方法。项目适用于舌诊等医学图像分析场景中多光谱图像的对齐处理。

---

## 项目架构

### 整体目录结构

```
24_hyper_registration/
├── src/
│   ├── python/                    # Python实现（核心代码）
│   │   ├── __init__.py           # 包初始化
│   │   ├── utils/                # 工具模块
│   │   │   ├── __init__.py
│   │   │   └── image_transform.py   # 图像变换函数
│   │   ├── metrics/              # 评价指标模块
│   │   │   ├── __init__.py
│   │   │   └── evaluation.py     # MI、NMI、NCC、NTG等指标
│   │   ├── losses/               # 损失函数模块
│   │   │   ├── __init__.py
│   │   │   └── registration_losses.py  # NCC、梯度损失
│   │   ├── networks/             # 神经网络模块
│   │   │   ├── __init__.py
│   │   │   └── registration_networks.py  # U-Net、SpatialTransformer
│   │   ├── preprocessing/       # 预处理模块
│   │   │   ├── __init__.py
│   │   │   └── hsi_to_rgb.py    # 高光谱转RGB
│   │   ├── registration/        # 配准方法模块
│   │   │   ├── __init__.py
│   │   │   └── methods.py       # VoxelMorph、Elastix、StackReg、KEREN
│   │   └── experiments/          # 实验脚本
│   │       ├── __init__.py
│   │       └── run_registration_exp.py  # 实验演示
│   └── matlab/                   # MATLAB实现
│       ├── core/
│       │   ├── image_transform/  # 图像变换
│       │   │   ├── shift.m
│       │   │   └── shiftandrotate.m
│       │   ├── registration/     # 配准算法
│       │   │   ├── keren.m
│       │   │   └── keren2.m
│       │   └── preprocessing/   # 预处理
│       │       ├── HSI2RGB.m
│       │       └── pca_function.m
│       └── demos/
│           └── cutandregistration_demo.m
├── data/                         # 数据目录（需创建）
│   ├── Test dataset/            # 测试数据集
│   ├── Tongue cut dataset/     # 舌诊数据集
│   └── registration result2/    # 配准结果输出
└── README.md
```

### 模块依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                    用户接口层                                │
│  run_registration_exp.py / cutandregistration_demo.m       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    配准方法层                                │
│  methods.py / keren.m / keren2.m                            │
│  - register_voxelmorph()    - VoxelMorph深度学习配准        │
│  - register_elastix()        - Elastix传统配准              │
│  - register_elastix_edge()   - Elastix+边缘检测             │
│  - register_elastix_histogram() - Elastix+直方图匹配        │
│  - register_stackreg()       - StackReg堆栈配准              │
│  - register_keren()          - KEREN金字塔光流              │
└─────────────────────────────────────────────────────────────┘
          │           │            │            │
          ▼           ▼            ▼            ▼
┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│  网络模块   │ │ 损失模块  │ │工具模块   │ │ 预处理模块   │
│            │ │          │ │          │ │              │
│ U-Net      │ │ NCC Loss │ │ shift()  │ │ hsi_to_rgb() │
│ Spatial-   │ │ Grad Loss│ │ shift_   │ │              │
│ Transformer│ │          │ │ and_rotate│ │              │
│ Discriminator│        │ │ normalize │ │              │
└────────────┘ └──────────┘ └──────────┘ └──────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   评价指标层      │
                    │                  │
                    │ compute_MI()     │
                    │ compute_NMI()    │
                    │ compute_NCC()    │
                    │ compute_NTG()    │
                    └──────────────────┘
```

---

## 各模块功能与核心实现

### 1. 工具模块 (utils/)

**文件**: `image_transform.py`

**功能**: 提供图像基本变换操作

| 函数 | 说明 |
|------|------|
| `shift(im1, x1, y1)` | 图像平移变换，支持亚像素精度插值 |
| `shift_and_rotate(im1, x1, y1, angle)` | 图像平移+旋转变换 |
| `normalize_image(image, method)` | 图像归一化，支持min-max和z-score |
| `denormalize_image(...)` | 图像反归一化 |

### 2. 评价指标模块 (metrics/)

**文件**: `evaluation.py`

**功能**: 提供配准质量评估指标

| 指标 | 说明 | 范围 |
|------|------|------|
| `compute_MI()` | 互信息 (Mutual Information) | ≥0，越大越好 |
| `compute_NMI()` | 归一化互信息 | 1-2，越大越好 |
| `compute_NCC()` | 归一化互相关 | -1到1，1为完全正相关 |
| `compute_NTG()` | 归一化总梯度 | 越小越好 |
| `compute_SSIM()` | 结构相似性指数 | -1到1，1为完全相同 |

### 3. 损失函数模块 (losses/)

**文件**: `registration_losses.py`

**功能**: 提供配准网络训练的损失函数

| 类 | 说明 |
|----|------|
| `NCC` | 局部归一化互相关损失，用于度量图像相似度 |
| `Grad` | 梯度损失（L1或L2），用于位移场平滑正则化 |

### 4. 网络模块 (networks/)

**文件**: `registration_networks.py`

**功能**: 提供配准网络结构

| 类 | 说明 |
|----|------|
| `RegistrationUNet` | U-Net风格配准网络，输入为固定/移动图像拼接，输出2通道位移场 |
| `SpatialTransformer` | 空间变换网络，根据位移场对图像进行双线性插值变形 |
| `Discriminator` | GAN判别器网络，用于GAN基础的配准方法 |

### 5. 预处理模块 (preprocessing/)

**文件**: `hsi_to_rgb.py`

**功能**: 将高光谱图像序列转换为RGB彩色显示

```python
def hsi_to_rgb(cropped_images, spectral_data_path=None):
    # cropped_images: 高光谱图像列表 [band1, band2, ..., bandN]
    # spectral_data_path: 光谱响应曲线Excel文件
    # 返回: RGB彩色图 (H, W, 3)
```

### 6. 配准方法模块 (registration/)

**文件**: `methods.py`

**功能**: 实现多种配准算法

#### 6.1 VoxelMorph（深度学习方法）

```python
def register_voxelmorph(fixed_image, moving_image, lr=1e-4, epochs=300, device='cuda', lamda=10):
    # 基于U-Net的深度学习配准
    # lr: 学习率
    # epochs: 训练轮数
    # lamda: 梯度损失权重
```

#### 6.2 Elastix（传统方法）

```python
def register_elastix(fixed_image, moving_image, epochs=20, spacinginvoxels=20):
def register_elastix_edge(fixed_image, moving_image, epochs=20, spacinginvoxels=20):
def register_elastix_histogram(fixed_image, moving_image, epochs=20, spacinginvoxels=20):
```

#### 6.3 StackReg

```python
def register_stackreg(fixed_image, moving_image, transform_type='bilinear'):
    # transform_type: 'translation', 'rigid', 'scaled_rotation', 'affine', 'bilinear'
```

#### 6.4 KEREN（Lucas-Kanade金字塔光流）

```python
def register_keren(img_list):
    # 多尺度金字塔Lucas-Kanade光流配准
    # 适用于存在平移和旋转的图像对
```

---

## 高光谱图像波段配准算法技术路线

### 配准流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  原始高光谱  │───▶│  图像预处理  │───▶│  配准算法   │───▶│  配准结果   │
│  波段图像    │    │(归一化/裁剪) │    │ (多种方法)  │    │ (对齐的图像) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                │
                                                ▼
                                        ┌─────────────┐
                                        │  质量评估   │
                                        │(MI/NMI/NCC) │
                                        └─────────────┘
```

### 支持的配准方法对比

| 方法 | 类型 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| VoxelMorph | 深度学习 | 端到端、自动学习特征 | 需GPU、训练时间长 | 大变形、非刚性 |
| Elastix | 传统 | 精度高、参数可调 | 计算量大 | 医学图像 |
| StackReg | 传统 | 速度快 | 只能处理刚性/仿射 | 简单变换 |
| KEREN | 传统 | 亚像素精度 | 仅支持平移+旋转 | 刚性对齐 |

---

## 数据流转路径

### 批量图像配准流程

```
输入目录                    处理流程                    输出目录
   │                           │                        │
   ▼                           ▼                        ▼
[Test dataset/] ────▶ [读取所有波段图像] ────▶ [img_cell列表]
                           │
                           ▼
                    [选择参考图像]
                    (通常为中间波段)
                           │
                           ▼
              ┌────────────────────────┐
              │ 依次对每个波段进行配准 │
              │ 参考图像: 已配准的前一波段 │
              └────────────────────────┘
                           │
                           ▼
                    [应用变换参数]
                           │
                           ▼
              [registration result2/]
```

---

## 部署运行步骤

### Python环境配置

**依赖库**:
```
numpy>=1.19.0
torch>=1.7.0
opencv-python>=4.5.0
SimpleITK>=2.0.0
pyelastix>=0.2.0
pystackreg>=0.2.0
scikit-image>=0.18.0
pandas>=1.2.0
matplotlib>=3.3.0
```

**安装命令**:
```bash
pip install numpy torch opencv-python SimpleITK pyelastix pystackreg scikit-image pandas matplotlib
```

### 运行实验

**Python实验**:
```python
from src.python.registration import register_voxelmorph
from src.python.metrics import compute_MI

# 加载图像
import cv2
fixed = cv2.imread('fixed.jpeg', cv2.IMREAD_GRAYSCALE)
moving = cv2.imread('moving.jpeg', cv2.IMREAD_GRAYSCALE)

# 配准
warped = register_voxelmorph(fixed, moving, epochs=300)

# 评估
mi = compute_MI(fixed, warped)
print(f"MI: {mi:.4f}")
```

**MATLAB实验**:
```matlab
% 添加路径
addpath('src/matlab/core/registration');
addpath('src/matlab/core/image_transform');
addpath('src/matlab/core/preprocessing');

% 运行配准
img_list = {img1, img2, img3};  % 图像元胞数组
[delta_est, phi_est] = keren2(img_list);
```

---

## 代码变更记录

### 重构变更摘要

| 变更类型 | 原状态 | 新状态 |
|----------|--------|--------|
| 目录重组 | 散落在`data/`、`python code/`等多处 | 统一到`src/python/`和`src/matlab/` |
| 代码去重 | `shift.py`等在多处重复 | 合并到`utils/image_transform.py` |
| 包结构 | 无Python包结构 | 完整的模块化包结构 |
| 文档 | 代码无注释/文档 | 添加完整docstring |

### 主要变更详情

#### 1. 目录结构重组

**重构前**:
```
24_hyper_registration/
├── data/
│   ├── shift.py
│   ├── shift_and_rotate.py
│   ├── hsi_to_rgb.py
│   ├── registrationfun_all.py
│   └── ...
├── python code/
│   ├── shift.py
│   ├── hsi_to_rgb.py
│   └── ...
└── (根目录散落各种.m文件)
```

**重构后**:
```
src/
├── python/
│   ├── utils/
│   ├── metrics/
│   ├── losses/
│   ├── networks/
│   ├── preprocessing/
│   └── registration/
└── matlab/
    ├── core/
    └── demos/
```

#### 2. 代码合并

| 原文件 | 合并到 | 说明 |
|--------|--------|------|
| `data/shift.py`, `python code/shift.py` | `utils/image_transform.py` | 合并为一个统一的shift函数 |
| `data/shift_and_rotate.py`, `python code/shift_and_rotate.py` | `utils/image_transform.py` | 合并 |
| `data/hsi_to_rgb.py`, `python code/hsi_to_rgb.py` | `preprocessing/hsi_to_rgb.py` | 合并 |
| `data/evaluation_all.py` | `metrics/evaluation.py` | 重命名更清晰 |
| `data/loss_all.py` | `losses/registration_losses.py` | 重命名 |
| `data/net_all.py` | `networks/registration_networks.py` | 重命名 |
| `data/registrationfun_all.py` | `registration/methods.py` | 拆分为独立函数 |
| `data/keren.m`, `keren2.m`, `keren3.m` | `matlab/core/registration/keren.m`, `keren2.m` | 合并版本 |

#### 3. 函数接口规范化

所有函数添加了标准的docstring文档：
```python
def function_name(param1, param2):
    """
    函数功能描述
    
    参数:
        param1: 参数1说明
        param2: 参数2说明
    
    返回:
        返回值说明
    """
```

#### 4. 删除的无关文件

以下文件被识别为与项目无关或测试代码，未包含在重构中：
- `light.py` - IDE流量监控工具（非项目代码）
- `voxelmorph-dev.zip` - 第三方库压缩包（应通过pip安装）
- 各种`test_for_*.py` - 测试脚本

---

## 使用示例

### 示例1: VoxelMorph配准

```python
import cv2
import torch
from src.python.registration import register_voxelmorph
from src.python.metrics import compute_MI, compute_NCC

# 加载图像
fixed = cv2.imread('650.jpeg', cv2.IMREAD_GRAYSCALE).astype('float32')
moving = cv2.imread('639.jpeg', cv2.IMREAD_GRAYSCALE).astype('float32')

# 归一化
fixed = fixed / 255.0
moving = moving / 255.0

# 配准
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warped = register_voxelmorph(fixed, moving, lr=1e-4, epochs=300, device=device)

# 评估
print(f"MI: {compute_MI(fixed, warped):.4f}")
print(f"NCC: {compute_NCC(fixed, warped):.4f}")
```

### 示例2: Elastix配准

```python
from src.python.registration import register_elastix

warped = register_elastix(fixed, moving, epochs=20, spacinginvoxels=20)
```

### 示例3: KEREN配准（MATLAB）

```matlab
% 加载图像
img_list = {};
for i = 1:30
    img_list{i} = imread(sprintf('band%d.jpeg', i));
end

% KEREN配准
[delta_est, phi_est] = keren2(img_list);

% 应用变换
for i = 1:length(img_list)
    registered{i} = shiftandrotate(img_list{i}, delta_est(i,2), delta_est(i,1), phi_est(i));
end
```

---

## 维护说明

### 添加新的配准方法

1. 在`src/python/registration/methods.py`中添加新函数
2. 在`src/python/registration/__init__.py`中导出
3. 在`src/python/__init__.py`中添加到顶层导出

### 修改评价指标

1. 在`src/python/metrics/evaluation.py`中修改或添加函数
2. 更新相关文档

---

*文档版本: 1.0.0*
*最后更新: 2024-06-22*
