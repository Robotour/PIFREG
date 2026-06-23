# PIFReg

**Pyramid Instance Flow Registration** — 金字塔实例流配准

逐对、无监督的 2D 深度学习配准：多尺度位移场金字塔 + U-Net 形变优化，面向高光谱相邻波段等场景。

详细说明见 [docs/PIFREG.md](docs/PIFREG.md)。

## 安装

```bash
pip install -r requirements.txt
```

需要 PyTorch（建议 GPU）与 `pystackreg`（仿射预配准可选）。

## 快速使用

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))  # 项目根目录

import cv2
import numpy as np
from src.python.registration import register_pifreg

fixed = cv2.imread("fixed.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
moving = cv2.imread("moving.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())
moving = (moving - moving.min()) / (moving.max() - moving.min())

warped = register_pifreg(fixed, moving, device="cuda")
```

## 实验脚本

将图像路径指向本地数据（**数据不在本仓库**）：

```bash
python src/python/experiments/run_pifreg_exp.py \
  --fixed path/to/fixed.jpeg \
  --moving path/to/moving.jpeg
```

## 仓库范围

本仓库**仅维护 PIFReg** 相关代码。Elastix / StackReg / KEREN 等传统方法及实验数据请在本地目录使用，已通过 `.gitignore` 排除。

## 许可

研究用途；网络 backbone 参考 VoxelMorph PyTorch 实现（见 `src/python/voxelmorph/`）。
