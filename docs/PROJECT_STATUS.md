# PIFREG 项目现状与下一步计划

> 生成日期：2026-07-20  
> 适用场景：离开项目约一周后快速恢复上下文

---

## 1. 项目是什么

**PIFREG**（Pyramid Instance Flow Registration）是一个面向**高光谱图像（HSI）波段间对齐**的 Python / MATLAB 工具包，主要应用场景为舌诊等多光谱医学图像分析。

- **仓库**：`main` 分支，与 `origin/main` 同步
- **远程**：https://github.com/Robotour/PIFREG
- **核心思路**：test-time optimization——每对/每组波段现场用 U-Net 优化位移场，无需预训练权重

---

## 2. 当前代码能力一览

### 2.1 配准方法（`src/python/registration/`）

| 类别 | 方法 | 函数 | 状态 |
|------|------|------|------|
| 逐对 | **PIFReg** | `register_pifreg()` | 成熟，有完整实验记录 |
| 群组 | **StackFlow** | `register_pifreg_groupwise_stackflow()` | 有历史实验结果 |
| 群组 | **StackFlow 3D** | `register_pifreg_groupwise_stackflow3d()` | 代码完成，epochs 偏少时效果一般 |
| 群组 | **Chain** | `register_pifreg_chain()` | 近期增强（调度/方向参数） |
| 群组 | **Joint** | `register_pifreg_groupwise_joint()` | 有历史结果 |
| 群组 | **Sliding Window** | `register_pifreg_groupwise_sliding_window()` | 代码已合入，尚未跑正式实验 |
| 群组 | **Spatial Window** | `register_pifreg_groupwise_spatial_window()` | 代码已合入，尚未跑正式实验 |
| 传统 | Elastix / StackReg / KEREN | `methods.py` | 基线可用 |

### 2.2 实验脚本（`src/python/experiments/`）

| 脚本 | 用途 |
|------|------|
| `run_registration_exp.py` | 逐对 PIFReg / 传统方法对比 + 误差热力图 |
| `run_pifreg_groupwise_stackflow_exp.py` | StackFlow 群组实验 |
| `run_pifreg_groupwise_stackflow3d_exp.py` | 3D StackFlow 群组实验 |
| `run_pifreg_groupwise_chain_exp.py` | 链式群组实验 |
| `run_pifreg_groupwise_joint_exp.py` | Joint 群组实验 |
| `run_pifreg_groupwise_sliding_window_exp.py` | Sliding Window 群组实验 |
| `run_pifreg_groupwise_spatial_window_exp.py` | Spatial Window 群组实验 |
| `compare_runs.py` | 扫描 outputs 生成对比表 |

### 2.3 未提交的诊断工具（4 个未 git add）

| 文件 | 功能 | 最后运行 |
|------|------|----------|
| `pairwise_heatmap.py` + `run_pairwise_rmse_heatmap.py` | 相邻波段差异热力图 | 2026-07-06 |
| `band_spectrum.py` + `run_band_spectrum.py` | 各波段 2D FFT 频谱 | 2026-07-07 |

---

## 3. 数据与输出

默认样本：`data/cut_images_all/2024-06-25_10-12-29-white/`（30 波段，404-695 nm）

### 群组配准历史结果（512x512）

| 方法 | NCC 前->后 | 耗时 | 备注 |
|------|-------------|------|------|
| Chain | 0.897 -> 0.926 | ~117 min | 弱对 597-590 |
| StackFlow | 0.897 -> 0.920 | ~23 min | 锚点 404 nm |
| Joint | 0.897 -> 0.942 | ~7.5 min | 256x256 |
| StackFlow3D | 0.897 -> 0.907 | ~115 min | epochs 偏少 |

### 逐对 PIFReg（2026-07-02，639 vs 650 nm）

最佳 run：NCC 0.990 -> 0.997，耗时 ~4.6 min。

---

## 4. Git 状态

- 分支 `main`，与 origin 同步
- 最新提交：2026-07-02（b41e1c3）
- 4 个未跟踪的诊断脚本

---

## 5. 建议的下一步

**今天**：查看 outputs 下热力图/频谱结果；提交诊断脚本；验证 CUDA。

**本周**：跑 Sliding Window / Spatial Window 实验；用 compare_runs.py 汇总。

**后续**：分析 597-590 弱对；StackFlow3D 重跑；配准后热力图对比。

---

## 6. 常用命令

```bash
python src/python/experiments/run_pairwise_rmse_heatmap.py
python src/python/experiments/run_band_spectrum.py
python src/python/experiments/run_pifreg_groupwise_sliding_window_exp.py --exp-name sliding_v1
python src/python/experiments/compare_runs.py
```

---

*文档生成日期：2026-07-20*
