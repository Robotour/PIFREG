"""
标准深度学习实验记录工具。

每次实验在 runs/<timestamp>_<exp_name>/ 下保存：
  config.json          — 数据路径、超参数、金字塔等完整配置
  architecture.txt     — 网络结构说明
  timing.json          — 耗时
  metrics/             — 配准前后指标及对比图
  images/              — RGB 重建对比
  bands/before|after/  — 各波长灰度图（原图强度，非归一化）
  flows/               — 位移场 .npy 及可视化图
  summary.md           — 人类可读实验摘要
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from src.python.voxelmorph.config import compact_unet_features, default_unet_features


def create_run_dir(
    base_dir: Path,
    exp_name: str = "run",
    timestamp: Optional[str] = None,
) -> Path:
    """创建带时间戳的实验目录 runs/<YYYYMMDD_HHMMSS>_<exp_name>/"""
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in exp_name.strip()) or "run"
    run_dir = Path(base_dir) / "runs" / f"{ts}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def describe_stackflow_architecture(
    image_size: Tuple[int, int],
    num_bands: int,
    anchor_band_idx: int,
    nb_unet_features: Optional[List] = None,
    int_steps: int = 3,
    int_downsize: int = 2,
    fast_mode: bool = True,
    feature_mode: str = "mean_anchor",
    spectral_enc_channels: int = 4,
    spectral_enc_kernel: int = 3,
) -> str:
    """生成 PerBandStackFlowNet 架构文字说明。"""
    h, w = image_size
    num_moving = num_bands - 1
    enc_nf, dec_nf = nb_unet_features or (
        compact_unet_features() if fast_mode else default_unet_features()
    )
    if feature_mode == "spectral_encoder":
        infeats = spectral_enc_channels + 1
        input_lines = [
            f"Input features ({infeats} channels) — spectral_encoder mode:",
            f"  - SpectralEncoder1D: stack (1, {num_bands}, H, W)",
            f"      Conv1d({num_bands}→{max(spectral_enc_channels * 2, 8)}, k={spectral_enc_kernel}) + ReLU",
            f"      Conv1d(...→{spectral_enc_channels}, k={spectral_enc_kernel}) + ReLU",
            f"      → (1, {spectral_enc_channels}, H, W)",
            f"  - anchor band {anchor_band_idx}  (1, 1, H, W)",
        ]
    else:
        infeats = 2
        input_lines = [
            "Input features (2 channels) — mean_anchor mode:",
            "  - stack mean  (1, 1, H, W)",
            f"  - anchor band {anchor_band_idx}  (1, 1, H, W)",
        ]
    lines = [
        "PIFReg StackFlow — PerBandStackFlowNet",
        "=" * 50,
        "",
        "Backbone: VoxelMorph 2D U-Net (src/python/voxelmorph/networks.py)",
        "",
        *input_lines,
        "",
        f"U-Net inshape=({h}, {w}), infeats={infeats}",
        f"  encoder features: {enc_nf}",
        f"  decoder features: {dec_nf}",
        "",
        f"Output head: Conv2d(final_nf → {num_moving * 2}, k=3)",
        f"  → reshape → flow_stack (1, {num_moving}, 2, H, W)",
        "",
        "Post-processing (per flow field):",
        f"  ResizeTransform down ×{int_downsize}",
        f"  VecInt integration steps = {int_steps}",
        f"  ResizeTransform up ×{int_downsize}",
        "",
        "Warp: SpatialTransformer per moving band (anchor fixed)",
        "",
        "Loss:",
        "  sequential_pairwise_ncc_mean + lamda * per_flow_grad_l2",
        f"  ({num_bands - 1} adjacent NCC pairs, {num_moving} flow smooth terms)",
    ]
    return "\n".join(lines)


def save_architecture(run_dir: Path, text: str) -> Path:
    path = run_dir / "architecture.txt"
    path.write_text(text, encoding="utf-8")
    return path


def save_config(run_dir: Path, config: Dict[str, Any]) -> Path:
    return _save_and_return(run_dir / "config.json", config)


def _save_and_return(path: Path, data: Any) -> Path:
    save_json(path, data)
    return path


def save_timing(run_dir: Path, elapsed_seconds: float, extra: Optional[Dict] = None) -> Path:
    payload = {"elapsed_seconds": elapsed_seconds, "elapsed_minutes": elapsed_seconds / 60.0}
    if extra:
        payload.update(extra)
    return _save_and_return(run_dir / "timing.json", payload)


def save_metrics_bundle(
    run_dir: Path,
    metrics_before: Dict,
    metrics_after: Dict,
    metrics_summary: Dict,
) -> Dict[str, Path]:
    """保存指标 JSON、逐波段 CSV 及对比柱状图。"""
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "before": metrics_dir / "before.json",
        "after": metrics_dir / "after.json",
        "summary": metrics_dir / "summary.json",
    }
    save_json(paths["before"], metrics_before)
    save_json(paths["after"], metrics_after)
    save_json(paths["summary"], metrics_summary)

    csv_path = metrics_dir / "per_band_comparison.csv"
    _save_per_band_csv(csv_path, metrics_before, metrics_after)
    paths["per_band_csv"] = csv_path

    chart_path = metrics_dir / "comparison_bar.png"
    _save_metrics_bar_chart(chart_path, metrics_summary)
    paths["comparison_chart"] = chart_path

    return paths


def _save_per_band_csv(
    path: Path,
    metrics_before: Dict,
    metrics_after: Dict,
) -> None:
    keys = ("MI", "NMI", "NCC", "NTG")
    before_map = {r["band_index"]: r for r in metrics_before["per_band"]}
    after_map = {r["band_index"]: r for r in metrics_after["per_band"]}
    band_indices = sorted(before_map.keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["band_index"]
        for k in keys:
            header.extend([f"{k}_before", f"{k}_after", f"{k}_delta"])
        writer.writerow(header)
        for idx in band_indices:
            row = [idx]
            for k in keys:
                b = before_map[idx][k]
                a = after_map[idx][k]
                row.extend([f"{b:.6f}", f"{a:.6f}", f"{a - b:.6f}"])
            writer.writerow(row)


def _save_metrics_bar_chart(path: Path, metrics_summary: Dict) -> None:
    keys = list(metrics_summary.keys())
    before = [metrics_summary[k]["before"] for k in keys]
    after = [metrics_summary[k]["after"] for k in keys]

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, before, width, label="Before", color="#4C72B0")
    ax.bar(x + width / 2, after, width, label="After", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Score")
    ax.set_title("Registration Metrics (mean over bands)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_rgb_outputs(
    run_dir: Path,
    rgb_before: np.ndarray,
    rgb_after: np.ndarray,
    title_after: str = "RGB After Registration",
) -> Dict[str, Path]:
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    p_before = images_dir / "rgb_before.png"
    p_after = images_dir / "rgb_after.png"
    p_compare = images_dir / "rgb_compare.png"

    cv2.imwrite(str(p_before), rgb_before)
    cv2.imwrite(str(p_after), rgb_after)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb_before)
    axes[0].set_title("RGB Before Registration")
    axes[0].axis("off")
    axes[1].imshow(rgb_after)
    axes[1].set_title(title_after)
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(p_compare, dpi=150, bbox_inches="tight")
    plt.close()

    return {"before": p_before, "after": p_after, "compare": p_compare}


def _band_to_uint8(band: np.ndarray) -> np.ndarray:
    """原图强度 [0,255] 或归一化 [0,1] 均可正确写出 uint8。"""
    arr = np.asarray(band, dtype=np.float32)
    if arr.max() <= 1.0 and arr.min() >= 0.0:
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def flow_to_color_image(flow_xy: np.ndarray) -> np.ndarray:
    """位移场 (2, H, W) → RGB 彩色编码图 (H, W, 3) uint8。"""
    fx = flow_xy[0].astype(np.float32)
    fy = flow_xy[1].astype(np.float32)
    rad = np.sqrt(fx * fx + fy * fy)
    max_rad = float(np.max(rad))
    if max_rad > 0:
        rad_norm = np.clip(rad / max_rad, 0, 1)
    else:
        rad_norm = np.zeros_like(rad)
    ang = np.arctan2(fy, fx)
    h = (ang + np.pi) / (2 * np.pi)
    hsv = np.stack([h, np.ones_like(h), rad_norm], axis=-1)
    rgb = (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)
    return rgb


def flow_magnitude_image(flow_xy: np.ndarray) -> np.ndarray:
    """位移场 (2, H, W) → 幅值灰度图 uint8。"""
    mag = np.sqrt(flow_xy[0] ** 2 + flow_xy[1] ** 2).astype(np.float32)
    max_mag = float(np.max(mag))
    if max_mag > 0:
        mag = mag / max_mag
    return (np.clip(mag, 0, 1) * 255).astype(np.uint8)


def save_flow_fields(
    run_dir: Path,
    flow_stack: np.ndarray,
    band_files: Sequence[Path],
    moving_band_indices: Sequence[int],
    anchor_band_idx: int,
) -> Dict[str, Path]:
    """
    保存 N-1 个位移场：stack.npy、彩色编码图、幅值图。

    flow_stack: (M, 2, H, W)，moving_band_indices[j] 对应第 j 个位移场所属波段。
    """
    flows_dir = run_dir / "flows"
    color_dir = flows_dir / "color"
    magnitude_dir = flows_dir / "magnitude"
    color_dir.mkdir(parents=True, exist_ok=True)
    magnitude_dir.mkdir(parents=True, exist_ok=True)

    npy_path = flows_dir / "flow_stack.npy"
    np.save(npy_path, flow_stack.astype(np.float32))

    paths: Dict[str, Path] = {
        "npy": npy_path,
        "color_dir": color_dir,
        "magnitude_dir": magnitude_dir,
    }

    for flow_idx, band_idx in enumerate(moving_band_indices):
        wavelength = band_files[band_idx].stem
        flow_xy = flow_stack[flow_idx]
        color_path = color_dir / f"{wavelength}_flow.png"
        mag_path = magnitude_dir / f"{wavelength}_magnitude.png"
        cv2.imwrite(str(color_path), cv2.cvtColor(flow_to_color_image(flow_xy), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mag_path), flow_magnitude_image(flow_xy))
        paths[f"flow_color_{wavelength}"] = color_path
        paths[f"flow_mag_{wavelength}"] = mag_path

    meta = {
        "anchor_band_idx": anchor_band_idx,
        "num_flow_fields": int(flow_stack.shape[0]),
        "flow_stack_shape": list(flow_stack.shape),
        "moving_band_indices": list(moving_band_indices),
        "wavelength_nm": [band_files[i].stem for i in moving_band_indices],
    }
    meta_path = flows_dir / "flow_meta.json"
    save_json(meta_path, meta)
    paths["meta"] = meta_path
    return paths


def save_band_stacks(
    run_dir: Path,
    bands_before: Sequence[np.ndarray],
    bands_after: Sequence[np.ndarray],
    band_files: Sequence[Path],
    save_before: bool = True,
) -> Dict[str, Path]:
    """保存配准前后各波长灰度图（保留原图像素强度）。"""
    bands_dir = run_dir / "bands"
    after_dir = bands_dir / "after"
    after_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {"after": after_dir}
    for path, band in zip(band_files, bands_after):
        out = after_dir / path.name
        cv2.imwrite(str(out), _band_to_uint8(band))

    if save_before:
        before_dir = bands_dir / "before"
        before_dir.mkdir(parents=True, exist_ok=True)
        paths["before"] = before_dir
        for path, band in zip(band_files, bands_before):
            out = before_dir / path.name
            cv2.imwrite(str(out), _band_to_uint8(band))

    return paths


def write_summary_md(
    run_dir: Path,
    config: Dict[str, Any],
    metrics_summary: Dict,
    elapsed_seconds: float,
    output_paths: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    extra_sections: Optional[List[str]] = None,
) -> Path:
    """生成人类可读的实验摘要 Markdown。"""
    exp_label = config.get("experiment", "registration")
    lines = [
        f"# {title or exp_label} — Experiment Summary",
        "",
        f"- **Run directory**: `{run_dir.name}`",
        f"- **Method**: `{exp_label}`",
        f"- **Exp name**: `{config.get('exp_name', 'run')}`",
        f"- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Elapsed**: {elapsed_seconds:.1f}s ({elapsed_seconds / 60:.1f} min)",
        "",
        "## Data",
        "",
    ]
    if "stack_dir" in config:
        lines.extend([
            f"- Stack: `{config.get('stack_dir', 'N/A')}`",
            f"- Bands: {config.get('num_bands', 'N/A')}",
            f"- Image size: {config.get('image_size', 'N/A')}",
        ])
    if "fixed_image" in config:
        lines.append(f"- Fixed: `{config['fixed_image']}`")
    if "moving_image" in config:
        lines.append(f"- Moving: `{config['moving_image']}`")
    if config.get("anchor_band") is not None:
        lines.append(f"- Anchor band: {config['anchor_band']}")
    if config.get("eval_ref_band") is not None:
        lines.append(f"- Eval ref band: {config['eval_ref_band']}")

    reg = config.get("registration", {})
    if reg:
        lines.extend(["", "## Hyperparameters", ""])
        for key in sorted(reg.keys()):
            lines.append(f"- `{key}`: {reg[key]}")

    if extra_sections:
        lines.extend(extra_sections)

    lines.extend([
        "",
        "## Mean Metrics",
        "",
        "| Metric | Before | After | Delta |",
        "|--------|--------|-------|-------|",
    ])
    for key, vals in metrics_summary.items():
        lines.append(
            f"| {key} | {vals['before']:.4f} | {vals['after']:.4f} | {vals['delta']:+.4f} |"
        )

    if output_paths:
        lines.extend(["", "## Output Files", ""])
        for label, rel in sorted(output_paths.items()):
            lines.append(f"- {label}: `{rel}`")

    path = run_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def describe_joint_architecture(image_size, num_bands, ref_band_idx=None, fast_mode=True) -> str:
    h, w = image_size
    enc_nf, dec_nf = compact_unet_features() if fast_mode else default_unet_features()
    ref = ref_band_idx if ref_band_idx is not None else "auto (middle)"
    return "\n".join([
        "PIFReg Joint Groupwise — Shared Flow",
        "=" * 50,
        "",
        "Backbone: VoxelMorph 2D U-Net",
        f"Input: stack variance (1,1,H,W) + ref band {ref} (1,1,H,W) → 2 channels",
        f"U-Net inshape=({h}, {w})",
        f"  encoder: {enc_nf}, decoder: {dec_nf}",
        "Output: single shared 2D flow (1, 2, H, W) applied to ALL bands",
        "Loss: stack variance minimization + flow smoothness",
    ])


def describe_chain_architecture(num_bands, descending=True) -> str:
    direction = "high→low wavelength" if descending else "low→high wavelength"
    return "\n".join([
        "PIFReg Chain Groupwise",
        "=" * 50,
        "",
        f"Bands: {num_bands}",
        f"Chain direction: {direction}",
        "Method: N-1 independent pairwise PIFReg steps",
        "Each step: VoxelMorph U-Net pairwise registration",
        "Loss per step: NCC + smoothness (test-time optimization)",
        "Primary metric: mean adjacent-band NCC along chain",
    ])


def describe_cascade_architecture(
    num_bands,
    keyframe_interval=5,
    anchor_band_idx=-1,
    refine_pyramid=(128, 256, 512),
    refine_epochs=(300, 500, 800),
    feature_mode="mean_anchor",
) -> str:
    return "\n".join([
        "PIFReg Cascade — Keyframe Scaffold + StackFlow Refine",
        "=" * 50,
        "",
        f"Bands: {num_bands}, anchor index: {anchor_band_idx}",
        "",
        "Stage 1 — Keyframe scaffold",
        f"  Keyframe interval: every {keyframe_interval} bands (+ endpoints + anchor)",
        f"  PIFReg calls: ~{max(1, num_bands // keyframe_interval)} (not {num_bands - 1})",
        "  Flow interpolation: linear along band index between keyframes",
        "",
        "Stage 2 — StackFlow residual refine",
        f"  Warm-start: init_flow_stack from Stage 1",
        f"  Pyramid: {list(refine_pyramid)}",
        f"  Epochs per level: {list(refine_epochs)}",
        f"  Feature mode: {feature_mode}",
        "  Loss: sequential pairwise NCC mean + per-flow smoothness",
        "",
        "Goal: ~Chain accuracy, faster than full chain, global drift correction",
    ])


def describe_stackflow3d_architecture(image_size, num_bands, anchor_band_idx=0, fast_mode=True) -> str:
    h, w = image_size
    return "\n".join([
        "PIFReg StackFlow3D — Scheme A",
        "=" * 50,
        "",
        f"Input: full stack as 3D volume (1, 1, {num_bands}, {h}, {w})",
        "Backbone: 3D U-Net (VoxelMorph-style)",
        f"Output: per-band 2D flows (1, {num_bands}, 2, H, W), anchor={anchor_band_idx} fixed",
        "Warp: 2D SpatialTransformer per band",
        "Loss: sequential pairwise NCC mean + per-flow smoothness",
    ])


def describe_pairwise_architecture(image_size, fast_mode=False, multiscale=True) -> str:
    h, w = image_size
    enc_nf, dec_nf = compact_unet_features() if fast_mode else default_unet_features()
    return "\n".join([
        "PIFReg Pairwise Registration",
        "=" * 50,
        "",
        "Backbone: VxmDense (VoxelMorph 2D U-Net)",
        f"Input: fixed + moving concat → (1, 2, {h}, {w})",
        f"  encoder: {enc_nf}, decoder: {dec_nf}",
        f"Multiscale pyramid: {multiscale}",
        "Affine init: StackReg + optional histogram matching",
        "Loss: NCC + flow smoothness (test-time optimization)",
    ])


def record_groupwise_experiment(
    run_dir: Path,
    config: Dict[str, Any],
    architecture_text: str,
    bands_raw_before: Sequence[np.ndarray],
    bands_raw_after: Sequence[np.ndarray],
    band_files: Sequence[Path],
    rgb_before: np.ndarray,
    rgb_after: np.ndarray,
    metrics_before: Dict,
    metrics_after: Dict,
    metrics_summary: Dict,
    elapsed_seconds: float,
    flow_stack: Optional[np.ndarray] = None,
    moving_band_indices: Optional[Sequence[int]] = None,
    anchor_band_idx: int = 0,
    save_before_bands: bool = True,
    rgb_title_after: str = "RGB After Registration",
    summary_extra_sections: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    群组配准实验标准记录（所有 groupwise 脚本共用）。

    bands_raw_* : 原图强度；配准在归一化空间优化，保存/RGB 应对原图施加位移场后的结果。
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config(run_dir, config)
    save_architecture(run_dir, architecture_text)
    save_timing(run_dir, elapsed_seconds)

    metric_paths = save_metrics_bundle(run_dir, metrics_before, metrics_after, metrics_summary)
    image_paths = save_rgb_outputs(run_dir, rgb_before, rgb_after, title_after=rgb_title_after)
    band_paths = save_band_stacks(
        run_dir, bands_raw_before, bands_raw_after, band_files, save_before=save_before_bands
    )

    rel = lambda p: str(p.relative_to(run_dir))  # noqa: E731
    output_index = {
        "config": rel(run_dir / "config.json"),
        "architecture": rel(run_dir / "architecture.txt"),
        "timing": rel(run_dir / "timing.json"),
        "metrics_before": rel(metric_paths["before"]),
        "metrics_after": rel(metric_paths["after"]),
        "metrics_summary": rel(metric_paths["summary"]),
        "metrics_csv": rel(metric_paths["per_band_csv"]),
        "metrics_chart": rel(metric_paths["comparison_chart"]),
        "rgb_before": rel(image_paths["before"]),
        "rgb_after": rel(image_paths["after"]),
        "rgb_compare": rel(image_paths["compare"]),
        "bands_after": rel(band_paths["after"]),
    }
    if save_before_bands and "before" in band_paths:
        output_index["bands_before"] = rel(band_paths["before"])

    if flow_stack is not None and flow_stack.size > 0 and moving_band_indices is not None:
        flow_paths = save_flow_fields(
            run_dir, flow_stack, band_files, moving_band_indices, anchor_band_idx
        )
        output_index["flows_npy"] = rel(flow_paths["npy"])
        output_index["flows_meta"] = rel(flow_paths["meta"])
        output_index["flows_color_dir"] = rel(flow_paths["color_dir"])
        output_index["flows_magnitude_dir"] = rel(flow_paths["magnitude_dir"])

    summary_path = write_summary_md(
        run_dir, config, metrics_summary, elapsed_seconds, output_index,
        extra_sections=summary_extra_sections,
    )
    output_index["summary_md"] = rel(summary_path)

    manifest = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "experiment": config.get("experiment"),
        "exp_name": config.get("exp_name"),
        "metrics_summary": metrics_summary,
        "outputs": output_index,
    }
    save_json(run_dir / "manifest.json", manifest)
    return manifest


def record_pairwise_experiment(
    run_dir: Path,
    config: Dict[str, Any],
    architecture_text: str,
    fixed_raw: np.ndarray,
    moving_raw: np.ndarray,
    warped_raw: np.ndarray,
    metrics_before: Dict,
    metrics_after: Dict,
    metrics_summary: Dict,
    elapsed_seconds: float,
    flow: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Pairwise PIFReg 实验记录。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config(run_dir, config)
    save_architecture(run_dir, architecture_text)
    save_timing(run_dir, elapsed_seconds)
    save_metrics_bundle(run_dir, metrics_before, metrics_after, metrics_summary)

    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(images_dir / "fixed.png"), _band_to_uint8(fixed_raw))
    cv2.imwrite(str(images_dir / "moving.png"), _band_to_uint8(moving_raw))
    cv2.imwrite(str(images_dir / "warped.png"), _band_to_uint8(warped_raw))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, img, title in zip(
        axes,
        [fixed_raw, moving_raw, warped_raw],
        ["Fixed", "Moving", "Warped"],
    ):
        ax.imshow(_band_to_uint8(img), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(images_dir / "compare.png", dpi=150, bbox_inches="tight")
    plt.close()

    rel = lambda p: str(p.relative_to(run_dir))  # noqa: E731
    output_index = {
        "config": rel(run_dir / "config.json"),
        "architecture": rel(run_dir / "architecture.txt"),
        "timing": rel(run_dir / "timing.json"),
        "metrics_summary": rel(run_dir / "metrics" / "summary.json"),
        "fixed": rel(images_dir / "fixed.png"),
        "moving": rel(images_dir / "moving.png"),
        "warped": rel(images_dir / "warped.png"),
        "compare": rel(images_dir / "compare.png"),
    }
    if flow is not None and flow.size > 0:
        flows_dir = run_dir / "flows"
        flows_dir.mkdir(parents=True, exist_ok=True)
        np.save(flows_dir / "flow.npy", flow.astype(np.float32))
        cv2.imwrite(
            str(flows_dir / "flow_color.png"),
            cv2.cvtColor(flow_to_color_image(flow), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(flows_dir / "flow_magnitude.png"), flow_magnitude_image(flow))
        output_index["flow_npy"] = rel(flows_dir / "flow.npy")

    write_summary_md(run_dir, config, metrics_summary, elapsed_seconds, output_index)
    output_index["summary_md"] = "summary.md"
    manifest = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "experiment": config.get("experiment"),
        "exp_name": config.get("exp_name"),
        "metrics_summary": metrics_summary,
        "outputs": output_index,
    }
    save_json(run_dir / "manifest.json", manifest)
    return manifest


# 向后兼容
record_stackflow_experiment = record_groupwise_experiment
