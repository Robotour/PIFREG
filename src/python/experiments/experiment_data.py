"""群组配准实验共用：数据加载、指标计算。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.python.preprocessing.band_preprocess import histogram_equalize_band
from src.python.metrics import compute_MI, compute_NMI, compute_NCC, compute_NTG


def resolve_path(path, project_root: Path, extra_bases: Optional[Sequence[Path]] = None):
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    bases = [project_root] + list(extra_bases or [])
    for base in bases:
        resolved = base / candidate
        if resolved.exists():
            return resolved.resolve()
    raise FileNotFoundError(f"Path not found: {path}")


def resolve_image_path(path, project_root: Path, data_dir: Path):
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    for base in (project_root, data_dir):
        resolved = base / candidate
        if resolved.is_file():
            return resolved.resolve()
    raise FileNotFoundError(f"Image not found: {path}")


def sort_band_files(folder: Path) -> List[Path]:
    files = list(Path(folder).glob("*.jpeg")) + list(Path(folder).glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No jpeg images found in {folder}")

    def band_key(path):
        try:
            return int(path.stem)
        except ValueError:
            return path.stem

    return sorted(files, key=band_key)


def load_hsi_stack(
    folder,
    image_size: Optional[Tuple[int, int]] = (512, 512),
    return_wavelengths: bool = False,
):
    """
    加载高光谱栈。

    返回:
        bands_prep: 逐波段直方图均衡（配准优化输入）
        bands_raw: resize 后原图灰度强度（位移/变换作用对象 + 指标/RGB）
        band_files: 文件路径列表
        [wavelengths]: 可选波长 stem 列表
    """
    band_files = sort_band_files(Path(folder))
    bands_prep, bands_raw, wavelengths = [], [], []
    for path in band_files:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = img.astype(np.float32)
        if image_size is not None:
            img = cv2.resize(img, image_size)
        bands_raw.append(img.copy())
        bands_prep.append(histogram_equalize_band(img))
        wavelengths.append(path.stem)
    if return_wavelengths:
        return bands_prep, bands_raw, band_files, wavelengths
    return bands_prep, bands_raw, band_files


def load_pair_images(
    fixed_path,
    moving_path,
    project_root: Path,
    data_dir: Path,
    image_size: Optional[Tuple[int, int]] = (512, 512),
):
    """加载 pairwise 图像：归一化版 + 原图强度。"""
    fixed = cv2.imread(str(resolve_image_path(fixed_path, project_root, data_dir)), cv2.IMREAD_GRAYSCALE)
    moving = cv2.imread(str(resolve_image_path(moving_path, project_root, data_dir)), cv2.IMREAD_GRAYSCALE)
    if fixed is None or moving is None:
        raise ValueError("Failed to read input images")
    fixed = fixed.astype(np.float32)
    moving = moving.astype(np.float32)
    if image_size is not None:
        fixed = cv2.resize(fixed, image_size)
        moving = cv2.resize(moving, image_size)
    fixed_raw, moving_raw = fixed.copy(), moving.copy()
    fixed = histogram_equalize_band(fixed)
    moving = histogram_equalize_band(moving)
    return fixed, moving, fixed_raw, moving_raw


def evaluate_stack(bands: Sequence[np.ndarray], ref_idx: int):
    ref = bands[ref_idx]
    metrics = {"ref_band_index": ref_idx, "per_band": [], "mean": {}}
    keys = ("MI", "NMI", "NCC", "NTG")
    sums = {k: 0.0 for k in keys}
    for i, band in enumerate(bands):
        if i == ref_idx:
            row = {
                "band_index": i,
                "MI": float(compute_MI(ref, ref)),
                "NMI": float(compute_NMI(ref, ref)),
                "NCC": float(compute_NCC(ref, ref)),
                "NTG": float(compute_NTG(ref, ref)),
            }
        else:
            row = {
                "band_index": i,
                "MI": float(compute_MI(ref, band)),
                "NMI": float(compute_NMI(ref, band)),
                "NCC": float(compute_NCC(ref, band)),
                "NTG": float(compute_NTG(ref, band)),
            }
        metrics["per_band"].append(row)
        for k in keys:
            sums[k] += row[k]
    n = len(bands)
    for k in keys:
        metrics["mean"][k] = sums[k] / n
    return metrics


def compare_metrics(before, after):
    summary = {}
    for key in ("MI", "NMI", "NCC", "NTG"):
        b, a = before["mean"][key], after["mean"][key]
        summary[key] = {"before": b, "after": a, "delta": a - b}
    return summary


def build_groupwise_config(
    experiment: str,
    exp_name: str,
    stack_dir,
    image_size,
    device,
    num_bands,
    band_files,
    eval_ref_band_idx,
    spectral_path,
    registration_kwargs,
    anchor_band=None,
    extra=None,
):
    cfg = {
        "experiment": experiment,
        "exp_name": exp_name,
        "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "stack_dir": str(stack_dir),
        "spectral_path": str(spectral_path),
        "num_bands": num_bands,
        "wavelength_range_nm": [band_files[0].stem, band_files[-1].stem],
        "image_size": list(image_size),
        "device": str(device),
        "eval_ref_band": eval_ref_band_idx,
        "registration": registration_kwargs,
    }
    if anchor_band is not None:
        cfg["anchor_band"] = anchor_band
    if extra:
        cfg.update(extra)
    return cfg


def warp_bands_with_elastix_fields(bands_raw, fields) -> List[np.ndarray]:
    """将 Elastix 群组配准返回的逐波段位移场作用于原图。"""
    import SimpleITK as sitk

    warped = []
    for band, field in zip(bands_raw, fields):
        field_x, field_y = field
        fx = sitk.GetImageFromArray(np.asarray(field_x, dtype=np.float64))
        fy = sitk.GetImageFromArray(np.asarray(field_y, dtype=np.float64))
        size = fx.GetSize()
        displacement = sitk.Image(size, sitk.sitkVectorFloat64)
        for i in range(size[0]):
            for j in range(size[1]):
                displacement.SetPixel((i, j), (float(fx.GetPixel((i, j))), float(fy.GetPixel((i, j)))))
        transform = sitk.DisplacementFieldTransform(displacement)
        moving = sitk.GetImageFromArray(np.asarray(band, dtype=np.float32))
        out = sitk.Resample(moving, transform)
        warped.append(sitk.GetArrayFromImage(out).astype(np.float32))
    return warped


def warp_bands_with_shared_flow(bands_raw, flow_2hw: np.ndarray, device: str = "cpu"):
    """将共享位移场 (2,H,W) 作用于原图各波段。"""
    import torch
    from src.python.voxelmorph.layers import SpatialTransformer

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    h, w = bands_raw[0].shape
    flow_t = torch.tensor(flow_2hw, dtype=torch.float32, device=device).unsqueeze(0)
    transformer = SpatialTransformer((h, w)).to(device)
    out = []
    for band in bands_raw:
        bt = torch.tensor(band, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            warped = transformer(bt, flow_t)
        out.append(warped.squeeze().cpu().numpy().astype(np.float32))
    return out


def pairwise_metrics_dict(fixed, moving, warped, keys=("MI", "NMI", "NCC", "NTG")):
    """Pairwise 指标 before/after 字典。"""
    before = {
        "MI": float(compute_MI(fixed, moving)),
        "NMI": float(compute_NMI(fixed, moving)),
        "NCC": float(compute_NCC(fixed, moving)),
        "NTG": float(compute_NTG(fixed, moving)),
    }
    after = {
        "MI": float(compute_MI(fixed, warped)),
        "NMI": float(compute_NMI(fixed, warped)),
        "NCC": float(compute_NCC(fixed, warped)),
        "NTG": float(compute_NTG(fixed, warped)),
    }
    summary = {k: {"before": before[k], "after": after[k], "delta": after[k] - before[k]} for k in keys}
    return {"before": before, "after": after, "summary": summary, "mean": after}
