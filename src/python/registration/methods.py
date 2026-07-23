# 图像配准算法模块
# 包含多种配准方法的实现：PIFReg、Elastix、StackReg、KEREN

import cv2
import numpy as np
from ..vendor import pyelastix
import SimpleITK as sitk
from pystackreg import StackReg
from skimage.exposure import match_histograms
from scipy.fft import fft2, ifft2
from scipy.ndimage import rotate

from .pif_registration import register_pifreg
from ..utils.image_transform import shift, shift_and_rotate
from ..preprocessing.band_preprocess import histogram_equalize_band, refresh_histogram_equalized


def _warp_with_elastix_field(moving_image, field):
    """Apply Elastix (fx, fy) displacement field to an image."""
    field_x = sitk.GetImageFromArray(field[0])
    field_y = sitk.GetImageFromArray(field[1])
    size = field_x.GetSize()
    displacement_field = sitk.Image(size, sitk.sitkVectorFloat64)
    for i in range(size[0]):
        for j in range(size[1]):
            displacement_field.SetPixel((i, j), [field_x.GetPixel((i, j)), field_y.GetPixel((i, j))])
    field_sitk = sitk.DisplacementFieldTransform(displacement_field)
    moving_sitk = sitk.GetImageFromArray(np.asarray(moving_image, dtype=np.float32))
    moving_deformed_sitk = sitk.Resample(moving_sitk, field_sitk)
    return sitk.GetArrayFromImage(moving_deformed_sitk).astype(np.float32)


def register_elastix(
    fixed_image,
    moving_image,
    epochs=20,
    spacinginvoxels=20,
    moving_raw=None,
):
    """
    Elastix传统配准方法

    在 fixed_image / moving_image（通常为直方图均衡图）上估计位移场，
    将位移场作用于 moving_raw（原图）；若未提供 moving_raw 则作用于 moving_image。
    """
    params = pyelastix.get_default_params()
    params.MaximumNumberOfIterations = epochs
    params.FinalGridSpacingInVoxels = spacinginvoxels

    _, field = pyelastix.register(moving_image, fixed_image, params)
    warp_target = moving_raw if moving_raw is not None else moving_image
    return _warp_with_elastix_field(warp_target, field)


def register_voxelmorph(*args, **kwargs):
    """已弃用，请使用 register_pifreg。"""
    import warnings
    warnings.warn(
        'register_voxelmorph 已重命名为 register_pifreg（PIFReg），'
        '二者不再等同 VoxelMorph 论文方法。',
        DeprecationWarning,
        stacklevel=2,
    )
    return register_pifreg(*args, **kwargs)


# ============== Elastix 配准方法 ==============

def register_elastix_groupwise(img_list, epochs=100, spacinginvoxels=20, verbose=1):
    """
    Elastix 群组配准（BSplineStackTransform + VarianceOverLastDimensionMetric）

    将多波段图像作为 3D 栈一次性配准到共同空间，避免链式 pairwise 的误差累积。

    参数:
        img_list: 波段图像列表，每个元素为 (H, W) 的 numpy 数组，建议按波长升序排列
        epochs: 每个金字塔层最大迭代次数
        spacinginvoxels: B 样条网格间距（体素）
        verbose: Elastix 输出详细程度 0/1/2

    返回:
        registered_list: 配准后的图像列表，长度与 img_list 相同
        fields: 每个波段对应的位移场列表，元素为 (field_x, field_y)
    """
    params = pyelastix.get_default_params()
    params.MaximumNumberOfIterations = epochs
    params.FinalGridSpacingInVoxels = spacinginvoxels

    warped_stack, fields = pyelastix.register(
        img_list, None, params, verbose=verbose
    )

    if isinstance(warped_stack, np.ndarray):
        registered_list = [
            warped_stack[i].astype(np.float32) for i in range(warped_stack.shape[0])
        ]
    else:
        registered_list = [np.asarray(band, dtype=np.float32) for band in warped_stack]

    return registered_list, fields


def _chain_pair_indices(n, descending=True):
    """Return (fixed_idx, moving_idx) pairs for wavelength chain registration."""
    if descending:
        return [(i + 1, i) for i in range(n - 2, -1, -1)]
    return [(i - 1, i) for i in range(1, n)]


def register_elastix_chain(
    img_list,
    epochs=20,
    spacinginvoxels=20,
    descending=True,
    raw_list=None,
):
    """
    Elastix 链式 pairwise 配准（与 VoxelMorph 推理链一致）

    在直方图均衡图上估计位移，位移作用于原图；链上下一步 fixed 为
    「上一 band 原图 warp 后再做直方图均衡」的结果。
    """
    eq_list = [np.asarray(b, dtype=np.float32).copy() for b in img_list]
    raw_list = [
        np.asarray(b, dtype=np.float32).copy()
        for b in (raw_list if raw_list is not None else img_list)
    ]
    if len(eq_list) < 2:
        return raw_list
    for fixed_idx, moving_idx in _chain_pair_indices(len(eq_list), descending=descending):
        raw_list[moving_idx] = register_elastix(
            eq_list[fixed_idx],
            eq_list[moving_idx],
            epochs=epochs,
            spacinginvoxels=spacinginvoxels,
            moving_raw=raw_list[moving_idx],
        )
        eq_list[moving_idx] = refresh_histogram_equalized(raw_list[moving_idx])
    return raw_list


def register_elastix_edge(fixed_image, moving_image, epochs=20, spacinginvoxels=20):
    """
    Elastix + 边缘检测配准方法
    
    参数:
        fixed_image: 固定图像
        moving_image: 移动图像
        epochs: 最大迭代次数
        spacinginvoxels: 网格间距
    
    返回:
        warped_image: 配准后的图像
    """
    # 归一化到 0-255
    fixed_image_nor = ((fixed_image - np.min(fixed_image)) / (np.max(fixed_image) - np.min(fixed_image)) * 255).astype(np.uint8)
    moving_image_nor = ((moving_image - np.min(moving_image)) / (np.max(moving_image) - np.min(moving_image)) * 255).astype(np.uint8)

    # 边缘检测
    fixed_edge = cv2.Canny(cv2.GaussianBlur(fixed_image_nor, (5, 5), 0), 50, 150).astype(np.float32)
    moving_edge = cv2.Canny(cv2.GaussianBlur(moving_image_nor, (5, 5), 0), 50, 150).astype(np.float32)

    # 设置 Elastix 参数
    params = pyelastix.get_default_params()
    params.MaximumNumberOfIterations = epochs
    params.FinalGridSpacingInVoxels = spacinginvoxels

    # 进行配准（使用边缘图像）
    _, field = pyelastix.register(moving_edge, fixed_edge, params)

    return _warp_with_elastix_field(moving_image, field)


def register_elastix_histogram(fixed_image, moving_image, epochs=20, spacinginvoxels=20):
    """
    Elastix + 直方图匹配配准方法
    
    参数:
        fixed_image: 固定图像
        moving_image: 移动图像
        epochs: 最大迭代次数
        spacinginvoxels: 网格间距
    
    返回:
        warped_image: 配准后的图像
    """
    # 归一化
    fixed_image_nor = ((fixed_image - np.min(fixed_image)) / (np.max(fixed_image) - np.min(fixed_image)) * 255).astype(np.uint8)
    moving_image_nor = ((moving_image - np.min(moving_image)) / (np.max(moving_image) - np.min(moving_image)) * 255).astype(np.uint8)

    # 直方图匹配
    matched_moving_image = match_histograms(moving_image_nor, fixed_image_nor, channel_axis=None).astype(np.uint8)

    # Elastix配准
    params = pyelastix.get_default_params()
    params.MaximumNumberOfIterations = epochs
    params.FinalGridSpacingInVoxels = spacinginvoxels

    _, field = pyelastix.register(matched_moving_image, fixed_image_nor, params)

    return _warp_with_elastix_field(moving_image, field)


# ============== StackReg 配准方法 ==============

def register_stackreg(fixed_image, moving_image, transform_type='bilinear', moving_raw=None):
    """
    StackReg堆栈配准方法

    在 fixed_image / moving_image（直方图均衡）上估计变换，
    将变换作用于 moving_raw（原图）；未提供时作用于 moving_image。
    """
    if transform_type == 'translation':
        sr = StackReg(StackReg.TRANSLATION)
    elif transform_type == 'rigid':
        sr = StackReg(StackReg.RIGID_BODY)
    elif transform_type == 'scaled_rotation':
        sr = StackReg(StackReg.SCALED_ROTATION)
    elif transform_type == 'affine':
        sr = StackReg(StackReg.AFFINE)
    elif transform_type == 'bilinear':
        sr = StackReg(StackReg.BILINEAR)
    else:
        raise ValueError(f"不支持的 transform_type: {transform_type}")

    sr.register(fixed_image, moving_image)
    warp_target = moving_raw if moving_raw is not None else moving_image
    registered_image = sr.transform(warp_target)

    return registered_image.astype(np.float32)


# ============== KEREN 配准方法 ==============

def register_keren(img_list):
    """
    KEREN金字塔Lucas-Kanade配准方法
    
    参数:
        img_list: 图像列表，第一幅为参考图像
    
    返回:
        delta_est: 每幅图像的平移量 (N, 2)
        phi_est: 每幅图像的旋转角度 (N,)
    """
    img_tem = [img_list[0]]
    delta_est = np.zeros((len(img_list), 2))
    phi_est = np.zeros(len(img_list))

    for img_num in range(1, len(img_list)):
        lp = cv2.getGaussianKernel(3, 1)
        lp = np.outer(lp, lp.transpose())
        img_pro = [img_list[img_num]]

        pyrlevel_num = 5
        for i in range(1, pyrlevel_num):
            img_tem.append(cv2.resize(cv2.filter2D(img_tem[i - 1], -1, lp), (0, 0), fx=0.5, fy=0.5))
            img_pro.append(cv2.resize(cv2.filter2D(img_pro[i - 1], -1, lp), (0, 0), fx=0.5, fy=0.5))

        stot = np.zeros(3)

        # 多尺度金字塔配准
        for pyrlevel in range(pyrlevel_num - 1, -1, -1):
            f0 = img_tem[pyrlevel]
            f1 = img_pro[pyrlevel]

            y0, x0 = f0.shape
            xmean, ymean = x0 / 2, y0 / 2
            x = np.kron(np.arange(-xmean, xmean), np.ones(y0).reshape(-1, 1))
            y = np.kron(np.ones(x0), np.arange(-ymean, ymean).reshape(-1, 1))

            sigma = 1

            g1 = -np.exp(-((np.arange(y0)[:, None] - ymean) ** 2 + (np.arange(x0) - xmean) ** 2) / (2 * sigma ** 2)) * (
                        np.arange(y0)[:, None] - ymean) / (2 * np.pi * sigma ** 2)
            g2 = -np.exp(-((np.arange(y0)[:, None] - ymean) ** 2 + (np.arange(x0) - xmean) ** 2) / (2 * sigma ** 2)) * (
                        np.arange(x0) - xmean) / (2 * np.pi * sigma ** 2)
            g3 = np.exp(-((np.arange(y0)[:, None] - ymean) ** 2 + (np.arange(x0) - xmean) ** 2) / (2 * sigma ** 2)) / (
                        2 * np.pi * sigma ** 2)

            a = np.real(ifft2(fft2(f1) * fft2(g2)))
            c = np.real(ifft2(fft2(f1) * fft2(g1)))
            b = np.real(ifft2(fft2(f1) * fft2(g3))) - np.real(ifft2(fft2(f0) * fft2(g3)))
            R = c * x - a * y

            A = np.array([[np.sum(a * a), np.sum(a * c), np.sum(R * a)],
                          [np.sum(a * c), np.sum(c * c), np.sum(R * c)],
                          [np.sum(R * a), np.sum(R * c), np.sum(R * R)]])
            Ainv = np.linalg.inv(A)

            b1 = np.sum(a * b)
            b2 = np.sum(c * b)
            b3 = np.sum(R * b)
            s = Ainv @ np.array([b1, b2, b3])
            st = s.copy()

            it = 1
            while (np.abs(s[0]) + np.abs(s[1]) > 0.1) and it < 25:
                f0_ = shift(f0, -st[0], -st[1])
                b = np.real(ifft2(fft2(f1) * fft2(g3))) - np.real(ifft2(fft2(f0_) * fft2(g3)))
                s = Ainv @ np.array([np.sum(a * b), np.sum(c * b), np.sum(R * b)])
                st += s
                it += 1

            st[2] = -st[2] * 180 / np.pi
            stot[:2] += st[1::-1]
            stot[2] += st[2]

            if pyrlevel > 0:
                img_pro[pyrlevel - 1] = shift(img_pro[pyrlevel - 1], 2 * stot[1], 2 * stot[0])

        delta_est[img_num, :] = stot[:2]
        phi_est[img_num] = stot[2]

        img_tem[0] = (img_tem[0] + shift(img_list[img_num], stot[0], stot[1])) / 2

    return delta_est, phi_est
