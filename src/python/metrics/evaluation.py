# 图像配准评价指标模块
# 包含MI、NMI、NCC、NTG等评价指标的计算函数

import numpy as np


def compute_MI(image1, image2, bins=256):
    """
    计算两幅图像的互信息（Mutual Information）
    
    参数:
        image1: 第一幅图像 numpy数组
        image2: 第二幅图像 numpy数组
        bins: 直方图 bins数量，默认256
    
    返回:
        mutual_info: 互信息值
    """
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d))  # 计算联合概率分布
    px = np.sum(pxy, axis=1)  # 计算边际概率
    py = np.sum(pxy, axis=0)

    # 计算互信息
    px_py = np.outer(px, py)  # 计算 p(x) * p(y)
    non_zero = pxy > 0  # 避免log(0)
    mutual_info = np.sum(pxy[non_zero] * np.log(pxy[non_zero] / px_py[non_zero]))

    return mutual_info


def compute_NMI(image1, image2, bins=256):
    """
    计算归一化互信息（Normalized Mutual Information, NMI）
    
    参数:
        image1: 第一幅图像 numpy数组
        image2: 第二幅图像 numpy数组
        bins: 直方图 bins数量，默认256
    
    返回:
        nmi: 归一化互信息值，范围通常在1-2之间
    """
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d))

    Hx = -np.sum(np.sum(pxy, axis=1) * np.log(np.sum(pxy, axis=1) + 1e-10))  # 避免log(0)
    Hy = -np.sum(np.sum(pxy, axis=0) * np.log(np.sum(pxy, axis=0) + 1e-10))
    Hxy = -np.sum(pxy * np.log(pxy + 1e-10))

    return (Hx + Hy) / Hxy


def compute_NCC(image1, image2):
    """
    计算归一化互相关（Normalized Cross Correlation, NCC）
    
    参数:
        image1: 第一幅图像 numpy数组
        image2: 第二幅图像 numpy数组
    
    返回:
        ncc: 归一化互相关值，范围在-1到1之间，1表示完全正相关
    """
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    mean1 = np.mean(image1)
    mean2 = np.mean(image2)

    numerator = np.sum((image1 - mean1) * (image2 - mean2))
    denominator = np.sqrt(np.sum((image1 - mean1) ** 2) * np.sum((image2 - mean2) ** 2))

    return numerator / (denominator + 1e-10)  # 避免除零


def compute_NTG(image1, image2):
    """
    计算归一化总梯度（Normalized Total Gradient, NTG）指标
    
    参数:
        image1: 第一幅图像（灰度图），numpy数组，数据类型为float32或float64
        image2: 第二幅图像（灰度图），numpy数组，数据类型为float32或float64
    
    返回:
        ntg: NTG值，越小表示两幅图像越相似
    """
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    diff = image1 - image2

    def total_gradient(img):
        gx = np.abs(np.gradient(img, axis=1))  # x方向
        gy = np.abs(np.gradient(img, axis=0))  # y方向
        return gx + gy

    tg_diff = total_gradient(diff)
    tg_img1 = total_gradient(image1)
    tg_img2 = total_gradient(image2)

    numerator = np.sum(tg_diff)
    denominator = np.sum(tg_img1 + tg_img2) + 1e-10  # 避免除零

    ntg = numerator / denominator
    return ntg


def compute_SSIM(image1, image2):
    """
    计算结构相似性指标（需要skimage库）
    
    参数:
        image1: 第一幅图像 numpy数组
        image2: 第二幅图像 numpy数组
    
    返回:
        ssim: SSIM值，范围在-1到1之间，1表示完全相同
    """
    from skimage.metrics import structural_similarity as ssim
    return ssim(image1, image2)
