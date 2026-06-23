# 图像处理工具模块
# 包含图像变换、预处理等通用功能

import numpy as np
from scipy.ndimage import rotate


def shift(im1, x1, y1):
    """
    图像平移变换（亚像素精度）
    
    参数:
        im1: 输入图像 numpy数组
        x1: X方向平移量（可为小数）
        y1: Y方向平移量（可为小数）
    
    返回:
        im2: 平移后的图像
    """
    y0, x0 = im1.shape
    
    x1int = int(np.floor(x1))
    x1dec = x1 - x1int
    y1int = int(np.floor(y1))
    y1dec = y1 - y1int
    im2 = np.copy(im1)

    if y1 >= 0:
        for y in range(-y0, -y1int - 1, -1):
            im2[-y - 1, :] = (1 - y1dec) * im2[-y1int - y - 1, :] + y1dec * im2[-y1int - y - 2, :]
        if y1int < y0:
            im2[y1int, :] = (1 - y1dec) * im2[0, :]
        for y in range(max(-y1int, -y0), -2, -1):
            im2[-y - 1, :] = 0
    else:
        if y1dec == 0:
            y1dec += 1
            y1int -= 1
        for y in range(1, y0 + y1int + 1):
            im2[y - 1, :] = y1dec * im2[-y1int + y - 2, :] + (1 - y1dec) * im2[-y1int + y - 1, :]
        if -y1int <= y0:
            im2[y0 + y1int, :] = y1dec * im2[y0 - 1, :]
        for y in range(max(1, y0 + y1int + 2), y0 + 1):
            im2[y - 1, :] = 0

    if x1 >= 0:
        for x in range(-x0, -x1int - 1, -1):
            im2[:, -x - 1] = (1 - x1dec) * im2[:, -x1int - x - 1] + x1dec * im2[:, -x1int - x - 2]
        if x1int < x0:
            im2[:, x1int] = (1 - x1dec) * im2[:, 0]
        for x in range(max(-x1int, -x0), -2, -1):
            im2[:, -x - 1] = 0
    else:
        if x1dec == 0:
            x1dec += 1
            x1int -= 1
        for x in range(1, x0 + x1int + 1):
            im2[:, x - 1] = x1dec * im2[:, -x1int + x - 2] + (1 - x1dec) * im2[:, -x1int + x - 1]
        if -x1int <= x0:
            im2[:, x0 + x1int] = x1dec * im2[:, x0 - 1]
        for x in range(max(1, x0 + x1int + 2), x0 + 1):
            im2[:, x - 1] = 0

    return im2


def shift_and_rotate(im1, x1, y1, rotation_angle):
    """
    图像平移和旋转变换
    
    参数:
        im1: 输入图像 numpy数组
        x1: X方向平移量（可为小数）
        y1: Y方向平移量（可为小数）
        rotation_angle: 旋转角度（度）
    
    返回:
        im2: 变换后的图像
    """
    # 先进行平移变换
    y0, x0 = im1.shape
    x1int = int(np.floor(x1))
    x1dec = x1 - x1int
    y1int = int(np.floor(y1))
    y1dec = y1 - y1int
    im2 = np.copy(im1)

    if y1 >= 0:
        for y in range(-y0, -y1int - 1, -1):
            im2[-y - 1, :] = (1 - y1dec) * im2[-y1int - y - 1, :] + y1dec * im2[-y1int - y - 2, :]
        if y1int < y0:
            im2[y1int, :] = (1 - y1dec) * im2[0, :]
        for y in range(max(-y1int, -y0), -2, -1):
            im2[-y - 1, :] = 0
    else:
        if y1dec == 0:
            y1dec += 1
            y1int -= 1
        for y in range(1, y0 + y1int + 1):
            im2[y - 1, :] = y1dec * im2[-y1int + y - 2, :] + (1 - y1dec) * im2[-y1int + y - 1, :]
        if -y1int <= y0:
            im2[y0 + y1int, :] = y1dec * im2[y0 - 1, :]
        for y in range(max(1, y0 + y1int + 2), y0 + 1):
            im2[y - 1, :] = 0

    if x1 >= 0:
        for x in range(-x0, -x1int - 1, -1):
            im2[:, -x - 1] = (1 - x1dec) * im2[:, -x1int - x - 1] + x1dec * im2[:, -x1int - x - 2]
        if x1int < x0:
            im2[:, x1int] = (1 - x1dec) * im2[:, 0]
        for x in range(max(-x1int, -x0), -2, -1):
            im2[:, -x - 1] = 0
    else:
        if x1dec == 0:
            x1dec += 1
            x1int -= 1
        for x in range(1, x0 + x1int + 1):
            im2[:, x - 1] = x1dec * im2[:, -x1int + x - 2] + (1 - x1dec) * im2[:, -x1int + x - 1]
        if -x1int <= x0:
            im2[:, x0 + x1int] = x1dec * im2[:, x0 - 1]
        for x in range(max(1, x0 + x1int + 2), x0 + 1):
            im2[:, x - 1] = 0

    # 进行旋转校正
    im2 = rotate(im2, -rotation_angle, mode='nearest', reshape=False)
    return im2


def normalize_image(image, method='minmax'):
    """
    图像归一化
    
    参数:
        image: 输入图像
        method: 归一化方法，'minmax'（默认）或 'zscore'
    
    返回:
        normalized: 归一化后的图像
    """
    if method == 'minmax':
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val > 0:
            return (image - min_val) / (max_val - min_val)
        return image
    elif method == 'zscore':
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return (image - mean) / std
        return image
    return image


def denormalize_image(normalized, original):
    """
    图像反归一化
    
    参数:
        normalized: 归一化后的图像
        original: 原始图像（用于获取min/max或mean/std）
    
    返回:
        image: 恢复后的图像
    """
    min_val = np.min(original)
    max_val = np.max(original)
    return normalized * (max_val - min_val) + min_val
