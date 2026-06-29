# 高光谱图像转RGB彩色图模块
# 根据光谱响应曲线将高光谱图像转换为RGB显示

import numpy as np
import pandas as pd
from skimage.color import xyz2rgb


def hsi_to_rgb(cropped_images, spectral_data_path=None):
    """
    将高光谱图像序列转换为RGB彩色图
    
    参数:
        cropped_images: 高光谱图像列表，每个元素为一个波段的灰度图 numpy数组
        spectral_data_path: 光谱数据Excel文件路径，默认为'../HSI2RGB20240517.xlsx'
    
    返回:
        rgb_img: RGB彩色图 numpy数组，形状为(H, W, 3)
    """
    if spectral_data_path is None:
        spectral_data_path = '../HSI2RGB20240517.xlsx'
    
    hsi2rgb_df = pd.read_excel(spectral_data_path)
    X_m = hsi2rgb_df['R'].values
    Y_m = hsi2rgb_df['G'].values
    Z_m = hsi2rgb_df['B'].values

    band_num = len(cropped_images)
    row, col = np.shape(cropped_images[0])

    Xf = np.zeros((row, col))
    Yf = np.zeros((row, col))
    Zf = np.zeros((row, col))

    # 根据 band_num 累加每个通道的贡献
    for i, spectral_image in enumerate(cropped_images):
        I = np.float32(spectral_image)
        Xf += I * X_m[i]
        Yf += I * Y_m[i]
        Zf += I * Z_m[i]

    # 标准化 XYZ 值
    Xf /= sum(X_m)
    Yf /= sum(Y_m)
    Zf /= sum(Z_m)

    # 合并 XYZ 成一张图像
    xyz_img = np.stack((Xf, Yf, Zf), axis=-1)
    normalized_img = np.zeros_like(xyz_img, dtype=float)
    for channel in range(xyz_img.shape[2]):
        max_value = xyz_img[:, :, channel].max()
        if max_value != 0:
            normalized_img[:, :, channel] = xyz_img[:, :, channel] / max_value

    # 使用 skimage 将 XYZ 转换为 RGB
    rgb_img = xyz2rgb(normalized_img)
    r = np.uint8(rgb_img[:, :, 0] * 200)
    g = np.uint8(rgb_img[:, :, 1] * 200)
    b = np.uint8(rgb_img[:, :, 2] * 200)
    rgb_img = np.stack([b, g, r], axis=-1)
    return rgb_img
