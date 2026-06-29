# 配准网络模块 — 向后兼容入口（实现已迁移至 voxelmorph 包）

from ..voxelmorph import ConvBlock, SpatialTransformer, Unet, VxmDense

RegistrationUNet = VxmDense

from .registration_networks_discriminator import Discriminator

__all__ = [
    'VxmDense',
    'Unet',
    'ConvBlock',
    'SpatialTransformer',
    'RegistrationUNet',
    'Discriminator',
]
