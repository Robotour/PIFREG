"""Official VoxelMorph PyTorch backend (vendored into this project)."""

from .config import compact_unet_features, default_unet_features
from .layers import ResizeTransform, SpatialTransformer, VecInt
from .losses import Dice, Grad, MSE, NCC
from .modelio import LoadableModel, store_config_args
from .networks import ConvBlock, Unet, VxmDense
from .training import (
    build_adjacent_band_pairs,
    discover_band_folders,
    evaluate_voxelmorph_pairs,
    save_split_manifest,
    split_folders_train_test,
    train_voxelmorph,
)

__all__ = [
    'default_unet_features',
    'compact_unet_features',
    'ResizeTransform',
    'SpatialTransformer',
    'VecInt',
    'NCC',
    'MSE',
    'Dice',
    'Grad',
    'LoadableModel',
    'store_config_args',
    'ConvBlock',
    'Unet',
    'VxmDense',
    'build_adjacent_band_pairs',
    'discover_band_folders',
    'evaluate_voxelmorph_pairs',
    'save_split_manifest',
    'split_folders_train_test',
    'train_voxelmorph',
]
