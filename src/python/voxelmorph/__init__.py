"""Official VoxelMorph PyTorch backend (vendored into this project)."""

from .config import compact_unet_features, default_unet_features
from .layers import ResizeTransform, SpatialTransformer, VecInt
from .losses import Dice, Grad, MSE, NCC
from .modelio import LoadableModel, store_config_args
from .networks import ConvBlock, Unet, VxmDense
from .experiment import create_run_dir, run_full_experiment
from .training import (
    build_adjacent_band_pairs,
    discover_band_folders,
    evaluate_voxelmorph_pairs,
    evaluate_voxelmorph_sessions,
    save_split_manifest,
    split_folders_train_test,
    train_voxelmorph,
    train_voxelmorph_baseline,
    train_voxelmorph_stack_spatial,
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
    'create_run_dir',
    'run_full_experiment',
    'build_adjacent_band_pairs',
    'discover_band_folders',
    'evaluate_voxelmorph_pairs',
    'evaluate_voxelmorph_sessions',
    'save_split_manifest',
    'split_folders_train_test',
    'train_voxelmorph',
    'train_voxelmorph_baseline',
    'train_voxelmorph_stack_spatial',
]
