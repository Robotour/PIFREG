# 配准模块
from .pif_registration import register_pifreg, METHOD_NAME, METHOD_FULL_NAME
from .pif_groupwise_stackflow import register_pifreg_groupwise_stackflow
from .pif_groupwise_stackflow3d import register_pifreg_groupwise_stackflow3d
from .pif_groupwise_chain import register_pifreg_chain
from .pif_groupwise_joint import register_pifreg_groupwise_joint
from .pif_groupwise_sliding_window import register_pifreg_groupwise_sliding_window
from .pif_groupwise_spatial_window import register_pifreg_groupwise_spatial_window
from .methods import (
    register_elastix,
    register_elastix_groupwise,
    register_elastix_edge,
    register_elastix_histogram,
    register_stackreg,
    register_keren,
    register_voxelmorph,
)

__all__ = [
    'register_pifreg',
    'register_pifreg_groupwise_stackflow',
    'register_pifreg_groupwise_stackflow3d',
    'register_pifreg_chain',
    'register_pifreg_groupwise_joint',
    'register_pifreg_groupwise_sliding_window',
    'register_pifreg_groupwise_spatial_window',
    'register_voxelmorph',
    'register_elastix',
    'register_elastix_groupwise',
    'register_elastix_edge',
    'register_elastix_histogram',
    'register_stackreg',
    'register_keren',
    'METHOD_NAME',
    'METHOD_FULL_NAME',
]
