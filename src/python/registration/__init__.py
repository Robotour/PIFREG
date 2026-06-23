# PIFReg 配准模块（GitHub 维护范围）
from .pif_registration import register_pifreg, METHOD_NAME, METHOD_FULL_NAME


def register_voxelmorph(*args, **kwargs):
    """已弃用别名，请使用 register_pifreg。"""
    import warnings
    warnings.warn(
        'register_voxelmorph 已重命名为 register_pifreg',
        DeprecationWarning,
        stacklevel=2,
    )
    return register_pifreg(*args, **kwargs)


__all__ = [
    'register_pifreg',
    'register_voxelmorph',
    'METHOD_NAME',
    'METHOD_FULL_NAME',
]

# 本地扩展（methods.py 未纳入 Git 时自动跳过）
try:
    from .methods import (
        register_elastix,
        register_elastix_groupwise,
        register_elastix_edge,
        register_elastix_histogram,
        register_stackreg,
        register_keren,
    )
    __all__ += [
        'register_elastix',
        'register_elastix_groupwise',
        'register_elastix_edge',
        'register_elastix_histogram',
        'register_stackreg',
        'register_keren',
    ]
except ImportError:
    pass
