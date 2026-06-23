# PIFReg package
"""
Pyramid Instance Flow Registration (PIFReg)

    from src.python.registration import register_pifreg
    warped = register_pifreg(fixed, moving, device='cuda')
"""

__version__ = '1.0.0'

from .metrics import compute_MI, compute_NMI, compute_NCC, compute_NTG
from .losses import NCC, Grad, MSE
from .registration import register_pifreg, METHOD_NAME, METHOD_FULL_NAME
from .voxelmorph import VxmDense, Unet, SpatialTransformer
