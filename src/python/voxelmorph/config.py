"""VoxelMorph default configuration (from official voxelmorph py utils)."""


def default_unet_features():
    """Default U-Net encoder/decoder feature channels."""
    return [
        [16, 32, 32, 32],
        [32, 32, 32, 32, 32, 16, 16],
    ]


def compact_unet_features():
    """Lightweight U-Net (~4× fewer params) for faster PIFReg test-time optimization."""
    return [
        [8, 16, 16, 16],
        [16, 16, 16, 16, 16, 8, 8],
    ]
