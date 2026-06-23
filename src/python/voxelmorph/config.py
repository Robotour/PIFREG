"""VoxelMorph default configuration (from official voxelmorph py utils)."""


def default_unet_features():
    """Default U-Net encoder/decoder feature channels."""
    return [
        [16, 32, 32, 32],
        [32, 32, 32, 32, 32, 16, 16],
    ]
