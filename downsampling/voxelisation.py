"""
Coordinates to voxel indexes functions
"""


import numpy as np


def binary_voxelisation(
        xyz_cloud: np.ndarray, 
        voxel_size: float,
        scale: float,
    ) -> np.ndarray:

    steps = voxel_size/ scale
    n = int(np.round(np.log2(steps)))
    mersenne_number = (1 << n) - 1

    print(f"[downsampling] Binary voxel size {(1 << n) * scale}")

    mask = ~mersenne_number
    xyz_voxels = xyz_cloud & mask
    return xyz_voxels
