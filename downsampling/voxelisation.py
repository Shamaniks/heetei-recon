"""
Coordinates to voxel indexes functions
"""

from typing import Callable, Literal

import numpy as np


def voxelisation(
        mode: Literal["binary", "division"],
        xyz_cloud: np.ndarray, 
        voxel_size: float,
        scale: float,
    ) -> np.ndarray:
    return VOXEL_METHODS[mode](
        xyz_cloud,
        voxel_size,
        scale
    )
    

def division_voxelisation(
        xyz_cloud: np.ndarray, 
        voxel_size: float,
        scale: float,
    ) -> np.ndarray:

    xyz_voxels = xyz_cloud // (voxel_size / scale)
    
    return xyz_voxels


def binary_voxelisation(
        xyz_cloud: np.ndarray, 
        voxel_size: float,
        scale: float,
    ) -> np.ndarray:

    steps = voxel_size / scale
    n = int(np.round(np.log2(steps)))
    mersenne_number = (1 << n) - 1

    print(f"[downsampling] Binary voxel size {(1 << n) * scale}")

    mask = ~mersenne_number
    xyz_voxels = xyz_cloud & mask
    return xyz_voxels


VOXEL_METHODS: dict[
    Literal["binary", "division"], 
    Callable[[np.ndarray, float, float], np.ndarray]
] = {
    "binary": binary_voxelisation,
    "division": division_voxelisation,
}


