"""
Stage 1 – Raw point ingestion from a .laz file.

Reads X, Y, Z as float32 and intensity as uint8.
No defensive error handling: a malformed file must crash loudly.
"""

import laspy
import numpy as np


def load_point_cloud(laz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Open a .laz file and return (xyz, intensity).

    Returns
    -------
    xyz       : np.ndarray, shape (N, 3), dtype float32
    intensity : np.ndarray, shape (N,),   dtype uint8

    Notes on explicit dtype casting
    --------------------------------
    las.x / las.y / las.z return laspy ScaledArrayView objects backed by
    float64. Materialising them directly into float32 NumPy arrays halves
    memory consumption (300 MB vs 600 MB for 26.5M points) without any
    meaningful precision loss for metre-scale cave geometry.

    Intensity is stored as uint16 in LAS PF1, but EDA confirms values never
    exceed 255, so uint8 is safe and saves another 50 MB.
    """
    # laspy 2.x API: laspy.read() returns a LasData object directly.
    # No context manager used to avoid version-specific __exit__ issues.
    las = laspy.read(laz_path)

    x = np.array(las.x, dtype=np.float32)
    y = np.array(las.y, dtype=np.float32)
    z = np.array(las.z, dtype=np.float32)

    xyz = np.column_stack((x, y, z))
    assert xyz.ndim == 2 and xyz.shape[1] == 3, (
        f"Expected (N, 3) point array, got shape {xyz.shape}"
    )

    intensity = np.array(las.intensity, dtype=np.uint8)
    assert intensity.shape[0] == xyz.shape[0], (
        "Intensity length does not match point count — file may be corrupt"
    )

    print(f"[ingest] Loaded {xyz.shape[0]:,} points from '{laz_path}'")
    return xyz, intensity
