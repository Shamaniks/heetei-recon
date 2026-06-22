"""
Restores real (approximately) intensity 
$A(d) = \\frac{1}{K_c + K_l \\cdot d + K_q \\cdot d^2}$
"""


import numpy as np
from scipy.spatial import cKDTree as KDTree


# Public API
def restore(
        xyz_cloud:  "np.ndarray",
        xyz_track:  "np.ndarray",
        intensity:  "np.ndarray",
        chunk_size: int,
        lidar_max_range: float,
        restoration_exp: float,
    ) -> "np.ndarray":
    """
    Parameters
    ----------
    xyz_cloud  : (N, 3) float32 point positions
    xyz_track  : (N, 3) float32 point positions
    intensity  : (N,)   uint8 intensity
    chunk_size : point number processed each iteration from config
    """

    track_tree = KDTree(xyz_track)
    n_points = len(xyz_cloud)

    adjusted_intensity = np.zeros(n_points, dtype=np.float32)

    for i in range(0, n_points, chunk_size):
        end_idx = min(i + chunk_size, n_points)
        p_chunk = xyz_cloud[i:end_idx]

        r_meters, _ = track_tree.query(p_chunk, k=1, workers=-1)
        r_coef = r_meters ** restoration_exp
        
        chunk_intensity = intensity[i:end_idx].astype(np.float32) * r_coef
        chunk_intensity[r_meters > lidar_max_range] = 0.0
        
        adjusted_intensity[i:end_idx] = chunk_intensity
    return adjusted_intensity
