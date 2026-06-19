"""
Preprocessing: voxel downsampling and normal estimation.
"""

import gc

from tqdm import tqdm

import numpy as np
from scipy.spatial import cKDTree


def voxel_downsample(
        xyz_cloud: np.ndarray, 
        intensity: np.ndarray,
        voxel_size: float,
    ) -> np.ndarray:
    """
    Reduce point density via voxel grid max pooling
    """
    assert voxel_size > 0.0, "voxel_size must be a positive number"
    assert len(xyz_cloud) == len(intensity), "Cloud and intensity arrays must have the same length"
    
    # Voxeling cloud
    # voxel_size required there:
    n_points = len(xyz_cloud)
    
    p_vx = np.floor(xyz_cloud[:, 0] / voxel_size).astype(np.int32)
    p_vy = np.floor(xyz_cloud[:, 1] / voxel_size).astype(np.int32)
    p_vz = np.floor(xyz_cloud[:, 2] / voxel_size).astype(np.int32)
    
    point_hash = (p_vx.astype(np.int64) << 42) ^ \
             (p_vy.astype(np.int64) << 21) ^ \
              p_vz.astype(np.int64)
    del p_vx, p_vy, p_vz

    sort_order = np.lexsort((intensity, point_hash))

    sorted_hash = point_hash[sort_order]
    del point_hash

    _, unique_indices = np.unique(sorted_hash[::-1], return_index=True)
    del sorted_hash

    keep_indices = sort_order[::-1][unique_indices]
    del sort_order, unique_indices
    gc.collect()

    return keep_indices

'''
def estimate_normals(
    xyz: np.ndarray,
    knn: int,
    orient_toward_origin: bool,
) -> o3d.geometry.PointCloud:
    """
    Estimate per-point normals via PCA over KNN neighbourhoods.

    Returns the PointCloud WITH normals attached.
    The caller is responsible for the explicit C++ teardown after use
    (see stage_preprocess in main.py).

    PATCH v1.1 memory note
    ----------------------
    estimate_normals() internally builds a KD-tree and a KNN neighbour
    index. Both live on the C++ heap and are NOT freed when the function
    returns — they remain attached to the PointCloud object. The caller
    MUST wipe pcd.points and pcd.normals before del pcd to reclaim this
    memory before reconstruction starts.
    """
    assert knn >= 3, "Need at least 3 neighbours for PCA normal estimation"

    pcd = _build_o3d_cloud(xyz)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
    )

    if orient_toward_origin:
        pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0.0, 0.0, 0.0])
        )

    normals = np.asarray(pcd.normals)
    assert normals.shape[0] == xyz.shape[0], (
        "Normal count does not match point count after estimation"
    )

    print(f"[preprocess] Normals estimated (knn={knn}, "
          f"orient_to_origin={orient_toward_origin})")

    # Return the full pcd so the caller can pass it to BPA.
    # The caller owns the teardown sequence.
    return pcd
'''
