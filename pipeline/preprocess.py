"""
Stage 2 – Preprocessing: voxel downsampling and normal estimation.
"""

import gc

from tqdm import tqdm

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_o3d_cloud(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    """Wrap a (N, 3) float32 array into an Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return pcd


# ── public API ────────────────────────────────────────────────────────────────

def voxel_downsample(
        xyz_cloud: np.ndarray, 
        intensity: np.ndarray,
        xyz_track: np.ndarray,
        voxel_size: float,
        chunk_size: int
    ) -> np.ndarray:
    """
    Reduce point density via voxel grid max pooling
    """
    assert voxel_size > 0.0, "voxel_size must be a positive number"
    assert len(xyz_cloud) == len(intensity), "Cloud and intensity arrays must have the same length"
    

    # Voxeling track
    # voxel_size required there:
    t_vx = np.floor(xyz_track[:, 0] / voxel_size).astype(np.int32)
    t_vy = np.floor(xyz_track[:, 1] / voxel_size).astype(np.int32)
    t_vz = np.floor(xyz_track[:, 2] / voxel_size).astype(np.int32)
    
    track_hashes = (t_vx.astype(np.int64) << 42) ^ \
                   (t_vy.astype(np.int64) << 21) ^ \
                    t_vz.astype(np.int64)
    del t_vx, t_vy, t_vz

    unique_track_hashes = np.unique(track_hashes)
    del track_hashes
    
    t_vx_u = (unique_track_hashes >> 42) & 0x1FFFFF
    t_vy_u = (unique_track_hashes >> 21) & 0x1FFFFF
    t_vz_u = unique_track_hashes & 0x1FFFFF
    del unique_track_hashes
    
    t_vx_u = np.where(t_vx_u >= 1048576, t_vx_u - 2097152, t_vx_u)
    t_vy_u = np.where(t_vy_u >= 1048576, t_vy_u - 2097152, t_vy_u)
    t_vz_u = np.where(t_vz_u >= 1048576, t_vz_u - 2097152, t_vz_u)
    
    track_voxel_coords = np.column_stack((t_vx_u, t_vy_u, t_vz_u)).astype(np.int32)
    print(
        f"[preprocess] Track downsample ({voxel_size} m)"
        f" → {track_voxel_coords.shape[0]:,} points "
    )
    del t_vx_u, t_vy_u, t_vz_u

    track_voxel_tree = cKDTree(track_voxel_coords)
    del track_voxel_coords
    gc.collect()

    # Voxeling cloud
    # voxel_size required there:
    n_points = len(xyz_cloud)
    
    p_vx = np.floor(xyz_cloud[:, 0] / voxel_size).astype(np.int32)
    p_vy = np.floor(xyz_cloud[:, 1] / voxel_size).astype(np.int32)
    p_vz = np.floor(xyz_cloud[:, 2] / voxel_size).astype(np.int32)
    
    adjusted_intensity = np.zeros(n_points, dtype=np.float32)
    max_range_voxels = int(30.0 / voxel_size) # 30 is hardcoded lidar range FIXME

    for i in range(0, n_points, chunk_size):
        end_idx = min(i + chunk_size, n_points)
        
        p_chunk = np.column_stack((p_vx[i:end_idx], p_vy[i:end_idx], p_vz[i:end_idx]))
        
        min_voxel_dist, _ = track_voxel_tree.query(p_chunk, k=1, workers=-1)
        
        r_meters = min_voxel_dist * voxel_size
        
        chunk_intensity = intensity[i:end_idx].astype(np.float32) * (r_meters ** 2)
        
        chunk_intensity[min_voxel_dist > max_range_voxels] = 0.0
        
        adjusted_intensity[i:end_idx] = chunk_intensity
    del track_voxel_tree 

    point_hash = (p_vx.astype(np.int64) << 42) ^ \
             (p_vy.astype(np.int64) << 21) ^ \
              p_vz.astype(np.int64)
    del p_vx, p_vy, p_vz

    sort_order = np.lexsort((adjusted_intensity, point_hash))
    del adjusted_intensity

    sorted_hash = point_hash[sort_order]
    del point_hash

    _, unique_indices = np.unique(sorted_hash[::-1], return_index=True)
    del sorted_hash

    keep_indices = sort_order[::-1][unique_indices]
    del sort_order, unique_indices
    gc.collect()

    downsampled = xyz_cloud[keep_indices]
    del keep_indices
    
    assert downsampled.shape[0] < xyz_cloud.shape[0], (
        "Downsampling produced no reduction — voxel_size may be smaller "
        "than point spacing; check config.yaml"
    )

    print(
        f"[preprocess] Cloud downsample ({voxel_size} m)"
        f" → {downsampled.shape[0]:,} points "
    )

    return downsampled


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
