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

    track_tree = cKDTree(xyz_track, compact_nodes=True)

    n_points = xyz_cloud.shape[0]
    distances = np.zeros(n_points, dtype=np.float32)

    # Calculating distances to closest track point for each cloud point
    # chunk_size required only there:
    total_chunks = int(n_points / chunk_size) + 1
    pbar = tqdm(total=total_chunks, ascii=" -", bar_format="{l_bar}{bar:20}{r_bar}")

    for i in range(0, n_points, chunk_size):
        end_idx = min(i + chunk_size, n_points)
        pbar.set_description(f"[downsampling] {i}:{end_idx}")

        dists, _ = track_tree.query(xyz_cloud[i:end_idx], k=1, workers=-1)
        distances[i:end_idx] = dists.astype(np.float32)

        pbar.update(1)
    pbar.close()

    del track_tree
    gc.collect()

    # Restoring approx real intensity
    norm_intensity = intensity * (distances ** 2) 
    del distances

    # Voxeling coordinates
    # voxel_size required only there:
    vx = np.floor(xyz_cloud[:, 0] / voxel_size).astype(np.int32)
    vy = np.floor(xyz_cloud[:, 1] / voxel_size).astype(np.int32)
    vz = np.floor(xyz_cloud[:, 2] / voxel_size).astype(np.int32)

    voxel_hash = (vx.astype(np.int64) << 42) ^ \
                 (vy.astype(np.int64) << 21) ^ \
                  vz.astype(np.int64)

    del vx, vy, vz
    gc.collect()

    # Sorting voxels for future max pooling
    sort_order = np.lexsort((norm_intensity, voxel_hash))
    del norm_intensity

    sorted_hash = voxel_hash[sort_order]
    del voxel_hash
    
    _, unique_indices = np.unique(sorted_hash[::-1], return_index=True)
    del sorted_hash

    keep_indices = sort_order[::-1][unique_indices]
    del sort_order, unique_indices
    gc.collect()

    downsampled = xyz_cloud[keep_indices]
    
    assert downsampled.shape[0] < xyz.shape[0], (
        "Downsampling produced no reduction — voxel_size may be smaller "
        "than point spacing; check config.yaml"
    )

    print(
        f"[preprocess] Voxel downsample ({voxel_size} m): "
        f"{xyz.shape[0]:,} → {downsampled.shape[0]:,} points "
        f"({100.0 * downsampled.shape[0] / xyz.shape[0]:.1f}% retained)"
    )

    pcd.points = o3d.utility.Vector3dVector()   # deallocates C++ storage
    del pcd
    gc.collect()

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
