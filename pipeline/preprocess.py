"""
Stage 2 – Preprocessing: voxel downsampling and normal estimation.

Downsampling is always applied.
Normal estimation is only performed when the reconstruction algorithm
requires it (Ball Pivoting). Alpha Shapes do not need normals.
"""

import numpy as np
import open3d as o3d


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_o3d_cloud(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    """Wrap a (N,3) float32 array into an Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    # open3d expects float64 internally; it will upcast silently, but we pass
    # the array directly and let open3d manage its own copy.
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return pcd


# ── public API ────────────────────────────────────────────────────────────────

def voxel_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Reduce point density via voxel grid averaging.

    Each voxel cell retains one representative point (centroid of all points
    that fall inside it). This is Open3D's built-in implementation, which
    operates in C++ and does not materialise intermediate Python objects.

    Parameters
    ----------
    xyz        : (N, 3) float32 raw point positions
    voxel_size : edge length of each cubic voxel cell, in metres

    Returns
    -------
    downsampled : (M, 3) float32,  M < N
    """
    assert voxel_size > 0.0, "voxel_size must be a positive number"

    pcd = _build_o3d_cloud(xyz)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    downsampled = np.asarray(pcd_down.points, dtype=np.float32)
    assert downsampled.shape[0] < xyz.shape[0], (
        "Downsampling produced no reduction — voxel_size may be smaller "
        "than point spacing; check config.yaml"
    )

    print(
        f"[preprocess] Voxel downsample ({voxel_size} m): "
        f"{xyz.shape[0]:,} → {downsampled.shape[0]:,} points "
        f"({100.0 * downsampled.shape[0] / xyz.shape[0]:.1f}% retained)"
    )
    return downsampled


def estimate_normals(
    xyz: np.ndarray,
    knn: int,
    orient_toward_origin: bool,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Estimate per-point normals via PCA over KNN neighbourhoods.

    Orientation strategy
    --------------------
    For a gallery cave scanned from the inside, pointing normals toward the
    scan origin (0, 0, 0) makes the normals face inward — toward the scanner.
    Ball Pivoting rolls the ball on the side the normals point toward, so
    inward-facing normals cause it to reconstruct the interior wall surface,
    which is exactly what we want.

    Parameters
    ----------
    xyz                  : (N, 3) float32 downsampled positions
    knn                  : number of neighbours for PCA
    orient_toward_origin : if True, flip normals to face (0, 0, 0)

    Returns
    -------
    pcd     : Open3D PointCloud with normals attached
    normals : (N, 3) float64 normal vectors
    """
    assert knn >= 3, "Need at least 3 neighbours for PCA normal estimation"

    pcd = _build_o3d_cloud(xyz)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
    )

    if orient_toward_origin:
        # orient_normals_toward_camera_location treats the given point as the
        # "camera"; using (0,0,0) orients all normals to face the origin,
        # which is the scanner position for a zero-offset cave scan.
        pcd.orient_normals_toward_camera_location(
            camera_location=np.array([0.0, 0.0, 0.0])
        )

    normals = np.asarray(pcd.normals, dtype=np.float64)
    assert normals.shape[0] == xyz.shape[0], (
        "Normal count does not match point count after estimation"
    )

    print(f"[preprocess] Normals estimated (knn={knn}, "
          f"orient_to_origin={orient_toward_origin})")
    return pcd, normals
