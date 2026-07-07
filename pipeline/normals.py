"""
Stage — Normal estimation (required for Ball Pivoting reconstruction only).

Gap note
--------
pipeline.reconstruct.reconstruct() dispatches to Ball Pivoting when
configured, and asserts it is given a PointCloud with normals attached.
No working implementation of that estimation step existed in the
repository: the only version present was dead code (commented out) inside
downsampling/max_pooling.py, referencing an undefined `_build_o3d_cloud`
helper. This module is that missing piece, rebuilt as its own stage per
the single-responsibility/decomposition rule rather than left inline in
the downsampling module.
"""

import numpy as np
import open3d as o3d


def _build_o3d_cloud(xyz_metres: np.ndarray) -> o3d.geometry.PointCloud:
    assert xyz_metres.ndim == 2 and xyz_metres.shape[1] == 3, (
        "xyz_metres must have shape (N, 3)"
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_metres.astype(np.float64))
    return pcd


def estimate_normals(
        xyz_metres: np.ndarray,
        knn: int,
        orient_toward_origin: bool,
    ) -> o3d.geometry.PointCloud:
    """
    Estimate per-point normals via PCA over KNN neighbourhoods.

    Returns the PointCloud WITH normals attached. The caller owns the
    explicit C++ teardown after use (wipe .points/.normals before del),
    since the KD-tree and KNN index built internally stay attached to the
    PointCloud and are not freed on return.

    Parameters
    ----------
    xyz_metres : (N, 3) float64 point positions, already scaled to metres
                 (this stage runs after the int32-grid -> metres
                 conversion; see main.py stage_to_metres).
    knn        : neighbour count for the PCA normal estimate.
    """
    assert knn >= 3, "Need at least 3 neighbours for PCA normal estimation"

    pcd = _build_o3d_cloud(xyz_metres)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
    )

    if orient_toward_origin:
        pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0.0, 0.0, 0.0])
        )

    normals = np.asarray(pcd.normals)
    assert normals.shape[0] == xyz_metres.shape[0], (
        "Normal count does not match point count after estimation"
    )
    print(f"[normals] Estimated normals (knn={knn}, "
          f"orient_to_origin={orient_toward_origin})")
    return pcd
