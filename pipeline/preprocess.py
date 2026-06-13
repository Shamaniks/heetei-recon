"""
Stage 2 – Preprocessing: voxel downsampling and normal estimation.

PATCH v1.1 — In-place downsampling + explicit C++ vector teardown
-----------------------------------------------------------------
Open3D PointCloud objects hold their data in C++ std::vector<Eigen::Vector3d>
buffers. CPython's reference-counting GC can decrement the Python wrapper's
refcount to zero, but the C++ destructor only runs when the *wrapper* is
collected — and even then, the OS allocator may not immediately return the
pages to the system (glibc malloc holds free pages in its own pool).

The pattern:
    pcd.points = o3d.utility.Vector3dVector()   # deallocates C++ storage NOW
    pcd.normals = o3d.utility.Vector3dVector()  # same for normals buffer
    del pcd
    gc.collect()

...forces the C++ destructor to run on the *old* vector immediately (by
replacing it with an empty one), then removes the Python wrapper, then runs
any pending Python finalizers. This is the only reliable way to reclaim
Open3D C++ memory before the next large allocation.
"""

import gc

import numpy as np
import open3d as o3d


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_o3d_cloud(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    """Wrap a (N, 3) float32 array into an Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return pcd


# ── public API ────────────────────────────────────────────────────────────────

def voxel_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Reduce point density via voxel grid averaging.

    PATCH v1.1 memory changes
    -------------------------
    1. pcd is built from xyz (xyz is still live — caller owns it).
    2. voxel_down_sample() returns a NEW PointCloud; we assign it back to
       the same name `pcd` so the original 26M-point object loses its last
       Python reference and its C++ destructor runs immediately.
    3. We extract the numpy array, then wipe pcd's C++ buffer before del.
    4. The caller is responsible for del xyz after this call returns.
    """
    assert voxel_size > 0.0, "voxel_size must be a positive number"

    pcd = _build_o3d_cloud(xyz)
    # xyz is still referenced by the caller; we only own `pcd` here.

    # ── PATCH: in-place name rebind frees the 26M-point C++ buffer ────────
    # voxel_down_sample() allocates a new PointCloud internally. By assigning
    # the result back to `pcd`, the old wrapper (holding 26M Eigen::Vector3d)
    # loses its last reference and its destructor runs synchronously in C++.
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # ── end PATCH ─────────────────────────────────────────────────────────

    downsampled = np.asarray(pcd.points, dtype=np.float32)

    assert downsampled.shape[0] < xyz.shape[0], (
        "Downsampling produced no reduction — voxel_size may be smaller "
        "than point spacing; check config.yaml"
    )

    print(
        f"[preprocess] Voxel downsample ({voxel_size} m): "
        f"{xyz.shape[0]:,} → {downsampled.shape[0]:,} points "
        f"({100.0 * downsampled.shape[0] / xyz.shape[0]:.1f}% retained)"
    )

    # ── PATCH: explicit C++ teardown before returning ──────────────────────
    # np.asarray above shares memory with the C++ buffer via a zero-copy
    # view. We must copy first (dtype=np.float32 cast already copies it),
    # then we can safely wipe the C++ side.
    pcd.points = o3d.utility.Vector3dVector()   # deallocates C++ storage
    del pcd
    gc.collect()
    # ── end PATCH ─────────────────────────────────────────────────────────

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
        pcd.orient_normals_toward_camera_location(
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
