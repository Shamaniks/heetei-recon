"""
Stage 3 – Mesh reconstruction.

Two algorithms are provided, both gap-preserving by construction:

  Alpha Shapes
  ────────────
  A triangle is included in the mesh only if its circumscribed sphere has
  radius ≤ alpha AND contains no other points. In a cave squeeze (a narrow
  passage with no scan points inside), the circumsphere of any hypothetical
  spanning triangle would be large (>> alpha) or would contain points from
  the opposite wall. Either condition excludes the triangle. Result: gaps
  stay open without any post-hoc patching.

  Ball Pivoting (BPA)
  ───────────────────
  An imaginary ball of radius r is rolled across the point cloud surface.
  A triangle is created only when the ball simultaneously touches three
  points. In an unpopulated passage, the ball cannot find a third contact
  point and simply stops rolling — the gap is preserved implicitly.

Both are followed by an explicit edge-length filter (configurable) as a
second line of defence: any triangle whose longest edge exceeds
max_edge_length is removed. This handles the edge case where a large alpha
or ball radius might just barely bridge a tight squeeze.
"""

import numpy as np
import open3d as o3d


# ── edge-length filter ────────────────────────────────────────────────────────

def filter_long_edges(
    mesh: o3d.geometry.TriangleMesh,
    max_edge_length: float,
) -> o3d.geometry.TriangleMesh:
    """
    Remove triangles whose longest edge exceeds max_edge_length.

    This is the explicit safety net against gap bridging. Operates purely on
    triangle vertex indices and positions — no normals required.

    Strategy
    --------
    For each triangle (i, j, k), compute the three edge lengths and keep the
    triangle only if max(e_ij, e_jk, e_ki) <= max_edge_length.

    The filter is intentionally strict: it is better to have a small hole in
    a wall than to silently seal a speleologically important passage.
    """
    assert max_edge_length > 0.0, "max_edge_length must be positive"

    vertices = np.asarray(mesh.vertices, dtype=np.float64)   # (V, 3)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)   # (T, 3)

    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert triangles.ndim == 2 and triangles.shape[1] == 3

    v0 = vertices[triangles[:, 0]]   # (T, 3)
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # Squared edge lengths — avoid sqrt until comparison threshold is needed.
    max_edge_sq = max_edge_length ** 2
    e01_sq = np.sum((v0 - v1) ** 2, axis=1)
    e12_sq = np.sum((v1 - v2) ** 2, axis=1)
    e20_sq = np.sum((v2 - v0) ** 2, axis=1)

    keep_mask = (
        (e01_sq <= max_edge_sq) &
        (e12_sq <= max_edge_sq) &
        (e20_sq <= max_edge_sq)
    )

    filtered = mesh.select_by_index(
        np.where(keep_mask)[0].tolist(),
        cleanup=False,   # do not remove unreferenced vertices yet
    )

    # Remove vertices no longer referenced by any triangle.
    filtered.remove_unreferenced_vertices()

    n_removed = int((~keep_mask).sum())
    print(
        f"[reconstruct] Edge filter (max={max_edge_length} m): "
        f"removed {n_removed:,} triangles, "
        f"{int(keep_mask.sum()):,} retained"
    )
    return filtered


# ── reconstruction algorithms ─────────────────────────────────────────────────

def reconstruct_alpha_shape(
    xyz: np.ndarray,
    alpha: float,
    edge_filter_cfg: dict,
    smoothing_cfg: dict
) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct mesh via 3D Alpha Shapes.

    Does NOT require normals — the algorithm works purely on point geometry.
    Open3D's implementation uses a Delaunay tetrahedralization internally and
    retains only simplices whose circumsphere radius ≤ alpha.

    Gap preservation
    ----------------
    A narrow cave squeeze contains no points inside it. Any triangle that
    would span the squeeze has a large circumsphere (much larger than the
    wall thickness). With alpha set to approximately the wall point spacing,
    such triangles are never included. The gap remains topologically open.

    Parameters
    ----------
    xyz             : (N, 3) float32 downsampled point positions
    alpha           : circumsphere radius threshold in metres
    edge_filter_cfg : dict with keys 'enabled' (bool) and 'max_edge_length'
    """
    assert alpha > 0.0, "alpha must be a positive float"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    print(f"[reconstruct] Running Alpha Shapes (alpha={alpha} m) …")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=alpha
    )

    n_tri = len(mesh.triangles)
    assert n_tri > 0, (
        f"Alpha Shapes produced an empty mesh with alpha={alpha}. "
        "Increase alpha in config.yaml or check point cloud quality."
    )
    print(f"[reconstruct] Alpha Shapes: {n_tri:,} triangles before filtering")

    if edge_filter_cfg["enabled"]:
        mesh = filter_long_edges(mesh, edge_filter_cfg["max_edge_length"])

    assert len(mesh.triangles) > 0, (
        "Mesh is empty after edge filtering — max_edge_length is too small "
        "relative to alpha. Relax edge_filter.max_edge_length in config.yaml."
    )

    if smoothing_cfg.get("enabled", False):
        iters = smoothing_cfg.get("iterations", 15)
        print(f"[reconstruct] Applying Taubin smoothing ({iters} iterations) …")
        mesh = mesh.filter_smooth_taubin(number_of_iterations=iters)

    mesh.compute_vertex_normals()
    return mesh


def reconstruct_ball_pivoting(
    pcd: o3d.geometry.PointCloud,
    radii: list[float],
    edge_filter_cfg: dict,
) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct mesh via Ball Pivoting Algorithm (BPA).

    Requires an Open3D PointCloud with pre-computed normals.

    Gap preservation
    ----------------
    The ball of radius r can only form a triangle when it simultaneously
    touches three points. An empty squeeze or passage has no point for the
    ball to pivot to — reconstruction simply stops at the passage boundary.
    Providing multiple radii allows the algorithm to fill gaps at different
    scales while still respecting the absence of points in passages.

    Parameters
    ----------
    pcd             : Open3D PointCloud with normals
    radii           : list of ball radii in metres (ascending order preferred)
    edge_filter_cfg : dict with keys 'enabled' (bool) and 'max_edge_length'
    """
    assert len(pcd.normals) > 0, (
        "Ball Pivoting requires normals. "
        "Set reconstruction.algorithm=ball_pivoting and ensure normals are "
        "estimated in the preprocess stage."
    )
    assert len(radii) > 0, "radii list must not be empty"
    assert all(r > 0 for r in radii), "All ball radii must be positive"

    o3d_radii = o3d.utility.DoubleVector(sorted(radii))

    print(f"[reconstruct] Running Ball Pivoting (radii={sorted(radii)} m) …")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d_radii
    )

    n_tri = len(mesh.triangles)
    assert n_tri > 0, (
        "Ball Pivoting produced an empty mesh. "
        "Check that normals are correctly oriented and radii are appropriate "
        "for the point cloud density."
    )
    print(f"[reconstruct] Ball Pivoting: {n_tri:,} triangles before filtering")

    if edge_filter_cfg["enabled"]:
        mesh = filter_long_edges(mesh, edge_filter_cfg["max_edge_length"])

    assert len(mesh.triangles) > 0, (
        "Mesh is empty after edge filtering — max_edge_length too restrictive."
    )

    mesh.compute_vertex_normals()
    return mesh


# ── dispatcher ────────────────────────────────────────────────────────────────

def reconstruct(
    xyz: np.ndarray,
    pcd_with_normals: o3d.geometry.PointCloud | None,
    cfg: dict,
) -> o3d.geometry.TriangleMesh:
    """
    Dispatch to the configured reconstruction algorithm.

    Parameters
    ----------
    xyz               : (N, 3) float32 downsampled points
    pcd_with_normals  : Open3D PointCloud with normals (required for BPA,
                        pass None for Alpha Shapes)
    cfg               : full config dict loaded from config.yaml

    Returns
    -------
    mesh : reconstructed and edge-filtered TriangleMesh
    """
    algorithm = cfg["reconstruction"]["algorithm"]
    edge_filter_cfg = cfg["edge_filter"]
    smoothing_cfg = cfg["smoothing"]

    if algorithm == "alpha_shape":
        alpha = float(cfg["reconstruction"]["alpha_shape"]["alpha"])
        return reconstruct_alpha_shape(xyz, alpha, edge_filter_cfg, smoothing_cfg)

    if algorithm == "ball_pivoting":
        assert pcd_with_normals is not None, (
            "Ball Pivoting requires a PointCloud with normals; "
            "got None. Check preprocess stage."
        )
        radii = [float(r) for r in cfg["reconstruction"]["ball_pivoting"]["radii"]]
        return reconstruct_ball_pivoting(pcd_with_normals, radii, edge_filter_cfg)

    # Any other string in config.yaml is a hard misconfiguration.
    raise ValueError(
        f"Unknown reconstruction algorithm '{algorithm}'. "
        "Valid options: 'alpha_shape', 'ball_pivoting'"
    )
