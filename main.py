"""
main.py – Cave LiDAR mesh reconstruction pipeline.
PATCH v1.1 — explicit cross-stage C++ memory teardown.

Usage
-----
    python main.py --file Proba1.laz
"""

import argparse
import gc
import os
import sys

import yaml
import open3d as o3d

from pipeline.ingest import load_point_cloud
from pipeline.preprocess import voxel_downsample, estimate_normals
from pipeline.reconstruct import reconstruct
from pipeline.visualize import save_mesh, show_mesh


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    assert isinstance(cfg, dict), "config.yaml must be a YAML mapping at root"
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a 3D cave mesh from a .laz point cloud."
    )
    parser.add_argument("--file", required=True,
                        help="Name of the .laz file inside scans/")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    return parser.parse_args()


def stage_ingest(laz_path: str) -> tuple:
    assert os.path.isfile(laz_path), f"Input file not found: '{laz_path}'"
    xyz, intensity = load_point_cloud(laz_path)
    return xyz, intensity


def stage_preprocess(xyz: "np.ndarray", cfg: dict):
    """
    Downsample, release the 26M raw array, then conditionally estimate normals.

    PATCH v1.1 memory sequence
    --------------------------
    Step A: voxel_downsample() builds a temporary Open3D cloud from xyz,
            downsamples it, extracts a float32 numpy array, and internally
            wipes the C++ buffer before returning. It does NOT free xyz.

    Step B: After downsampling, xyz (26.5M × 12 B = 318 MB) and intensity
            are no longer needed. We delete them here — at the stage boundary
            — so they are freed before normal estimation allocates the KD-tree.

    Step C: estimate_normals() returns a PointCloud with the KD-tree and KNN
            index still attached. We return it as-is; the teardown happens
            in stage_reconstruct, immediately before the heavy mesh allocation.
    """
    import numpy as np

    ds_cfg = cfg["downsampling"]
    xyz_down = voxel_downsample(xyz, voxel_size=float(ds_cfg["voxel_size"]))

    # ── PATCH Step B: release raw 26M-point buffer ─────────────────────────
    # xyz and intensity are not needed past this point. Explicitly delete
    # them here so ~370 MB (xyz float32 + intensity uint8) are freed before
    # the KD-tree allocation in estimate_normals().
    del xyz
    gc.collect()
    # ── end PATCH ──────────────────────────────────────────────────────────

    algorithm = cfg["reconstruction"]["algorithm"]

    if algorithm == "ball_pivoting":
        n_cfg = cfg["normals"]
        pcd_with_normals = estimate_normals(
            xyz_down,
            knn=int(n_cfg["knn"]),
            orient_toward_origin=bool(n_cfg["orient_toward_origin"]),
        )
        return xyz_down, pcd_with_normals

    return xyz_down, None


def stage_reconstruct(
    xyz_down: "np.ndarray",
    pcd_with_normals: "o3d.geometry.PointCloud | None",
    cfg: dict,
) -> "o3d.geometry.TriangleMesh":
    """
    PATCH v1.1 — tear down the PointCloud C++ buffers before reconstruction.

    Why this ordering matters
    -------------------------
    After normal estimation, the PointCloud object holds three simultaneous
    C++ allocations:
      1. points vector  (M × 3 × 8 B float64, e.g. ~96 MB for 4M points)
      2. normals vector (M × 3 × 8 B float64, another ~96 MB)
      3. KD-tree + KNN index (proportional to M, typically 50–150 MB)

    Surface reconstruction (Alpha Shapes or BPA) needs to allocate its own
    Delaunay / tetrahedralisation structures, which can spike to 2–4× the
    input size. Running both simultaneously on a 12 GB Colab instance causes
    OOM.

    The fix: wipe the C++ vectors explicitly BEFORE calling reconstruct().
    Assigning an empty Vector3dVector() calls the destructor of the old
    std::vector immediately (C++ RAII), releasing the pages before the
    reconstruct call begins its own allocation.

    Note: for Ball Pivoting, pcd_with_normals is passed INTO reconstruct()
    and must remain valid for that call. We only tear it down AFTER the mesh
    is returned.
    """
    algorithm = cfg["reconstruction"]["algorithm"]

    if algorithm == "alpha_shape":
        # Alpha Shapes does not need pcd_with_normals at all.
        # pcd_with_normals is None on this branch (asserted in reconstruct()).
        assert pcd_with_normals is None

        # ── PATCH: no pcd to tear down; just collect before heavy alloc ────
        gc.collect()
        # ── end PATCH ──────────────────────────────────────────────────────

        mesh = reconstruct(xyz_down, None, cfg)

    else:
        # Ball Pivoting: pcd_with_normals must stay live during reconstruct().
        assert pcd_with_normals is not None

        mesh = reconstruct(xyz_down, pcd_with_normals, cfg)

        # ── PATCH: tear down PointCloud AFTER BPA returns the mesh ─────────
        # The mesh is now fully built and stored in `mesh`. The PointCloud
        # (points + normals + KD-tree) is no longer referenced by anyone.
        # Wipe its C++ storage explicitly before it is garbage-collected.
        pcd_with_normals.points  = o3d.utility.Vector3dVector()
        pcd_with_normals.normals = o3d.utility.Vector3dVector()
        del pcd_with_normals
        gc.collect()
        # ── end PATCH ──────────────────────────────────────────────────────

    n_verts = len(mesh.vertices)
    n_tris  = len(mesh.triangles)
    print(f"[main] Final mesh: {n_verts:,} vertices, {n_tris:,} triangles")
    return mesh


def stage_output(mesh, cfg: dict, laz_path: str) -> None:
    out_cfg = cfg["output"]
    if out_cfg["save_mesh"]:
        save_mesh(mesh, out_cfg["mesh_filename"])
    window_title = f"Cave Mesh – {os.path.basename(laz_path)}"
    show_mesh(mesh, window_title)


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    laz_path = os.path.join("scans", args.file)

    xyz, intensity         = stage_ingest(laz_path)
    # intensity is not used downstream; free it at the earliest opportunity.
    del intensity
    gc.collect()

    xyz_down, pcd_normals  = stage_preprocess(xyz, cfg)
    # xyz is deleted inside stage_preprocess; do not reference it here.

    mesh                   = stage_reconstruct(xyz_down, pcd_normals, cfg)
    del xyz_down
    gc.collect()

    stage_output(mesh, cfg, laz_path)


if __name__ == "__main__":
    main()
