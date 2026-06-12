"""
main.py – Cave LiDAR mesh reconstruction pipeline.

Usage
-----
    python main.py --file Proba1.laz

The target .laz file must reside inside the scans/ directory.
All tunable parameters live in config.yaml — no hardcoded math here.
"""

import argparse
import os
import sys

import yaml

from pipeline.ingest import load_point_cloud
from pipeline.preprocess import voxel_downsample, estimate_normals
from pipeline.reconstruct import reconstruct
from pipeline.visualize import save_mesh, show_mesh


# ── config loading ────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    assert isinstance(cfg, dict), "config.yaml must be a YAML mapping at root"
    return cfg


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a 3D cave mesh from a .laz point cloud."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Name of the .laz file inside the scans/ directory "
             "(e.g. Proba1.laz)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    return parser.parse_args()


# ── pipeline stages ───────────────────────────────────────────────────────────

def stage_ingest(laz_path: str):
    """Load raw points from disk."""
    assert os.path.isfile(laz_path), f"Input file not found: '{laz_path}'"
    xyz, intensity = load_point_cloud(laz_path)
    return xyz, intensity


def stage_preprocess(xyz, cfg: dict):
    """
    Downsample, then conditionally estimate normals.

    Normals are only needed for Ball Pivoting. Skipping them for Alpha Shapes
    saves both time and memory on 26.5M-point clouds.
    """
    ds_cfg = cfg["downsampling"]
    xyz_down = voxel_downsample(xyz, voxel_size=float(ds_cfg["voxel_size"]))

    algorithm = cfg["reconstruction"]["algorithm"]

    if algorithm == "ball_pivoting":
        n_cfg = cfg["normals"]
        pcd_with_normals, _ = estimate_normals(
            xyz_down,
            knn=int(n_cfg["knn"]),
            orient_toward_origin=bool(n_cfg["orient_toward_origin"]),
        )
        return xyz_down, pcd_with_normals

    # Alpha Shapes path — no normals needed.
    return xyz_down, None


def stage_reconstruct(xyz_down, pcd_with_normals, cfg: dict):
    """Dispatch to the configured reconstruction algorithm."""
    mesh = reconstruct(xyz_down, pcd_with_normals, cfg)

    n_verts = len(mesh.vertices)
    n_tris  = len(mesh.triangles)
    print(f"[main] Final mesh: {n_verts:,} vertices, {n_tris:,} triangles")
    return mesh


def stage_output(mesh, cfg: dict, laz_path: str) -> None:
    """Save and/or display the mesh."""
    out_cfg = cfg["output"]

    if out_cfg["save_mesh"]:
        save_mesh(mesh, out_cfg["mesh_filename"])

    window_title = f"Cave Mesh – {os.path.basename(laz_path)}"
    show_mesh(mesh, window_title)


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    laz_path = os.path.join("scans", args.file)

    xyz, _intensity        = stage_ingest(laz_path)
    xyz_down, pcd_normals  = stage_preprocess(xyz, cfg)
    mesh                   = stage_reconstruct(xyz_down, pcd_normals, cfg)
    stage_output(mesh, cfg, laz_path)


if __name__ == "__main__":
    main()
