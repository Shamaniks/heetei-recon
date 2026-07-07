"""
main.py — Heetei cave LiDAR reconstruction pipeline.

Unified, linear pipeline orchestrator. Stages execute strictly in order;
each stage's output is the next stage's input — no graph, no branching
orchestration. Two ingestion modes are supported (config: ingestion.mode):

    chunks       Raw split Mandeye session (lidarXXXX.laz + imuXXXX.csv +
                 statusXXXX.json [+ lidarXXXX.sn]), discovered and merged
                 from a session directory. See ingest/chunk_reader.py.
    single_file  One pre-merged cloud .laz (legacy behaviour).

Usage
-----
    python main.py --config config.yaml
    python main.py --config config.yaml --session-dir /path/to/session
    python main.py --config config.yaml --cloud cloud.laz --track track.laz

Coordinate representation (read this before touching stage internals)
-----------------------------------------------------------------------
Two different existing modules disagree about what "xyz" means:

  - downsampling.max_pooling.voxel_downsample() bit-masks raw LAS grid
    coordinates directly (`xyz & mask`) — it REQUIRES int32.
  - restore_intensity.restore() computes a real Euclidean KD-tree
    distance and compares it against lidar_max_range in metres — it
    REQUIRES scaled, metric float coordinates.

Both are treated as fixed, pre-existing contracts (Task 3: integrate the
existing modules, not rewrite them). The pipeline therefore keeps the
full point cloud as int32 grid coordinates from ingest through
downsampling — this is the RAM-optimised representation and the one
downsampling needs — and produces a METRIC copy only transiently, right
before restore_intensity's KD-tree query, freeing it immediately after
(see stage_restore_intensity). The one PERSISTENT int32 -> metres
conversion happens once, after downsampling has already reduced the
point count (stage_to_metres), since that is the array Open3D actually
needs from then on.

RAM note: the transient full-resolution metric copy needed for
restore_intensity is the peak-memory moment of this pipeline (roughly
2x the int32 array's footprint, alive for one function call). This is a
real, unavoidable consequence of restore_intensity's pre-existing
distance-based contract operating before downsampling reduces the point
count; see the module docstring in restore_intensity/restore_intensity.py.

Scope boundary: trajectory computation
---------------------------------------
Raw IMU samples read from session chunks are exposed to the caller
(SessionArrays.imu_by_chunk) but are NOT integrated into a trajectory by
this pipeline. Producing a trajectory from raw LiDAR + IMU is a
LiDAR-Inertial-Odometry / SLAM problem (the role HDMapping plays upstream
of heetei-recon) and is out of scope for this refactor. restore_intensity
therefore requires a separately supplied, precomputed track_path in
config.yaml regardless of ingestion mode; if absent, restore_intensity is
skipped explicitly (not silently) and raw recorded intensity passes
through unchanged.
"""

import argparse
import gc
import os
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import yaml

from utils.las_reader import LasReader
from ingest.chunk_reader import load_session_arrays
from restore_intensity.restore_intensity import restore as restore_intensity_fn
from downsampling.max_pooling import voxel_downsample
from pipeline.normals import estimate_normals
from pipeline.reconstruct import reconstruct
from pipeline.output import save_mesh, show_mesh
from visualization.colored_points import (
    build_colored_point_cloud,
    save_colored_point_cloud,
    show_colored_point_cloud,
)


@dataclass
class IngestResult:
    xyz: np.ndarray                 # (N, 3) int32 — raw LAS grid coordinates
    intensity: np.ndarray           # (N,)   uint8 — raw recorded intensity
    scale: np.ndarray               # (3,)   float32 — cloud's LAS header scale
    offset: np.ndarray              # (3,)   float32 — cloud's LAS header offset
    xyz_track: "np.ndarray | None"  # (M, 3) int32, or None if no track_path
    track_scale: "np.ndarray | None"
    track_offset: "np.ndarray | None"


def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    assert isinstance(cfg, dict), "config.yaml must be a YAML mapping at root"
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a 3D cave mesh from Mandeye LiDAR data."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--session-dir", default=None, help="Override ingestion.session_dir (mode: chunks)")
    parser.add_argument("--cloud", default=None, help="Override ingestion.cloud_path (mode: single_file)")
    parser.add_argument("--track", default=None, help="Override ingestion.track_path (either mode)")
    return parser.parse_args()


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    ing = cfg["ingestion"]
    if args.session_dir is not None:
        ing["session_dir"] = args.session_dir
    if args.cloud is not None:
        ing["cloud_path"] = args.cloud
    if args.track is not None:
        ing["track_path"] = args.track
    return cfg


# ── Stage 1: ingest ───────────────────────────────────────────────────────────

def stage_ingest(cfg: dict) -> IngestResult:
    ing = cfg["ingestion"]
    mode = ing["mode"]

    if mode == "chunks":
        session_dir = ing["session_dir"]
        assert session_dir is not None, "ingestion.session_dir is required when mode: chunks"
        sa = load_session_arrays(
            session_dir,
            chunk_glob=ing["chunk_glob"],
            require_imu=bool(ing["require_imu"]),
            require_status=bool(ing["require_status"]),
        )
        if sa.skipped_chunks:
            print(f"[ingest] {len(sa.skipped_chunks)} chunk(s) excluded: {sa.skipped_chunks}")
        xyz, intensity, scale, offset = sa.xyz, sa.intensity, sa.scale, sa.offset
        print(f"[ingest] Loaded {xyz.shape[0]:,} points from "
              f"{len(sa.imu_by_chunk)} session chunk(s) in '{session_dir}'")

    elif mode == "single_file":
        cloud_path = ing["cloud_path"]
        assert cloud_path is not None, "ingestion.cloud_path is required when mode: single_file"
        assert os.path.isfile(cloud_path), f"Input file not found: '{cloud_path}'"
        with LasReader(cloud_path) as lr:
            xyz, intensity, scale, offset = lr.get_xyz(), lr.get_intensity(), lr.scale, lr.offset
        print(f"[ingest] Loaded {xyz.shape[0]:,} points from '{cloud_path}'")

    else:
        raise ValueError(f"Unknown ingestion.mode '{mode}'. Valid options: 'chunks', 'single_file'")

    xyz_track = track_scale = track_offset = None
    track_path = ing["track_path"]
    if track_path is not None:
        assert os.path.isfile(track_path), f"Trajectory file not found: '{track_path}'"
        with LasReader(track_path) as lr:
            xyz_track, track_scale, track_offset = lr.get_xyz(), lr.scale, lr.offset
        print(f"[ingest] Loaded {xyz_track.shape[0]:,} trajectory points from '{track_path}'")

    return IngestResult(xyz, intensity, scale, offset, xyz_track, track_scale, track_offset)


# ── Shared: int32 grid -> metric float64 ──────────────────────────────────────

def stage_to_metres(xyz_grid: np.ndarray, scale: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    The only place raw int32 grid coordinates become real-world metres.
    Open3D requires float64 points regardless, so float64 is used both
    for this stage's persistent (post-downsampling) output and for the
    transient full-resolution copy restore_intensity needs — see the
    module docstring for why a transient full-resolution copy is
    unavoidable given restore_intensity's pre-existing contract.
    """
    return xyz_grid.astype(np.float64) * scale.astype(np.float64) + offset.astype(np.float64)


# ── Stage 2: restore intensity ────────────────────────────────────────────────

def stage_restore_intensity(ingested: IngestResult, cfg: dict) -> np.ndarray:
    if not cfg["pipeline"]["restore_intensity"] or ingested.xyz_track is None:
        if cfg["pipeline"]["restore_intensity"] and ingested.xyz_track is None:
            print("[restore_intensity] SKIPPED: no track_path configured; "
                  "raw intensity passed through unchanged.")
        return ingested.intensity

    # Transient metric conversion for this KD-tree query only. Freed
    # immediately below; ingested.xyz (int32) is untouched and still owned
    # by the caller for the downsampling stage that follows.
    xyz_cloud_metres = stage_to_metres(ingested.xyz, ingested.scale, ingested.offset)
    xyz_track_metres = stage_to_metres(ingested.xyz_track, ingested.track_scale, ingested.track_offset)

    ri_cfg = cfg["restore_intensity"]
    adjusted = restore_intensity_fn(
        xyz_cloud=xyz_cloud_metres,
        xyz_track=xyz_track_metres,
        intensity=ingested.intensity,
        chunk_size=int(ri_cfg["chunk_size"]),
        lidar_max_range=float(ri_cfg["lidar_max_range"]),
        restoration_exp=float(ri_cfg["restoration_exp"]),
    )
    del xyz_cloud_metres, xyz_track_metres
    gc.collect()

    print(f"[restore_intensity] Restored intensity for {len(adjusted):,} points")
    return adjusted


# ── Stage 3: downsample ───────────────────────────────────────────────────────

def stage_downsample(xyz_grid: np.ndarray, intensity: np.ndarray, scale: np.ndarray, cfg: dict):
    if not cfg["pipeline"]["downsampling"]:
        return xyz_grid, intensity

    # binary_voxelisation divides voxel_size by a single scalar scale and
    # bit-masks all three axes with one shared mask — a real invariant of
    # that bitwise approach, not an incidental simplification.
    assert np.allclose(scale[0], scale[1]) and np.allclose(scale[1], scale[2]), (
        f"binary_voxelisation requires a uniform per-axis LAS scale; got {scale}"
    )

    ds_cfg = cfg["downsampling"]
    keep_indices = voxel_downsample(
        xyz_grid, intensity,
        voxel_size=float(ds_cfg["voxel_size"]),
        scale=float(scale[0]),
    )
    xyz_down = xyz_grid[keep_indices]
    intensity_down = intensity[keep_indices]
    print(f"[downsampling] {xyz_grid.shape[0]:,} -> {xyz_down.shape[0]:,} points")
    return xyz_down, intensity_down


# ── Stage 4: normals (Ball Pivoting only) ─────────────────────────────────────

def stage_normals(xyz_metres: np.ndarray, cfg: dict):
    if cfg["reconstruction"]["algorithm"] != "ball_pivoting":
        return None
    n_cfg = cfg["normals"]
    return estimate_normals(
        xyz_metres,
        knn=int(n_cfg["knn"]),
        orient_toward_origin=bool(n_cfg["orient_toward_origin"]),
    )


# ── Stage 5: reconstruct ──────────────────────────────────────────────────────

def stage_reconstruct(xyz_metres: np.ndarray, pcd_with_normals, cfg: dict):
    if not cfg["pipeline"]["reconstruction"]:
        return None

    algorithm = cfg["reconstruction"]["algorithm"]
    if algorithm == "alpha_shape":
        assert pcd_with_normals is None
        mesh = reconstruct(xyz_metres, None, cfg)
    else:
        assert pcd_with_normals is not None
        mesh = reconstruct(xyz_metres, pcd_with_normals, cfg)
        # PATCH v1.1 memory note (ported from the original preprocess
        # module): the KD-tree / KNN index built by estimate_normals lives
        # on the C++ heap and is not freed until points/normals are
        # explicitly wiped.
        pcd_with_normals.points = o3d.utility.Vector3dVector()
        pcd_with_normals.normals = o3d.utility.Vector3dVector()
        del pcd_with_normals
        gc.collect()

    print(f"[main] Final mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    return mesh


# ── Stage 6: visualization & output ───────────────────────────────────────────

def stage_output(mesh, xyz_metres: np.ndarray, intensity: np.ndarray, cfg: dict) -> None:
    vis_enabled = cfg["pipeline"]["visualization"]
    vis_cfg = cfg["visualization"]
    interactive = bool(vis_cfg["interactive"])

    if vis_enabled and vis_cfg["colored_points"]:
        pcd = build_colored_point_cloud(xyz_metres, intensity)
        save_colored_point_cloud(pcd, vis_cfg["point_cloud_filename"])
        if interactive:
            show_colored_point_cloud(pcd)

    if mesh is not None:
        out_cfg = cfg["output"]
        if out_cfg["save_mesh"]:
            save_mesh(mesh, out_cfg["mesh_filename"])
        if vis_enabled and interactive:
            show_mesh(mesh)


# ── Orchestrator ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = _apply_cli_overrides(load_config(args.config), args)

    ingested = stage_ingest(cfg)

    intensity = stage_restore_intensity(ingested, cfg)
    xyz = ingested.xyz
    scale, offset = ingested.scale, ingested.offset
    del ingested
    gc.collect()

    xyz_down, intensity_down = stage_downsample(xyz, intensity, scale, cfg)
    del xyz, intensity
    gc.collect()

    xyz_metres = stage_to_metres(xyz_down, scale, offset)
    del xyz_down
    gc.collect()

    pcd_with_normals = stage_normals(xyz_metres, cfg)
    mesh = stage_reconstruct(xyz_metres, pcd_with_normals, cfg)

    stage_output(mesh, xyz_metres, intensity_down, cfg)


if __name__ == "__main__":
    main()
