"""
Raw multi-chunk session ingestion.

The Mandeye hardware controller (JanuszBedkowski/mandeye_controller,
code/save_data.cpp + code/main.cpp) writes one ~5 s recording chunk as a
quadruplet of sibling files sharing a zero-padded 4-digit index XXXX:

    lidarXXXX.laz    point cloud for this chunk          (required, anchor)
    imuXXXX.csv      IMU samples for this chunk          (required)
    statusXXXX.json  controller/system status snapshot   (required)
    lidarXXXX.sn     lidar serial-number manifest         (optional)

Verified on-disk formats (read from the controller's own source, since
this is real hardware, not a documented wire protocol):
  - imuXXXX.csv is whitespace-delimited (not comma-delimited) with one
    header line: "timestamp gyroX gyroY gyroZ accX accY accZ imuId
    timestampUnix".
  - lidarXXXX.sn is plain text ("<id> <serial>" per line), not binary.
    Per this task's ingestion contract it is still treated as an opaque,
    skip-safe sidecar (read as raw bytes, not parsed) since it carries no
    information needed for spatial alignment; the discrepancy with its
    true plain-text nature is noted here rather than silently assumed.

Pairing in this module is at the file/chunk level (shared XXXX index),
matching the ~5 s session-chunk granularity. Sub-chunk temporal alignment
between individual LiDAR points and individual IMU samples (e.g. via
gps_time) is a LiDAR-Inertial-Odometry concern and is explicitly out of
scope — see the "trajectory" note in main.py.

Trust boundary
--------------
A companion file's ABSENCE is an expected operational condition (the
recorder can be power-cycled or interrupted mid-chunk) and is handled
explicitly here: the chunk is excluded and the exclusion is logged. A
companion file's CONTENTS being malformed is not defensively checked —
upstream data is trusted, and a malformed file for a chunk that exists is
a genuine data/logic fault that must surface as a crash, not be masked.
"""

import glob
import json
import os
import re
import warnings
from dataclasses import dataclass

import laspy
import numpy as np

from utils.las_reader import LasReader


# ── IMU record layout ──────────────────────────────────────────────────────
# Matches the on-disk column order written by mandeye_controller's
# saveImuData(). Space-efficient dtypes: gyro/accel need only sensor-grade
# float32; the tick timestamp needs int64 range; timestamp_unix keeps
# float64 to preserve sub-millisecond precision at a ~1.7e9 s magnitude.
IMU_DTYPE = np.dtype([
    ("timestamp",      np.int64),
    ("gyro_x",         np.float32),
    ("gyro_y",         np.float32),
    ("gyro_z",         np.float32),
    ("acc_x",          np.float32),
    ("acc_y",          np.float32),
    ("acc_z",          np.float32),
    ("imu_id",         np.int16),
    ("timestamp_unix", np.float64),
])

_CHUNK_INDEX_RE = re.compile(r"lidar(\d{4})\.laz$")


@dataclass
class SessionChunk:
    """One paired, fully-loaded session chunk."""
    index: int
    xyz: np.ndarray            # (n, 3) int32  — raw LAS grid coordinates
    intensity: np.ndarray      # (n,)   uint8   — raw recorded intensity
    scale: np.ndarray          # (3,)   float32 — LAS header scale factors
    offset: np.ndarray         # (3,)   float32 — LAS header offsets
    imu: np.ndarray            # (m,)   IMU_DTYPE
    status: dict
    sn_bytes: object           # bytes, or None if lidarXXXX.sn is absent


@dataclass
class SessionArrays:
    """Aggregated arrays for a full session, ready for downstream stages."""
    xyz: np.ndarray             # (N, 3) int32
    intensity: np.ndarray       # (N,)   uint8
    scale: np.ndarray           # (3,)   float32 — shared across all chunks
    offset: np.ndarray          # (3,)   float32 — shared across all chunks
    imu_by_chunk: dict
    status_by_chunk: dict
    sn_by_chunk: dict
    skipped_chunks: list


def discover_chunk_indices(session_dir: str, chunk_glob: str = "lidar????.laz") -> list:
    """Find every lidarXXXX.laz in session_dir; return its sorted indices."""
    paths = sorted(glob.glob(os.path.join(session_dir, chunk_glob)))
    assert len(paths) > 0, (
        f"No chunk files matched '{chunk_glob}' in '{session_dir}'. "
        "Check ingestion.session_dir and ingestion.chunk_glob in config.yaml."
    )
    indices = []
    for p in paths:
        m = _CHUNK_INDEX_RE.search(os.path.basename(p))
        assert m is not None, (
            f"'{p}' matched chunk_glob but not the lidarXXXX.laz "
            "naming convention (4-digit zero-padded index required)."
        )
        indices.append(int(m.group(1)))
    return sorted(indices)


def _chunk_paths(session_dir: str, idx: int) -> dict:
    stem = f"{idx:04d}"
    return {
        "laz":    os.path.join(session_dir, f"lidar{stem}.laz"),
        "imu":    os.path.join(session_dir, f"imu{stem}.csv"),
        "status": os.path.join(session_dir, f"status{stem}.json"),
        "sn":     os.path.join(session_dir, f"lidar{stem}.sn"),
    }


def validate_chunk_indices(
        session_dir: str,
        indices: list,
        require_imu: bool = True,
        require_status: bool = True,
    ) -> tuple:
    """
    Split discovered indices into (usable, skipped) by companion presence.

    This is one explicit, logged policy decision made up front — not a
    try/except swallowed deep inside parsing code.
    """
    usable, skipped = [], []
    for idx in indices:
        paths = _chunk_paths(session_dir, idx)
        missing = []
        if require_imu and not os.path.isfile(paths["imu"]):
            missing.append("imu")
        if require_status and not os.path.isfile(paths["status"]):
            missing.append("status")

        if missing:
            print(f"[ingest] WARNING: chunk {idx:04d} is missing "
                  f"{'/'.join(missing)} sidecar(s) — excluded from session.")
            skipped.append(idx)
        else:
            usable.append(idx)

    assert len(usable) > 0, (
        f"All {len(indices)} discovered chunk(s) in '{session_dir}' are "
        "incomplete — nothing left to process."
    )
    return usable, skipped


def _read_imu_csv(path: str) -> np.ndarray:
    """Parse an imuXXXX.csv sidecar (whitespace-delimited, 1 header line)."""
    with warnings.catch_warnings():
        # An empty IMU buffer (chunk boundary with zero samples) is a
        # valid recording outcome; genfromtxt's "Empty input file"
        # UserWarning for that case is expected noise, not a fault.
        warnings.simplefilter("ignore", UserWarning)
        return np.genfromtxt(path, dtype=IMU_DTYPE, skip_header=1, ndmin=1)


def _read_sn_bytes(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        return fh.read()


def _load_one_chunk(session_dir: str, idx: int) -> SessionChunk:
    """Load single chunk with global coords."""
    paths = _chunk_paths(session_dir, idx)
    imu_path = paths["imu"]
    with LasReader(paths["laz"], imu_path=imu_path) as lr:
        xyz = lr.get_global_xyz()  
        intensity = lr.get_intensity()
        scale, offset = lr.scale, lr.offset

    return SessionChunk(
        index=idx,
        xyz=xyz,
        intensity=intensity,
        scale=scale,
        offset=offset,
        imu=_read_imu_csv(paths["imu"]),
        status=json.load(open(paths["status"], "r")),
        sn_bytes=_read_sn_bytes(paths["sn"]),
    )


def iter_session_chunks(session_dir: str, indices: list):
    """
    Stream already-validated chunks one at a time, in index order.

    This is the actual "structured stream of NumPy arrays" contract: each
    yielded SessionChunk is immediately usable on its own (e.g. for a
    future incremental consumer). load_session_arrays() below is a
    convenience aggregator built on top of this same generator for
    downstream stages that require whole-session arrays.
    """
    for idx in indices:
        yield _load_one_chunk(session_dir, idx)


def load_session_arrays(
        session_dir: str,
        chunk_glob: str = "lidar????.laz",
        require_imu: bool = True,
        require_status: bool = True,
    ) -> SessionArrays:
    """
    Discover, validate, and load a full session directory into aggregated,
    preallocated arrays.

    Memory strategy
    ---------------
    Each chunk's .laz header is opened once via laspy.open() (lazy reader,
    header-only — does not decompress point data) purely to read
    point_count, so the combined xyz/intensity arrays are allocated exactly
    once at their true final size. This avoids per-chunk list-append +
    concatenate, which would transiently double peak memory.
    """
    all_indices = discover_chunk_indices(session_dir, chunk_glob)
    usable, skipped = validate_chunk_indices(session_dir, all_indices, require_imu, require_status)

    counts = {}
    for idx in usable:
        with laspy.open(_chunk_paths(session_dir, idx)["laz"]) as reader:
            counts[idx] = reader.header.point_count
    total_points = sum(counts.values())

    xyz = np.empty((total_points, 3), dtype=np.int32)
    intensity = np.empty(total_points, dtype=np.uint8)
    imu_by_chunk, status_by_chunk, sn_by_chunk = {}, {}, {}
    scale_ref, offset_ref = None, None

    write_pos = 0
    for chunk in iter_session_chunks(session_dir, usable):
        n = counts[chunk.index]
        xyz[write_pos:write_pos + n] = chunk.xyz
        intensity[write_pos:write_pos + n] = chunk.intensity

        if scale_ref is None:
            scale_ref, offset_ref = chunk.scale, chunk.offset
        else:
            assert np.allclose(chunk.scale, scale_ref), (
                f"Chunk {chunk.index:04d} LAS scale {chunk.scale} differs "
                f"from the session's reference scale {scale_ref}. Mixed-"
                "scale sessions cannot be safely merged as raw int32 grid "
                "coordinates."
            )
            assert np.allclose(chunk.offset, offset_ref), (
                f"Chunk {chunk.index:04d} LAS offset {chunk.offset} differs "
                f"from the session's reference offset {offset_ref}."
            )

        imu_by_chunk[chunk.index] = chunk.imu
        status_by_chunk[chunk.index] = chunk.status
        sn_by_chunk[chunk.index] = chunk.sn_bytes
        write_pos += n

    assert write_pos == total_points, "Internal offset bookkeeping mismatch"

    return SessionArrays(
        xyz=xyz,
        intensity=intensity,
        scale=scale_ref,
        offset=offset_ref,
        imu_by_chunk=imu_by_chunk,
        status_by_chunk=status_by_chunk,
        sn_by_chunk=sn_by_chunk,
        skipped_chunks=skipped,
    )
