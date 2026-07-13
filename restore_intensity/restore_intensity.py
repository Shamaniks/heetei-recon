"""
Restores real (approximately) intensity
$A(d) = \\frac{1}{K_c + K_l \\cdot d + K_q \\cdot d^2}$

Matching strategy
------------------
Each cloud point is matched to the trajectory sample recorded at the
nearest gps_time — NOT the spatially closest trajectory point (the
previous implementation used a 3D KD-tree over xyz_track for this).
A scanner can revisit the same physical location at a different moment
(e.g. a loop closure inside a cave); in that case the spatially nearest
trajectory sample can belong to an entirely different pass, giving the
wrong range-to-sensor for this specific point. Matching in time is what
makes "range to sensor at the moment this point was captured" correct.

gps_time is compared as uint64 nanoseconds (exact integer ordering — no
floating-point epsilon issues), produced by utils.las_reader.LasReader.
get_gps_time(). Differences are computed in int64 during the nearest-
value search: naive uint64 subtraction wraps around instead of going
negative, so operands are cast to signed first. Real gps_time-in-ns
magnitudes (~1e14-1e17, see get_gps_time()'s docstring) sit far below
int64's ~9.2e18 range, so this cast is safe for any real timestamp —
guarded by an assert below rather than just assumed.

xyz_cloud / xyz_track are still required in metric (scaled) coordinates:
the actual A(d) formula needs a real Euclidean distance in metres, and
lidar_max_range is a metres threshold. gps_time only changes WHICH
track point is used for that distance calculation, not the distance
calculation itself.
"""

import numpy as np


def _nearest_track_index(gps_time_track_i64: "np.ndarray", gps_time_query_i64: "np.ndarray") -> "np.ndarray":
    """
    Vectorized nearest-value lookup in a sorted 1D array (already int64).

    Returns, for each value in gps_time_query_i64, the index into
    gps_time_track_i64 of the closest value.
    """
    n_track = len(gps_time_track_i64)
    insert_pos = np.searchsorted(gps_time_track_i64, gps_time_query_i64, side="left")
    insert_pos = np.clip(insert_pos, 1, n_track - 1)

    left, right = insert_pos - 1, insert_pos
    left_diff = np.abs(gps_time_query_i64 - gps_time_track_i64[left])
    right_diff = np.abs(gps_time_track_i64[right] - gps_time_query_i64)

    return np.where(left_diff <= right_diff, left, right)


# Public API
def restore(
        xyz_cloud: "np.ndarray",
        xyz_track: "np.ndarray",
        gps_time_cloud: "np.ndarray",
        gps_time_track: "np.ndarray",
        intensity: "np.ndarray",
        chunk_size: int,
        lidar_max_range: float,
        restoration_exp: float,
    ) -> "np.ndarray":
    """
    Parameters
    ----------
    xyz_cloud       : (N, 3) float64 point positions, metres
    xyz_track       : (M, 3) float64 point positions, metres
    gps_time_cloud  : (N,)   uint64 nanoseconds
    gps_time_track  : (M,)   uint64 nanoseconds — must be non-decreasing
                              (trajectory samples are assumed time-ordered;
                              this is what makes the nearest-value search
                              below valid)
    intensity       : (N,)   uint8 intensity
    chunk_size      : point number processed each iteration from config
    """
    assert len(xyz_cloud) == len(gps_time_cloud) == len(intensity), (
        "xyz_cloud, gps_time_cloud, and intensity must have matching length"
    )
    assert len(xyz_track) == len(gps_time_track), (
        "xyz_track and gps_time_track must have matching length"
    )

    int64_max = np.iinfo(np.int64).max
    assert gps_time_track.max() < int64_max and gps_time_cloud.max() < int64_max, (
        "gps_time exceeds int64 range — cannot safely diff uint64 "
        "timestamps for nearest-value matching."
    )

    # Small array (trajectory poses, not the full cloud) — safe to cast
    # once, up front.
    gps_time_track_i64 = gps_time_track.astype(np.int64)
    assert np.all(np.diff(gps_time_track_i64) >= 0), (
        "gps_time_track must be non-decreasing — trajectory samples are "
        "assumed time-ordered for the nearest-value search to be valid."
    )

    n_points = len(xyz_cloud)
    adjusted_intensity = np.zeros(n_points, dtype=np.float32)

    for i in range(0, n_points, chunk_size):
        end_idx = min(i + chunk_size, n_points)

        # Cast only this chunk's timestamps to int64 — avoids duplicating
        # the full-resolution cloud's gps_time array (potentially tens of
        # millions of points) just to get signed-safe subtraction.
        query_i64 = gps_time_cloud[i:end_idx].astype(np.int64)
        matched_idx = _nearest_track_index(gps_time_track_i64, query_i64)
        matched_xyz = xyz_track[matched_idx]

        r_meters = np.linalg.norm(xyz_cloud[i:end_idx] - matched_xyz, axis=1)
        r_coef = r_meters ** restoration_exp

        chunk_intensity = intensity[i:end_idx].astype(np.float32) * r_coef
        chunk_intensity[r_meters > lidar_max_range] = 0.0

        adjusted_intensity[i:end_idx] = chunk_intensity

    return adjusted_intensity
