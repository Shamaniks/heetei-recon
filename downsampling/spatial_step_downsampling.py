import numpy as np


def spatial_step_downsampling(track_xyz, scale, step_meters=3.0):
    """
    Parameters
    ----------
    track_xyz:   np.ndarray (N, 3)
    scale:       np.ndarray (3,)
    step_meters: float

    Returns
    -------
    indices:     np.ndarray (N,)
    """
    station_indices = [0]
    last_idx = 0
    last_point = track_xyz[last_idx] * scale
    for i in range(1, len(track_xyz)):
        curr_point = track_xyz[i] * scale
        if np.linalg.norm(curr_point - last_point) >= step_meters:
            station_indices.append(i)
            last_idx = i
            last_point = track_xyz[last_idx] * scale

    last_track_idx = len(track_xyz) - 1
    if station_indices[-1] != last_track_idx:
        station_indices.append(last_track_idx)

    return station_indices
