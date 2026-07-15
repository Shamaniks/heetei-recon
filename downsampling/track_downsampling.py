import numpy as np


def track_downsampling(track_xyz, step_meters=3.0):
    """
    Parameters
    ----------
    track_xyz:   np.ndarray (N, 3)
    step_meters: float

    Returns
    -------
    indices:     np.ndarray (N,)
    """
    station_indices = [0]
    last_idx = 0
    for i in range(1, len(track_xyz)):
        if np.linalg.norm(track_xyz[i] - track_xyz[last_idx]) >= step_meters:
            station_indices.append(i)
            last_idx = i

    print(f"[track_downsampling] extracted stations: {len(station_indices):,}")
    return station_indices
