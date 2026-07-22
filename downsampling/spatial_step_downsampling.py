import numpy as np


def spatial_step_downsampling(
        xyz:   np.ndarray, 
        scale: np.ndarray, 
        step_meters=3.0
    ):
    """
    Parameters
    ----------
    xyz:         np.ndarray (N, 3)
    scale:       np.ndarray (3,)
    step_meters: float

    Returns
    -------
    indices:     np.ndarray (N,)
    """
    station_indices = [0]
    last_idx = 0
    last_point = xyz[last_idx] * scale
    for i in range(1, len(xyz)):
        curr_point = xyz[i] * scale
        if np.linalg.norm(curr_point - last_point) >= step_meters:
            station_indices.append(i)
            last_idx = i
            last_point = xyz[last_idx] * scale

    last_idx = len(xyz) - 1
    if station_indices[-1] != last_idx:
        station_indices.append(last_idx)

    return station_indices
