import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def load_imu(imu_path: Path):
    """Load IMU / SLAM trajectory, returning time and metrics separately with their native dtypes."""
    dt = np.dtype([('time', np.int64), ('metrics', np.float64, (8, ))])
    structured_data = np.loadtxt(imu_path, delimiter=' ', skiprows=1, dtype=dt)
    return structured_data['time'].copy(), structured_data['metrics'].copy()


def clean_imu_times(imu_data: np.ndarray) -> np.ndarray:
    """Remove non-strictly-increasing timestamps."""
    times = imu_data[:, 0]
    valid_mask = np.diff(times) > 0
    if not np.all(valid_mask):
        valid_indices = np.hstack(([True], valid_mask))
        imu_data = imu_data[valid_indices].copy()
    return imu_data


def interpolate_pose(
        imu_times_ns: np.ndarray, 
        imu_metrics: np.ndarray, 
        query_times_ns: np.ndarray, 
        time_offset_sec: float = 0.0
    ) -> np.ndarray:
    """Interpolate position (linear) + orientation (Slerp) with clamping."""
    valid_mask = np.diff(imu_times_ns) > 0
    if not np.all(valid_mask):
        valid_indices = np.hstack(([True], valid_mask))
        imu_times_ns = imu_times_ns[valid_indices]
        imu_metrics = imu_metrics[valid_indices]
        
    offset_ns = int(time_offset_sec * 1_000_000_000)
    corrected_query_ns = query_times_ns.astype(np.int64) + offset_ns
    
    min_t_ns, max_t_ns = imu_times_ns.min(), imu_times_ns.max()
    query_clamped_ns = np.clip(corrected_query_ns, min_t_ns, max_t_ns)
    
    pos_interp = np.zeros((len(query_times_ns), 3), dtype=np.float64)
    for i in range(3):
        pos_interp[:, i] = np.interp(query_clamped_ns, imu_times_ns, imu_metrics[:, i])
   
    raw_w = imu_metrics[:, 3]
    raw_xyz = imu_metrics[:, 4:7]
    scipy_quats = np.hstack([raw_xyz, raw_w.reshape(-1, 1)])
   
    rotations = R.from_quat(scipy_quats)
    slerp_operator = Slerp(imu_times_ns, rotations)
    interp_rotations = slerp_operator(query_clamped_ns)
    interp_quats_scipy = interp_rotations.as_quat()
   
    return np.hstack([pos_interp, interp_quats_scipy])


def transform_local_to_global(
    points_local_meters: np.ndarray, 
    poses: np.ndarray, 
    las_scale: np.ndarray = None  # Больше не используем внутри для масштабирования точек
) -> np.ndarray:
    """
    Apply batched rigid transform.
    Оба массива (points_local_meters и poses) уже ДОЛЖНЫ быть в метрах (float64).
    """
    assert points_local_meters.shape[0] == poses.shape[0]
    assert points_local_meters.shape[1] == 3
    assert poses.shape[1] == 7

    # Траектория в метрах
    translations_meters = poses[:, :3]

    # Кватернион из интерполяции [X, Y, Z, W]
    quats_scipy = poses[:, 3:7]
    rotations = R.from_quat(quats_scipy)
    
    # Вращаем чистые метры и добавляем чистые метры траектории
    points_global_meters = rotations.apply(points_local_meters) + translations_meters
    
    return points_global_meters
