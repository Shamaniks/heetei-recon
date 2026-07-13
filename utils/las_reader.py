"""
Raw point ingestion from a .laz file with IMU absolute coord support
"""


import laspy
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import utils.imu_processor as ip


class LasReader:
    """Universal reader for both point cloud and trajectory. Extended for Mandeye IMU chunks."""
    def __init__(self, filepath: str, imu_path: str | None = None):
        self.filepath = filepath
        self.imu_path = Path(imu_path) if imu_path else None
        self.las = None
        self.scale = None
        self.offset = None


    def __enter__(self):
        print(f"[las_reader] entered reader for file {self.filepath}")
        self.las = laspy.read(self.filepath)

        self.scale = np.float32(self.las.header.scales)
        self.offset = np.float32(self.las.header.offsets)

        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"[las_reader] exited reader for file {self.filepath}")
        del self.las


    def get_xyz(self):
        """
        Extracts local coordinates (unchanged)
        """
        xyz = np.empty((self.las.header.point_count, 3), dtype=np.int32)
        xyz[:, 0] = np.array(self.las.X, dtype=np.int32)
        xyz[:, 1] = np.array(self.las.Y, dtype=np.int32)
        xyz[:, 2] = np.array(self.las.Z, dtype=np.int32)

        print("[las_reader] scale readed:", self.scale)
        print("[las_reader] offset readed:", self.offset)
        print("[las_reader] dtype:", xyz.dtype)
        print(f"[las_reader] points loaded: {len(xyz):,}")

        return xyz 

    
    def get_global_xyz(self, chunk_size: int = 1000000):
        """Returns absolute/global XYZ as int32 grid. No scale/offset applied."""
        assert self.imu_path is not None
        
        imu_times_ns, imu_metrics = ip.load_imu(self.imu_path)
        
        n_points = self.las.header.point_count
        assert hasattr(self.las, 'gps_time')
        
        global_xyz = np.empty((n_points, 3), dtype=np.int32)
        all_lidar_times_ns = self.get_gps_time()
        
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            
            xyz_local_int = np.empty((end - start, 3), dtype=np.int32)
            xyz_local_int[:, 0] = np.array(self.las.X[start:end], dtype=np.int32)
            xyz_local_int[:, 1] = np.array(self.las.Y[start:end], dtype=np.int32)
            xyz_local_int[:, 2] = np.array(self.las.Z[start:end], dtype=np.int32)
            
            xyz_local_meters = xyz_local_int.astype(np.float64) * self.scale
            query_times_ns = all_lidar_times_ns[start:end]
            
            poses = ip.interpolate_pose(imu_times_ns, imu_metrics, query_times_ns)
            
            translations_meters = poses[:, :3]
            quats_scipy = poses[:, 3:7]
            
            rotations = R.from_quat(quats_scipy)
            xyz_rotated_meters = rotations.apply(xyz_local_meters)
            
            xyz_rotated_int = np.round(xyz_rotated_meters / self.scale).astype(np.int32)
            translations_int = np.round(translations_meters / self.scale).astype(np.int32)
            
            global_xyz[start:end] = xyz_rotated_int + translations_int
        
        print(f"[las_reader] global XYZ (int32) reconstructed: {n_points:,} points")
        return global_xyz


    def get_intensity(self):
        raw_intensity = self.las.intensity
        print("[las_reader] intensity readed")
        print(f"[las_reader] intensity original dtype: {raw_intensity.dtype}")
        print(f"[las_reader] intensity min: {raw_intensity.min()}")
        print(f"[las_reader] intensity max: {raw_intensity.max()}")
        intensity = np.array(raw_intensity, dtype=np.uint8)
        return intensity


    def get_gps_time(self):
        raw_gps_time = self.las.gps_time
        print("[las_reader] gps_time readed")
        print(f"[las_reader] gps_time original dtype: {raw_gps_time.dtype}")
        print(f"[las_reader] gps_time min: {raw_gps_time.min():,}")
        print(f"[las_reader] gps_time max: {raw_gps_time.max():,}")
        assert np.all(raw_gps_time >= 0)
        gps_time_ns = np.round(raw_gps_time.astype(np.float64) * 1_000_000_000.0).astype(np.uint64)
        print(f"[las_reader] gps_time converted to uint64 ns, range [{gps_time_ns.min()}, {gps_time_ns.max()}]")
        return gps_time_ns
