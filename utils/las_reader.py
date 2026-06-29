"""
Raw point ingestion from a .laz file
"""

import laspy
import numpy as np


class LasReader:
    """Universal reader for both point cloud and traectory"""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.las = None
    

    def __enter__(self):
        print(f"[las_reader] entered reader for file {self.filepath}")

        self.las = laspy.read(self.filepath)
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"[las_reader] exited reader for file {self.filepath}")
        del self.las


    def get_xyz(self):
        """
        Extracts coordinates

        Returns
        -------
        xyz : np.ndarray, shape (N, 3), dtype float32
        """
        xyz = np.empty((self.las.header.point_count, 3), dtype=np.int32)

        xyz[:, 0] = np.array(self.las.X, dtype=np.int32)
        xyz[:, 1] = np.array(self.las.Y, dtype=np.int32)
        xyz[:, 2] = np.array(self.las.Z, dtype=np.int32)

        self.scale  = np.float32(self.las.header.scale)
        self.offset = np.float32(self.las.header.offset)

        print("[las_reader] scale readed:", self.scale)
        print("[las_reader] offset readed:", self.offset)
        print("[las_reader] dtype:", xyz.dtype)
        print(f"[las_reader] points loaded: {len(xyz):,}")

        return xyz 
    
    
    def get_intensity(self):
        """
        Returns
        -------
        intensity : np.ndarray, shape (N,), dtype uint8
        """
        # Intensity: uint16 in LAS PF1, but EDA confirms values <= 255.
        # Cast to uint8 immediately — saves 26.5M bytes vs keeping uint16.
        raw_intensity = self.las.intensity

        print("[las_reader] intensity readed")

        print(f"[las_reader] intensity original dtype: {raw_intensity.dtype}")
        print(f"[las_reader] intensity min: {raw_intensity.min()}")
        print(f"[las_reader] intensity max: {raw_intensity.max()}")

        intensity = np.array(raw_intensity, dtype=np.uint8)
        return intensity


    def get_gps_time(self):
        """
        Returns
        -------
        gps_time : np.ndarray, shape (N,), dtype uint8?
        """
        raw_gps_time = self.las.gps_time.view(np.uint64)

        print("[las_reader] gps_time readed")

        print(f"[las_reader] gps_time original dtype: {raw_gps_time.dtype}")
        print(f"[las_reader] gps_time min: {raw_gps_time.min()}")
        print(f"[las_reader] gps_time max: {raw_gps_time.max()}")

        # return intensity
