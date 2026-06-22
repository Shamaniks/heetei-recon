"""
Raw point ingestion from a .laz file.
"""

import laspy
import numpy as np


class LasReader:
    """ Universal reader for both point cloud and traectory """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.las = None
    

    def __enter__(self):
        self.las = laspy.read(self.filepath)
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
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

        return xyz 
    
    
    def get_intensity(self):
        """
        Returns
        -------
        intensity : np.ndarray, shape (N,), dtype uint8
        """
        # Intensity: uint16 in LAS PF1, but EDA confirms values ≤ 255.
        # Cast to uint8 immediately — saves 26.5M bytes vs keeping uint16.
        intensity = np.array(self.las.intensity, dtype=np.uint8)
        return intensity
