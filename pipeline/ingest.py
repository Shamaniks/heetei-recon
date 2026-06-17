"""
Stage 1 – Raw point ingestion from a .laz file.
"""

import laspy
import numpy as np


class LazReader:
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
        Extracts coordinates and casts to float32

        Returns
        -------
        xyz : np.ndarray, shape (N, 3), dtype float32
        """
        scale  = np.float32(self.las.header.scale)    # shape (3,) after float32 cast
        offset = np.float32(self.las.header.offset)   # shape (3,)

        xyz = np.empty((self.las.header.point_count, 3), dtype=np.float32)

        xyz[:, 0] = np.array(self.las.X, dtype=np.float32) * scale[0] + offset[0]
        xyz[:, 1] = np.array(self.las.Y, dtype=np.float32) * scale[1] + offset[1]
        xyz[:, 2] = np.array(self.las.Z, dtype=np.float32) * scale[2] + offset[2]

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
