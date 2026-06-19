"""
Point cloud restoration from a compressed .npz file.
"""

import numpy as np


class NpzReader:
    """ Universal reader for point cloud data saved in .npz format """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None
    

    def __enter__(self):
        self.data = np.load(self.filepath)
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.data is not None:
            self.data.close()
        del self.data


    def get_xyz(self):
        """
        Extracts coordinates and returns as float32

        Returns
        -------
        xyz : np.ndarray, shape (N, 3), dtype float32
        """
        return np.array(self.data['xyz'], dtype=np.float32)
    
    
    def get_intensity(self):
        """
        Returns
        -------
        intensity : np.ndarray, shape (N,), dtype uint8
        """
        if np.issubdtype(raw_data.dtype, np.floating):
            return raw_data.astype(np.float32)
        return raw_data.astype(np.uint8)
