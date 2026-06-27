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
        Extracts coordinates and returns as int32

        Returns
        -------
        xyz : np.ndarray, shape (N, 3), dtype int32
        """
        return np.array(self.data['xyz'], dtype=np.int32)
    
    
    def get_intensity(self):
        """
        Returns
        -------
        intensity : np.ndarray, shape (N,), dtype uint8
        """
        if np.issubdtype(self.data['intensity'].dtype, np.floating):
            return self.data['intensity'].astype(np.float32)
        return self.data['intensity'].astype(np.uint8)
