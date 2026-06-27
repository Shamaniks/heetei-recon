"""
Downsampling CLI dispatcher
Usage:
    command: python -m downsampling
        --cloud <cloud `.npz`>
        --voxel_size <float voxel_size in meters>
        [--intensity <intensity `.npz`, if provided max pooling will be used>]
        [--output <output file, default: `output.npz`]
"""


import argparse
import numpy as np

from downsampling.max_pooling import voxel_downsample
from utils.npz_reader import NpzReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converts .las/.laz data to .npz for future processing"
    )
    parser.add_argument("--cloud", required=True,
                        help="Cloud file `.npz`")
    parser.add_argument("--voxel_size", required=True, type=float,
                        help="Voxel size for voxel downsampling")
    parser.add_argument("--intensity", default=None,
                        help="Intensitu `.npz`")
    parser.add_argument("--output", default="output.npz",
                        help="Output file for xyz only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with NpzReader(args.cloud) as nr:
        xyz = nr.get_xyz()
    print(f"[downsampling] Original point number: {len(xyz):,}")

    if args.intensity:
        with NpzReader(args.intensity) as nr:
            intensity = nr.get_intensity()
        assert len(xyz) == len(intensity)

        indices = voxel_downsample(xyz, intensity, args.voxel_size, 1e-4)
    else:
        # Different downsampling method isn't implemented yet
        pass
    
    xyz_down = xyz[indices]
    print(f"[downsampling] Downsampled point number: {len(xyz_down):,}")
    np.savez_compressed(args.output, xyz=xyz_down)
    np.savez_compressed(f"{args.output[:-3]}_intensity.npz", intensity=intensity[indices])
