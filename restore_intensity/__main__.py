"""
Dispatcher of utils module
Usage:
    command: python -m restore_intensity 
        --cloud <file.npz> 
        --track <file.npz> 
        --intensity <file.npz>
        --chunk_size <int>
        [--lidar_max_range <default=30>]
        [--restoration_exp <default=2>]
        [--output <file.npz>]
"""


from utils.npz_reader import NpzReader
from restore_intensity.restore_intensity import restore_intensity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converts .las/.laz data to .npz for future processing"
    )
    parser.add_argument("--cloud", required=True,
                        help="Cloud file `.npz`")
    parser.add_argument("--track", required=True,
                        help="Track file `.npz`")
    parser.add_argument("--intensity", required=True,
                        help="Intensitu `.npz`")
    parser.add_argument("--chunk_size", type=int, required=True,
                        help="Intensitu `.npz`")
    parser.add_argument("--lidar_max_range", type=float, default=30.0,
                        help="Max range of LiDAR")
    parser.add_argument("--restoration_exp", type=float, default=2.0,
                        help="Float")
    parser.add_argument("--output", default="output.npz", 
                        help="Output file for xyz only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with NpzReader(args.cloud) as nr:
        xyz_cloud = nr.get_xyz()

    with NpzReader(args.track) as nr:
        xyz_track = nr.get_xyz()

    with NpzReader(args.intensity) as nr:
        intensity = nr.get_intensity()

    intensity = restore_intensity(
        xyz_cloud,
        xyz_track,
        intensity,
        args.chunk_size,
        agrs.lidar_max_range,
        args.restoration_exp,
    )
    
    np.savez_compressed(args.output, intensity=intensity)
