"""
Converts .las/.laz data to .npz for future processing
Usage:
    command: python utils/las_to_npz.py
        --source <source file for cloud or track, any with xyz>
        --xyz <output file for xyz only>
        [--intensity <output file for intensity only>]
        [--gps_time <output file for gps_time only>]
    note:
        intensity won't be readed and saved if not output file specified
        same for gps_time
"""


import argparse
import numpy as np

from utils.las_reader import LasReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converts .las/.laz data to .npz for future processing"
    )
    parser.add_argument("--source", required=True,
                        help="Source file for cloud or track, any with xyz")
    parser.add_argument("--xyz", required=True,
                        help="xyz file for xyz only")
    parser.add_argument("--intensity", default=None,
                        help="Output file for intensity only")
    parser.add_argument("--gps_time", default=None,
                        help="Output file for intensity only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with LasReader(args.source) as lr:

        xyz = lr.get_xyz()
        np.savez_compressed(args.xyz, xyz=xyz)
        print(f"[las_to_npz] point cloud converted to {args.xyz}")

        if args.intensity:
            intensity = lr.get_intensity()
            np.savez_compressed(args.intensity, intensity=intensity)
            print(f"[las_to_npz] intensity array converted to {args.intensity}")

        if args.gps_time:
            gps_time = lr.get_gps_time()
            np.savez_compressed(args.gps_time, gps_time=gps_time)
            print(f"[las_to_npz] gps_time array converted to {args.gps_time}")
