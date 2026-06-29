"""
Mesh visualisation and optional PLY export.
"""


import argparse
import open3d as o3d


from utils.npz_reader import NpzReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converts .las/.laz data to .npz for future processing"
    )
    parser.add_argument("--point_cloud", default=None,
                        help="Source file for cloud or track, any with xyz")
    return parser.parse_args()


if __name__ == "__main__":
    """
    Launch Open3D's interactive 3D viewer.
    """
    args = parse_args()

    with NpzReader(args.point_cloud) as nr:

        xyz_int32 = nr.get_xyz()
    xyz_fp64  = xyz_int32 * 1e-4
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_fp64)

    print("[visualize] Opening interactive viewer for point cloud"
          "(close the window to exit) …")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point cloud only visualization",
        width=1280,
        height=800,
        mesh_show_back_face=True,
        mesh_show_wireframe=False,
    )
