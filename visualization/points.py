"""
Mesh visualisation and optional PLY export.
"""


import argparse
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


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

    app = gui.Application.instance
    app.initialize()

    window = app.create_window("", 1024, 768)
    scene_widget = gui.SceneWidget()
    window.add_child(scene_widget)

    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.show_axes(True)

    # Используем defaultUnlit для отображения точек без расчета теней и бликов
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 2.0  # Опционально: задаем размер точек на экране

    # Добавляем облако точек (pcd) вместо сетки (mesh)
    scene_widget.scene.add_geometry("point_cloud", pcd, material)

    # Рассчитываем границы и камеру на основе pcd
    bounds = pcd.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    scene_widget.setup_camera(60, bounds, center)

    up_vector = [0.0, 0.0, 1.0] 

    cam_matrix = scene_widget.scene.camera.get_model_matrix()
    eye = cam_matrix[:3, 3]

    scene_widget.scene.camera.look_at(center, eye, up_vector)

    scene_widget.set_view_controls(gui.SceneWidget.Controls.FLY)

    app.run()
