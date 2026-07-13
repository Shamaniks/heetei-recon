"""
Mesh PLY visualization with manual WASD Fly Mode control (Fix for Open3D 0.19.0+).
"""

import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualizes a PLY mesh file with interactive WASD Fly Mode"
    )
    parser.add_argument("--mesh_file", required=True,
                        help="Path to the source .ply mesh file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"[loader] Loading mesh from: {args.mesh_file}")
    mesh = o3d.io.read_triangle_mesh(args.mesh_file)
    
    if not mesh.has_vertex_normals():
        print("[mesh] Computing vertex normals...")
        mesh.compute_vertex_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    if mesh.has_vertex_colors():
        pcd.colors = mesh.vertex_colors
    if mesh.has_vertex_normals():
        pcd.normals = mesh.vertex_normals

    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Mesh & Colored Vertices", 1024, 768)
    scene_widget = gui.SceneWidget()
    window.add_child(scene_widget)

    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.show_axes(True)

    mesh_material = rendering.MaterialRecord()
    mesh_material.shader = "defaultLit"

    point_material = rendering.MaterialRecord()
    point_material.shader = "defaultUnlit"  
    point_material.point_size = 5.0        

    scene_widget.scene.add_geometry("sphere_mesh", mesh, mesh_material)
    
    scene_widget.scene.add_geometry("mesh_vertices", pcd, point_material)

    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    scene_widget.setup_camera(60, bounds, center)

    up_vector = [0.0, 0.0, 1.0] 

    cam_matrix = scene_widget.scene.camera.get_model_matrix()
    eye = cam_matrix[:3, 3] 

    scene_widget.scene.camera.look_at(center, eye, up_vector)

    scene_widget.set_view_controls(gui.SceneWidget.Controls.FLY)

    app.run()
