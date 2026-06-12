"""
Stage 4 – Mesh visualisation and optional PLY export.
"""

import open3d as o3d


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: str) -> None:
    """Write the mesh to a PLY file."""
    ok = o3d.io.write_triangle_mesh(path, mesh, write_ascii=False)
    assert ok, f"Open3D failed to write mesh to '{path}'"
    print(f"[visualize] Mesh saved → '{path}'")


def show_mesh(mesh: o3d.geometry.TriangleMesh, window_title: str) -> None:
    """
    Launch Open3D's interactive 3D viewer.

    draw_geometries blocks until the window is closed.
    No context manager is used (avoids laspy/open3d version conflicts with
    shared OpenGL context teardown on certain Linux drivers).
    """
    print("[visualize] Opening interactive viewer  "
          "(close the window to exit) …")
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=window_title,
        width=1280,
        height=800,
        mesh_show_back_face=True,    # render both sides of each triangle
        mesh_show_wireframe=False,
    )
