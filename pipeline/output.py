"""
Stage — Final output: mesh persistence and optional interactive display.

Gap note
--------
The pre-existing main.py imported `save_mesh` / `show_mesh` from a
`pipeline.visualize` module that does not exist anywhere in the
repository. This module is that missing piece. Mesh saving is always
available (headless-safe); interactive display is opt-in per config,
since Edge deployments are frequently headless.
"""

import open3d as o3d


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: str) -> None:
    ok = o3d.io.write_triangle_mesh(path, mesh)
    assert ok, f"Failed to write mesh to '{path}'"
    print(f"[output] Saved mesh to '{path}'")


def show_mesh(mesh: o3d.geometry.TriangleMesh, window_title: str = "Heetei Cave Mesh") -> None:
    """Open an interactive Open3D window. Requires a display; skip on headless Edge hardware."""
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=window_title,
        width=1280,
        height=800,
        mesh_show_back_face=True,
        mesh_show_wireframe=False,
    )
    print("[output] Interactive mesh window closed.")
