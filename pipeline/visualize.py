"""
Stage 4 – Mesh visualisation and optional PLY export.

PATCH v1.1 — headless environment guard
----------------------------------------
Google Colab runs without a display server. Calling draw_geometries()
in a headless environment causes Open3D to either hang indefinitely or
raise an unhandled C++ exception that surfaces as a silent crash or a
misleading KeyboardInterrupt in the notebook.

The guard checks for the DISPLAY environment variable (set by X11/Wayland
on desktop systems; absent in Colab and most CI environments). If missing,
the visualizer call is skipped entirely and the function returns cleanly,
allowing the script to exit with status 0 after save_mesh() completes.
"""

import os

import open3d as o3d


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: str) -> None:
    """Write the mesh to a binary PLY file."""
    ok = o3d.io.write_triangle_mesh(path, mesh, write_ascii=False)
    assert ok, f"Open3D failed to write mesh to '{path}'"
    print(f"[visualize] Mesh saved → '{path}'")


def show_mesh(mesh: o3d.geometry.TriangleMesh, window_title: str) -> None:
    """
    Launch Open3D's interactive 3D viewer.

    PATCH v1.1: skips rendering silently in headless environments (Colab,
    SSH sessions without X forwarding, CI runners).

    The DISPLAY variable is set by every X11-capable display server on Linux
    and macOS (via XQuartz). Its absence is the canonical signal that no
    framebuffer is available. On Windows, draw_geometries() uses a native
    Win32 window and does not need DISPLAY — the guard is Linux/macOS-only
    in practice, but is harmless on Windows because DISPLAY is never set
    there anyway (so the guard fires and returns, which is fine since Colab
    only runs on Linux).
    """
    # ── PATCH: headless guard ──────────────────────────────────────────────
    if "DISPLAY" not in os.environ:
        print(
            "[visualize] WARNING: No display server detected (DISPLAY not set). "
            "Skipping interactive visualisation — this is expected in Google "
            "Colab and headless SSH sessions. The saved .ply file can be "
            "opened in MeshLab, CloudCompare, or Blender."
        )
        return
    # ── end PATCH ─────────────────────────────────────────────────────────

    print("[visualize] Opening interactive viewer "
          "(close the window to exit) …")
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=window_title,
        width=1280,
        height=800,
        mesh_show_back_face=True,
        mesh_show_wireframe=False,
    )
