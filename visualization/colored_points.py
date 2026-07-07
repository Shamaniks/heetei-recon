"""
Colored point cloud visualization module.

Fixes applied to the pre-existing version of this file
-------------------------------------------------------
The previous implementation referenced undefined names (`is_intensity`,
`colors_input`), never imported `intensity_to_viridis`, and mixed
Russian/English comments — it could not run. This version implements
exactly the contract documented in its own original docstring: take
metric xyz + restored intensity, map intensity to Viridis colors, and
render/save the result. The interactive Open3D window is now opt-in
(default off) since Edge deployments are frequently headless; a .ply
export is always available as the headless-safe output path.
"""

import numpy as np
import open3d as o3d

from utils.intensity_to_viridis import intensity_to_viridis


def build_colored_point_cloud(xyz_metres: np.ndarray, intensity: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Parameters
    ----------
    xyz_metres : (N, 3) float64 — true metric coordinates (scaled + offset).
    intensity   : (N,) numeric — restored laser intensity (may exceed 255).

    Returns
    -------
    pcd : o3d.geometry.PointCloud with Viridis-mapped .colors attached.
    """
    assert xyz_metres.ndim == 2 and xyz_metres.shape[1] == 3, "xyz_metres must have shape (N, 3)"
    assert len(xyz_metres) == len(intensity), "xyz and intensity length mismatch"

    rgb_colors = intensity_to_viridis(intensity)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_metres.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
    return pcd


def save_colored_point_cloud(pcd: o3d.geometry.PointCloud, path: str) -> None:
    ok = o3d.io.write_point_cloud(path, pcd)
    assert ok, f"Failed to write colored point cloud to '{path}'"
    print(f"[colored_points] Saved colored point cloud to '{path}'")


def show_colored_point_cloud(pcd: o3d.geometry.PointCloud, window_title: str = "Heetei Cave Reconstruction - Colored View") -> None:
    """Open an interactive Open3D window. Requires a display; skip on headless Edge hardware."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1280, height=720)
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.05, 0.05, 0.05])

    vis.run()
    vis.destroy_window()
    print("[colored_points] Interactive window closed.")


def draw_colored_points(xyz_metres: np.ndarray, intensity: np.ndarray) -> o3d.geometry.PointCloud:
    """Build a colored point cloud from xyz + intensity. Does not render or save by itself."""
    print(f"[colored_points] Building colored cloud for {len(xyz_metres):,} points...")
    return build_colored_point_cloud(xyz_metres, intensity)
