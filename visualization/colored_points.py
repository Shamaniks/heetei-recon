"""
Colored point cloud visualization module
"""

import numpy as np
import open3d as o3d




def draw_colored_points(xyz: np.ndarray, intensity: np.ndarray):
    """
    Renders the point cloud using Open3D with Viridis color mapping.

    Parameters
    ----------
    xyz : np.ndarray, shape (N, 3), dtype float32
        True metric coordinates of the cave points (scaled and offset).
    intensity : np.ndarray, shape (N,), dtype float32
        The array of restored intensities, can contain values greater than 255.
    """

    print(f"[colored_points] Transla{len(xyz):,} точек...")
    
    if is_intensity:
        # Переводим одномерный uint8 в трехмерный Viridis float64
        rgb_colors = intensity_to_viridis(colors_input)
    else:
        # Если зашел кастомный RGB
        if colors_input.dtype == np.uint8:
            # Обязательно делим uint8 на 255.0, преобразуя в float64
            rgb_colors = colors_input.astype(np.float64) / 255.0
        else:
            rgb_colors = colors_input.astype(np.float64)

    # 2. Инициализация геометрии Open3D
    pcd = o3d.geometry.PointCloud()
    
    # Передаем массивы через векторы Open3D (они требуют float64 под капотом)
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)

    print("[colored_points] Открытие интерактивного окна Open3D...")
    
    # 3. Настройка и запуск окна визуализации
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Heetei Cave Reconstruction - Colored View", width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Тонкая настройка рендеринга для Edge-девайсов и пещерной графики
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # Увеличиваем размер точек, чтобы стены пещеры не "просвечивали"
    render_option.background_color = np.array([0.05, 0.05, 0.05])  # Темно-серый фон
    
    vis.run()
    vis.destroy_window()
    print("[colored_points] Окно визуализации успешно закрыто.")
