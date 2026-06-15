import laspy
import open3d as o3d
import numpy as np

def main():
    # 1. Пути к вашим файлам (измените на свои)
    laz_path = "traectories/global_traectory.laz"
    ply_path = "output_mesh.ply"

    with laspy.open(laz_path) as fh:
        print(f"Количество точек в файле: {fh.header.point_count:,}")
if __name__ == "__main__":
    main()

