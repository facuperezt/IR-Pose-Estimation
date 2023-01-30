import open3d as o3d
import numpy as np


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud('../Reisch.pcd')
    print(pcd)
    o3d.visualization.draw_geometries([pcd])