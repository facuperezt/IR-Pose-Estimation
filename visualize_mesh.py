# examples/Python/Basic/mesh.py

import copy
import numpy as np
import open3d as o3d

if __name__ == "__main__":

    print("Testing mesh in open3d ...")
    mesh = o3d.io.read_triangle_mesh("data/train/models/201910292398/201910292398.obj", True)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
