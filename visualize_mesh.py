# examples/Python/Basic/mesh.py

import copy
import numpy as np
import open3d as o3d
import os

if __name__ == "__main__":
    f = open('obj_0_pts.txt', 'r')
    objs = f.readlines()
    f.close()
    path = 'data/train/split'
    for obj in objs:
        folder = '_'.join(os.path.splitext(obj)[0].split('_')[:-1])
        mesh = o3d.io.read_triangle_mesh(f'{path}/{folder}/{obj.strip()}', True)
        print(mesh)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])

    # path = 'data/train/split/1170138_ROB_2_180/1170138_ROB_2_180_189.obj'
    # mesh = o3d.io.read_triangle_mesh(path, True)
    # mesh.compute_vertex_normals()
    # # o3d.visualization.draw_geometries([mesh])

    # f = open('obj_0_pts.txt', 'r')
    # objs = f.readlines()
    # f.close()
    # print(objs)
