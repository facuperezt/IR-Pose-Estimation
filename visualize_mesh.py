# examples/Python/Basic/mesh.py

import copy
import numpy as np
import open3d as o3d
import os

def find_collisions(torch_mesh_path, slice_pcl):
    """computes collision between slice PCL and torch mesh

    Args:
        torch_mesh_path (_type_): _description_
        slice_pcl (_type_): _description_
        query_points (_type_): _description_

    Returns:
        _type_: _description_
    """
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.io.read_triangle_mesh(torch_mesh_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
    occupancy = scene.compute_occupancy(slice_pcl)

    return occupancy.any()

if __name__ == "__main__":
    pass


def visualize_mesh():
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
    
