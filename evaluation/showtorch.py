import open3d as o3d
import numpy as np
import sys
sys.path.append('./')
from utils.xml_parser import parse_frame_dump, list2array
from copy import copy
from tqdm import tqdm
from typing import Literal
import scipy.linalg as linalg

comp = '1'

PATH_TORCH_1 = './Datasets/torch/MRW510_10GH.obj' # path to torch MRW510_10GH
PATH_TORCH_2 = './Datasets/torch/TAND_GERAD_DD.obj' # path to torch TAND_GERAD_DD
PATH_COMP = f'./Datasets/both/{comp}.obj' # path to mesh
PATH_XML = f'./Datasets/both/{comp}.xml' # path to xml file
PATH_XML = f'data/test/results/{comp}/{comp}_predicted.xml' # path to xml file
# PATH_TORCH_1 = 'vis/MRW510_10GH.obj' # path to torch MRW510_10GH
# PATH_TORCH_2 = 'vis/TAND_GERAD_DD.obj' # path to torch TAND_GERAD_DD
# PATH_COMP = 'vis/201910204483_R1.obj' # path to mesh
# PATH_XML = 'vis/201910204483_R1.xml' # path to xml file
COLOR_TORCH_1 = np.array([0,1,0]) # color of torch MRW510_10GH
COLOR_TORCH_2 = np.array([1,0,0]) # color of torch TAND_GERAD_DD

def show_poses(pose_index= None):
    mesh_model = o3d.io.read_triangle_mesh(PATH_COMP)
    mesh_model.compute_vertex_normals()
    

    torch1 = o3d.io.read_triangle_mesh(PATH_TORCH_1)
    torch2 = o3d.io.read_triangle_mesh(PATH_TORCH_2)

    frames = list2array(parse_frame_dump(PATH_XML, True))
    frames = frames[:,3:]
    if pose_index is not None:
        frames = frames[pose_index]
    _show_frames(mesh_model, torch1, torch2, frames, False)

def print_statistics(consider_rotations= False, frame_percentage=1, find_collision_args= None):
    mesh_model = o3d.io.read_triangle_mesh(PATH_COMP)
    mesh_model.compute_vertex_normals()
    

    torch1 = o3d.io.read_triangle_mesh(PATH_TORCH_1)
    torch2 = o3d.io.read_triangle_mesh(PATH_TORCH_2)

    frames = list2array(parse_frame_dump(PATH_XML, True))
    frames = frames[:,3:][:int(len(frames)*frame_percentage)]
    _print_statistics(mesh_model, torch1, torch2, frames, consider_rotations, find_collision_args)

def _print_statistics(mesh_model, torch1, torch2, frames, consider_rotations= False, find_collision_args= None):
    collisions_counter = []
    _find_collision_args = {'type' : 'pcl', 'n_points' : 10000, 'poisson' : False}
    _find_collision_args.update(**find_collision_args)
    for frame in tqdm(frames, "Frames:"):
        mesh_torch, color, tf = process_frame(frame, torch1, torch2, consider_rotations)
        mesh_torch.paint_uniform_color(color)
        mesh_torch.transform(tf)
        mesh_torch.compute_vertex_normals()
        collisions, points, distances = _find_collision(mesh_model, mesh_torch, **_find_collision_args)
        collisions_counter.append(collisions.any())
    collisions_counter = np.array(collisions_counter)
    for collision_present, amount in zip(*np.unique(collisions_counter, return_counts= True)):
        print(f"{amount} {'collision' if collision_present else 'collision free'} cases detected.")

def show_random_poses(nr_poses= 10, consider_rotations=False):
    mesh_model = o3d.io.read_triangle_mesh(PATH_COMP)
    mesh_model.compute_vertex_normals()
    

    torch1 = o3d.io.read_triangle_mesh(PATH_TORCH_1)
    torch2 = o3d.io.read_triangle_mesh(PATH_TORCH_2)

    frames = list2array(parse_frame_dump(PATH_XML, True))
    frames = frames[:,3:]
    frames = frames[np.random.randint(0, len(frames), nr_poses)]

    _show_frames(mesh_model, torch1, torch2, frames, False, consider_rotations)

def _color_fader(color_collision, color_ok, mix=0):
    return (1-mix)*color_collision + mix*color_ok

def process_frame(frame, torch1, torch2, consider_rotations= False):
    """
    returns: mesh_torch, color, tf
    """
    # if frame[0] == 0: # 0 stands for torch MRW510_10GH 
    #     mesh_torch = copy(torch1)
    #     color = COLOR_TORCH_1
    # else:
    #     mesh_torch = copy(torch2)
    #     color = COLOR_TORCH_2
    torch = frame[0]
    if torch == 'MRW510_10GH' or torch == 'MRW510_CDD_10GH' or torch == 'TL2000_EFF20_17SO':
        mesh_torch = copy(torch1)
        color = COLOR_TORCH_1
    elif torch  == 'TAND_GERAD_DD':
        mesh_torch = copy(torch2)
        color = COLOR_TORCH_2
    else:
        raise ValueError(f'Weird Torch: <{torch}> , check XML-Parser wether WkzName is being put as the 4th place in the frame')
    rot_xyz = frame[10:13]
    tf = np.zeros((4,4))  # 4x4 homogenous transform
    tf[3,3] = 1.0
    tf[0:3,3] = frame[1:4]
    if consider_rotations:
        pos = np.zeros((3,3))
        pos[0:3, 0] = frame[14:17]
        pos[0:3, 1] = frame[17:20]
        pos[0:3, 2] = frame[20:23]
        for axis, rot in zip(['x', 'y', 'z'], rot_xyz):
            if rot != 0:
                rotation = rotation_mat(axis, rot*np.pi/180)
                pos = np.matmul(rotation, pos)
        tf[0:3,0:3] = pos
    else:
        tf[0:3, 0] = frame[14:17]
        tf[0:3, 1] = frame[17:20]
        tf[0:3, 2] = frame[20:23]

    return mesh_torch, color, tf

def _show_frames(mesh_model, torch1, torch2, frames, all_points= True, consider_rotations= False):
    elements = []
    elements.append(mesh_model)
    for frame in tqdm(frames, "Frames:"):
        mesh_torch, color, tf = process_frame(frame, torch1, torch2, consider_rotations)
        mesh_torch.paint_uniform_color(color)
        mesh_torch.transform(tf)
        mesh_torch.compute_vertex_normals()
        collisions, points, distances = _find_collision(mesh_model, mesh_torch, 'pcl', 100_000, False)
        collision_detected = collisions.any()
        if collision_detected:
            if all_points:
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
                pcd.paint_uniform_color(np.array([0,1,0]))
                colors = np.asarray(pcd.colors)
                colors[collisions] = np.array([1,0,0])
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[collisions]))
                pcd.paint_uniform_color(np.array([1,0,0]))
                mesh_torch.paint_uniform_color(np.array([0,0,1])) # Paint BLUE for collision
                elements.append(mesh_torch)
            elements.append(pcd)
        else:
            mesh_torch.paint_uniform_color(np.array([0,1,0])) # Paint GREEN else
            elements.append(mesh_torch)

    o3d.visualization.draw_geometries(elements)

def rotation_mat(axis, rot):
    '''
    axis_x, axis_y, axis_z = [1,0,0], [0,1,0], [0, 0, 1]
    '''
    axis_dic = {'x': [1,0,0], 'y': [0,1,0], 'z': [0,0,1]}
    axis = (axis_dic[axis])
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * rot))
    return rot_matrix

def rotate_matrix(M, axis : str, theta):
    """
    rotate matrix M around axis by angle theta
    """
    mat = np.zeros((4,4)) + np.eye(4)
    theta = float(theta)*np.pi/180

    if axis.lower() == 'x':
        mat[1,1] = np.cos(theta) ; mat[1,2] = np.sin(theta)
        mat[2,1] = -np.sin(theta) ; mat[2,2] = np.cos(theta)
        # axis = np.array([1,0,0])
    elif axis.lower() == 'y':
        mat[0,0] = np.cos(theta) ; mat[0,2] = -np.sin(theta)
        mat[2,0] = np.sin(theta) ; mat[2,2] = np.cos(theta)
        # axis = np.array([0,1,0])
    elif axis.lower() == 'z':
        mat[0,0] = np.cos(theta) ; mat[0,1] = -np.sin(theta)
        mat[1,0] = np.sin(theta) ; mat[1,1] = np.cos(theta)
        # axis = np.array([0,0,1])
    else:
        raise ValueError(f'Invalid axis <{axis}> - has to be either [x, y, z]')
    
    # mat = scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * theta))

    return M@mat

def find_collisions(frame, type : Literal['pcl', 'mesh'] = 'mesh'):
    mesh_model = o3d.io.read_triangle_mesh(PATH_COMP)
    mesh_model.compute_vertex_normals()

    torch1 = o3d.io.read_triangle_mesh(PATH_TORCH_1)
    torch2 = o3d.io.read_triangle_mesh(PATH_TORCH_2)

    mesh_torch, color, tf = process_frame(frame, torch1, torch2)

    mesh_torch.paint_uniform_color(color)
    mesh_torch.transform(tf)
    mesh_torch.compute_vertex_normals()

    return _find_collision(mesh_model, mesh_torch, type)


def _find_collision(mesh_model, mesh_torch, type : Literal['pcl', 'mesh'] = 'mesh', n_points= 5000, poisson= False, advanced_check= True, tolerance= 0.1):
    if type == 'pcl':
        occupancy, points, distances = _find_collision_pc(mesh_model, mesh_torch, n_points, poisson)
    elif type == 'mesh':
        occupancy, points, distances = np.array([mesh_torch.is_intersecting(mesh_model)]), None, None
    else:
        raise ValueError('type not recognized')
    if poisson == False and occupancy.any() and advanced_check:
        return _find_collision_pc(mesh_model, mesh_torch, 5000, True)
    else:
        return occupancy, points, distances

def _find_collision_pc(mesh_model, mesh_torch, n_points= 200, poisson= False, tolerance= 0.1):
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_model))
    if poisson:
        points = np.asarray(mesh_torch.sample_points_poisson_disk(n_points).points)
    else:
        points = np.asarray(mesh_torch.sample_points_uniformly(n_points).points)
    query_points = o3d.core.Tensor(points, dtype=o3d.core.float32)
    distances = scene.compute_signed_distance(query_points).numpy()
    occupancy = scene.compute_occupancy(query_points).numpy().astype(bool)

    occupancy[distances > -tolerance] = False

    return occupancy, points, distances

if __name__== '__main__':
    import sys
    # frames = list2array(parse_frame_dump(PATH_XML, True))
    # frames = frames[:,3:]
    # frame = frames[4]
    # print(find_collision(frame))
    # comp = 'Reisch'

    # PATH_COMP = f'./Datasets/both/{comp}.obj' # path to mesh
    # PATH_XML = f'./Datasets/both/{comp}.xml' # path to xml file
    show_random_poses(20, False)
    # print_statistics(False, 1, find_collision_args={'poisson' : False, 'n_points' : 15000, 'type' : 'pcl', 'advanced_check' : False, 'tolerance' : 0})
    # for i in range(10):
    #     print(i)
    #     show_poses([i])



""" 8_cls
[ 1.68488670e-01  1.73648180e-01  2.25000000e+02 -7.96280148e+02]
[-0.67567354 -0.6963642   5.         83.93862361]
[-7.17688560e-01  6.96364200e-01  4.96055740e+08  2.18206138e+03]
[0. 0. 0. 1.]

[ 4.50000000e+01, -7.96280148e+02,  8.39386236e+01,  2.18206138e+03,
-1.01071929e-13, -1.00000000e+00, -1.80411242e-14,  3.33066907e-14,
-1.88737914e-15,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
0.00000000e+00,  0.00000000e+00,  1.68488670e-01, -6.75673540e-01,
-7.17688560e-01,  1.73648180e-01, -6.96364200e-01,  6.96364200e-01,
2.25000000e+02,  5.00000000e+00,  4.96055740e+08,  2.70000000e+01,
0.00000000e+00,  4.15754073e+09]
"""


""" Normal
[-1.00000000e+00 -1.58487525e-14 -1.57262879e-14 -7.96280148e+02]
[ 2.23269252e-14 -7.07106781e-01 -7.07106781e-01  8.39386236e+01]
[ 8.65956056e-17 -7.07106781e-01  7.07106781e-01  2.18206138e+03]
[0. 0. 0. 1.]

[ 4.50000000e+01, -7.96280148e+02,  8.39386236e+01,  2.18206138e+03,
-1.01071929e-13, -1.00000000e+00, -1.80411242e-14,  3.33066907e-14,
-1.88737914e-15,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
0.00000000e+00,  0.00000000e+00, -1.00000000e+00,  2.23269252e-14,
8.65956056e-17, -1.58487525e-14, -7.07106781e-01, -7.07106781e-01,
-1.57262879e-14, -7.07106781e-01,  7.07106781e-01,  2.70000000e+01,
0.00000000e+00,  4.15754073e+09]
"""