"""
This does stuff
"""
#%%
import pickle
import open3d as o3d
import os
import numpy as np
import sys
PATH = '/Users/faqprezt/Desktop/Uni/WS23/Test_Pose_Estimation/IR-Pose-Estimation'
sys.path.append(PATH)
from utils.xml_parser import parse_frame_dump, list2array
from utils.compatibility import listdir
from showtorch import _find_collision, process_frame, rotate_matrix, rotation_mat
import copy
from functools import partial

PATH_TORCH_1 = 'Datasets/torch/MRW510_10GH.obj' # path to torch MRW510_10GH
PATH_TORCH_2 = 'Datasets/torch/TAND_GERAD_DD.obj' # path to torch TAND_GERAD_DD
COLOR_TORCH_1 = np.array([0,1,0]) # color of torch MRW510_10GH
COLOR_TORCH_2 = np.array([1,0,0]) # color of torch TAND_GERAD_DD
PATH_RESULTS = 'data/test/results'
PATH_TRAINING_PCL = 'data/train/welding_zone_comp'
PATH_TEST_PCL = 'data/test/welding_zone_test'
DATASET_PATH = 'Datasets/both'



def load_matches(model_name):
    """
    Loads matched dict for model
    """
    with open(f"{PATH}/{PATH_RESULTS}/{model_name}/matched_dict.pkl", "rb") as file:
        return pickle.load(file)

def load_original_pcd(model, slice):
    pcd_path = f"{PATH}/{PATH_TEST_PCL}"
    return o3d.io.read_point_cloud(os.path.join(pcd_path, str(model), f'{model}_{slice}.xyz'))

def load_matched_pcd(dic, model, slice):
    pcd_path = f"{PATH}/{PATH_TRAINING_PCL}"
    return o3d.io.read_point_cloud(os.path.join(pcd_path, dic[f'{model}_{slice}'] + '.pcd'))

def load_matched_pcd_initializer(dic):
    return partial(load_matched_pcd, dic= dic)

def load_model(model):
    return o3d.io.read_triangle_mesh(os.path.join(PATH, DATASET_PATH, f"{model}.obj"))

def load_torches():
    torch1 = o3d.io.read_triangle_mesh(os.path.join(PATH, PATH_TORCH_1))
    torch2 = o3d.io.read_triangle_mesh(os.path.join(PATH, PATH_TORCH_2))

    return torch1, torch2
TORCHES = load_torches()

def load_frames(model):
    frames_gt = list2array(parse_frame_dump(os.path.join(PATH, DATASET_PATH, f'{model}.xml'), True))
    frames_pred = list2array(parse_frame_dump(os.path.join(PATH, PATH_RESULTS, str(model), f'{model}_predicted.xml'), True))

    return frames_gt[:,3:-3], frames_pred[:,3:-3]

def load_single_frame(model, sl):
    return list2array(parse_frame_dump(os.path.join(\
        PATH, PATH_RESULTS, str(model), f'{model}_{sl}.xml'), True))[0,3:-3]

def get_model_slices(model):
    """
    Is there an easier way to do this? Sure. Are oneliners still fun? Hekk yeah brother.
    """
    slices = [os.path.splitext(a)[0].split('_')[-1] for a in listdir(os.path.join(PATH, PATH_RESULTS, str(model)))\
             if len(os.path.splitext(a)[0].split('_')) > 1 and os.path.splitext(a)[-1] == '.xml' and\
                  not os.path.splitext(a)[0].split('_')[-1] == 'predicted' and str(model) in a and str(model)+'.xml' != a]


    return sorted([int(s) for s in slices])

def find_collision_pcl(mesh_torch, pcl_slice, tf=None, tolerance= 0.1):
    if tf is not None:
        _mesh_torch = copy.deepcopy(mesh_torch)
        _mesh_torch.transform(tf)
        _mesh_torch.compute_vertex_normals()
    else:
        _mesh_torch = mesh_torch
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(_mesh_torch))
    query_points = o3d.core.Tensor(pcl_slice, dtype=o3d.core.float32)
    distances = scene.compute_signed_distance(query_points).numpy()
    occupancy = scene.compute_occupancy(query_points).numpy().astype(bool)

    occupancy[distances > -tolerance] = False # ignore tolerable collisions

    return occupancy, distances

def allign_torch_in_pcd(pcd_input, frame):
    pcd = copy.deepcopy(pcd_input)
    _frame = copy.deepcopy(frame)
    mesh_torch, color, tf = process_frame(_frame, *TORCHES, consider_rotations= True)
    tf2 = np.eye(4)
    tf2[:3, 3] = tf[:3, 3]
    mesh_torch.transform(tf)
    pcd.transform(tf2)
    mesh_torch.compute_vertex_normals()

    return mesh_torch, pcd

def _analyze_model_gt(model, **args):
    mesh_model = load_model(model)
    mesh_model.compute_vertex_normals()
    frames, _ = load_frames(model)
    model_slices = get_model_slices(model)
    assert len(frames) == len(model_slices)
    
    pcd_loader = load_original_pcd
    _analyze_model_inner_draw_lines(pcd_loader, frames, model_slices, **args)

def _analyze_model_pred(model, **args):
    mesh_model = load_model(model)
    mesh_model.compute_vertex_normals()
    _, frames = load_frames(model)
    matched_dict = load_matches(model)
    model_slices = get_model_slices(model)
    pcd_loader = load_matched_pcd_initializer(matched_dict)

    _analyze_model_inner_draw_lines(pcd_loader, frames, model_slices, **args)

def _analyze_model_inner(pcd_loader : callable, frames : np.ndarray, model_slices : list, **args):
    counter = 0
    for i, (frame, sl) in enumerate(zip(frames, model_slices)):
        assert np.allclose(frame[1:13].astype(float), load_single_frame(model, sl)[1:13].astype(float))
        elements = []
        pcd = pcd_loader(model=model, slice=sl)
        mesh_torch, pcd = allign_torch_in_pcd(pcd, frame)
        collision, distance = find_collision_pcl(mesh_torch, np.asarray(pcd.points), tolerance= 0.0)
        pcd.paint_uniform_color(np.array([0,1,0]))
        colors = np.asarray(pcd.colors)
        colors[collision] = np.array([1,0,0])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if args.get('plot_origin', False):
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300, origin=[0, 0, 0])
            elements.append(origin_frame)
        if args.get('plot_torch', True): elements.append(mesh_torch)
        if args.get('plot_pcd', True): elements.append(pcd)
        if collision.any():
            counter +=1 
            print('collision')
            if elements and args.get('plot', True): o3d.visualization.draw_geometries(elements)
        else:
            print('clear')
    print(f"From {len(frames)} analyzed Frames: {counter} collisions | {len(frames) - counter} safe")

def _analyze_model_inner_draw_lines(pcd_loader : callable, frames : np.ndarray, model_slices : list, **args):
    counter = 0
    for i, (frame, sl) in enumerate(zip(frames, model_slices)):
        assert np.allclose(frame[1:13].astype(float), load_single_frame(model, sl)[1:13].astype(float))
        elements = []
        pcd = pcd_loader(model=model, slice=sl)
        pcd.paint_uniform_color(np.array([0,1,1]))
        mesh_torch, pcd = allign_torch_in_pcd(pcd, frame)
        collision, distance = find_collision_pcl(mesh_torch, np.asarray(pcd.points), tolerance= 0.0)
        if args.get('plot_torch', False): elements.append(mesh_torch)
        if args.get('plot_pcd', True): elements.append(pcd)
        flag = False
        if collision.any():
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[collision])
            new_pcd.paint_uniform_color([1,0,0])
            torch_pcd : o3d.geometry.PointCloud = mesh_torch.sample_points_poisson_disk(2000)
            torch_pcd.paint_uniform_color(np.array([0,1,0]))
            hull_torch, _ = torch_pcd.compute_convex_hull()
            if args.get('plot_torch_hull', True): elements.append(hull_torch)
            torch_and_collisions_pcd = torch_pcd + new_pcd
            hull_torch_and_collisions, _ = torch_and_collisions_pcd.compute_convex_hull()
            if args.get('plot_torch_hull_and_collisions', True): elements.append(hull_torch_and_collisions)
            # if the point really is in collision, then the Axis-aligned bounding boxes (AABBs) should be similar
            aabb_features_hull = np.asarray(hull_torch.get_axis_aligned_bounding_box().get_box_points())
            aabb_features_hull_and_colls = np.asarray(hull_torch_and_collisions.get_axis_aligned_bounding_box().get_box_points())
            if not np.allclose(aabb_features_hull, aabb_features_hull_and_colls): # if they are not close, there is no collision
                flag = True
                collision = np.array([False]) # no collision, compatible with .any()
            pcd_tree = o3d.geometry.KDTreeFlann(torch_pcd)
            matched_points = []
            for i,point in enumerate(new_pcd.points):
                [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
                matched_points.append(torch_pcd.points[np.asarray(idx)])
                matched_points.append(point)
            matched_points = o3d.utility.Vector3dVector(np.array(matched_points))
            matched_lines = o3d.utility.Vector2iVector(np.arange(len(matched_points)).reshape(-1,2))
            line_set = o3d.geometry.LineSet()
            line_set.points = matched_points
            line_set.lines = matched_lines
            if args.get('plot_lines', True): elements.append(line_set)
            if args.get('plot_torch_pcd', True): elements.append(torch_pcd)
            if args.get('plot_collision_pcd', True): elements.append(new_pcd)
        if args.get('plot_origin', False):
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300, origin=[0, 0, 0])
            elements.append(origin_frame)
        if collision.any() or flag:
            if not flag: counter +=1 
            av = " AVOIDED" if flag else ""
            print("collision" + av)
            if elements and args.get('plot', False): o3d.visualization.draw_geometries(elements)
        else:
            print('clear')
    print(f"From {len(frames)} analyzed Frames: {counter} collisions | {len(frames) - counter} safe")

#%%
if __name__ == '__main__':
    model = 1   
    _analyze_model_gt(model)


"""
def _analyze_model_inner_draw_lines_gui(pcd_loader : callable, frames : np.ndarray, model_slices : list, **args):
    counter = 0
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Mesh-Viewer", 1024, 750)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)
    for i, (frame, sl) in enumerate(zip(frames, model_slices)):
        assert np.allclose(frame[1:13].astype(float), load_single_frame(model, sl)[1:13].astype(float))
        elements = []
        pcd = pcd_loader(model=model, slice=sl)
        pcd.paint_uniform_color(np.array([0,1,1]))
        mesh_torch, pcd = allign_torch_in_pcd(pcd, frame)
        collision, distance = find_collision_pcl(mesh_torch, np.asarray(pcd.points), tolerance= 0.0)
        if collision.any():
            if args.get('plot_torch', True): scene.scene.add_geometry("torch", mesh_torch, rendering.MaterialRecord())
            if args.get('plot_pcd', True): scene.scene.add_geometry("slice_pcd", pcd, rendering.MaterialRecord())
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[collision])
            new_pcd.paint_uniform_color([1,0,0])
            torch_pcd : o3d.geometry.PointCloud = mesh_torch.sample_points_poisson_disk(2000)
            torch_pcd.paint_uniform_color(np.array([0,1,0]))
            pcd_tree = o3d.geometry.KDTreeFlann(torch_pcd)
            matched_points = []
            for i,point in enumerate(new_pcd.points):
                [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
                matched_points.append(torch_pcd.points[np.asarray(idx)])
                matched_points.append(point)
            matched_points = o3d.utility.Vector3dVector(np.array(matched_points))
            matched_lines = o3d.utility.Vector2iVector(np.arange(len(matched_points)).reshape(-1,2))
            line_set = o3d.geometry.LineSet()
            line_set.points = matched_points
            line_set.lines = matched_lines
            if args.get('plot_lines', True): scene.scene.add_geometry("collision_lines", line_set, rendering.MaterialRecord())
            if args.get('plot_torch_pcd', True): scene.scene.add_geometry("torch_pcd", torch_pcd, rendering.MaterialRecord())
            if args.get('plot_collision_pcd', True): scene.scene.add_geometry("collision_points", new_pcd, rendering.MaterialRecord())
        if args.get('plot_origin', False):
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300, origin=[0, 0, 0])
            elements.append(origin_frame)
        if collision.any():
            counter +=1 
            print('collision')
            if elements and args.get('plot', True):
                bounds = mesh_torch.get_axis_aligned_bounding_box()
                scene.setup_camera(60, bounds, bounds.get_center())
                for coordinate in np.asarray(new_pcd.points):
                    scene.add_3d_label(coordinate, f"{coordinate}")
            gui.Application.instance.run()  # Run until user closes window
            input()
        else:
            print('clear')
    print(f"From {len(frames)} analyzed Frames: {counter} collisions | {len(frames) - counter} safe")
"""