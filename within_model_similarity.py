#%%
import os
import pickle
from utils.compatibility import listdir
import numpy as np
from argparse import ArgumentParser
import open3d as o3d
from utils.foundation import load_pcd_data
from matplotlib import pyplot as plt

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required= True, help= 'Name of the model.')
    parser.add_argument('-s', '--slices', nargs='+', type=str, required= True, help= 'Name of the main slice.')
    parser.add_argument('-f', '--folder_path', type=str, required=False, default="data/ss_lookup_table/dict", help= 'Path to the folder containing the feature dictionaries.')
    parser.add_argument('-rtol', '--relative_tolerance', type=float, required= False, default= 0.1, help= 'Maximum relative deviation from original slice.')
    parser.add_argument('-atol', '--absolute_tolerance', type=float, required= False, default= 0.1, help= 'Maximum absolute deviation from original slice.')
    parser.add_argument('-v', '--verbose', action='store_true', required=False, help= 'Flag to print Dissimilarity scores w.r.t. candidates.')
    parser.add_argument('-vis', '--visualize', action='store_true', help= 'Whether to visualize each slice (not recommended for large amounts of slices.)')

    return parser.parse_args()

def find_similar_slices(model : str, slice: str, folder_path: str = "data/ss_lookup_table/dict", rtol: float  =  0.1, atol : float = 5, verbose= False):
    """Finds similar slices within the same model.

    Args:
        model (str): Name of model.
        slice (str): Name of slice.
        folder_path (str): Path to folder containing pickled. feature dictionaries. 
        rtol (float): Maximum relative deviation from original slice.
    
    Returns:
        matched_slices (list): List of matching slices' pickle files.
        
    ExampleArgs:
        model = "160151"
        slice = "4"
        folder = "data/ss_lookup_table/dict"
    """
    with open(os.path.join(folder_path, '_'.join([model, slice]) + '.pkl'), 'rb') as pickle_file:
        chosen_fd = pickle.load(pickle_file)
    ch = [c for c in chosen_fd.values() if c is not None]
    if verbose:
        print(ch[2:])
    out = []
    for file in listdir(folder_path):
        if '_'.join(os.path.splitext(file)[0].split('_')[:-1]) == model:
            with open(os.path.join(folder_path, file), 'rb') as compare_file:
                compare_fd = pickle.load(compare_file)
            co = [c for c in compare_fd.values() if c is not None]
            if len(ch) != len(co):
                continue # No need to compare slices that don't have similar geometrical properties
            closeness_per_class = np.array([np.allclose(a,b, rtol= rtol, atol= atol) for a,b in zip(co, ch)]) # Original slice has to be "b" parameter for np.allclose, see Notes in https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
            if closeness_per_class.all(): # Make sure that all geometrical properties are in similar positions
                if file != '_'.join([model, slice + '.pkl']):
                    if verbose:
                        print('\nFor obj: ', file)
                        print('\tDissimilarity Score = ', '{:.2f}'.format(np.sum(sum([abs(ab) for ab in [a - b for a,b in zip(co,ch)]]))))
                        print(co[2:])
                    out.append(os.path.splitext(file)[0])

    return out

def visualize_slices(slices):
    pcds = []
    cmap_names = ['Spectral', 'coolwarm', 'bwr', 'seismic']
    for _slice, cmap_name in zip(slices, cmap_names):
        pcd = o3d.io.read_point_cloud(f'data/train/welding_zone/{_slice}.pcd')
        labels = load_pcd_data(f'data/train/welding_zone/{_slice}.pcd')[:, 3].astype(int)
        nr_unique_labels = np.unique(labels).shape[0]
        cmap = plt.get_cmap(cmap_name, nr_unique_labels)
        colors_choice = np.array([np.array(cmap(i/nr_unique_labels))[:3] for i in range(cmap.N)])
        colors = np.array([cmap(np.linalg.norm(x)) for x in pcd.points])
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        pcds.append(pcd)
        o3d.visualization.draw_geometries(pcds)



if __name__ == '__main__':
    args = parse_args()
    slices = args.slices
    if len(slices) == 1 and slices[0] == 'all':
        slices = [os.path.splitext(s)[0].split('_')[-1] for s in listdir(args.folder_path) if args.model == '_'.join(os.path.splitext(s)[0].split('_')[:-1])]

    similar_onehot_matrix = np.zeros((len(slices), len(slices)), dtype= np.bool8)
    found = []
    for _slice in slices:
        if _slice in found: continue
        if args.visualize: visualize_slices([args.model + '_' + _slice])
        s_s = find_similar_slices(args.model, _slice, args.folder_path, args.relative_tolerance, args.absolute_tolerance, args.verbose)
        found.append(_slice)
        similar = [s.split('_')[-1] for s in s_s]
        found.extend(similar)
        for _similar_slice in similar + [_slice]:
            similar_onehot_matrix[int(_slice), int(_similar_slice)] = True
        if args.visualize: visualize_slices(s_s)
    similar_onehot_matrix = np.maximum(similar_onehot_matrix, similar_onehot_matrix.transpose())
    np.save(args.model+'_similarity_onehot_matrix.npy', similar_onehot_matrix)