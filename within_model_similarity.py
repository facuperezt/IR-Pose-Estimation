#%%
import os
import pickle
from utils.compatibility import listdir
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required= True, help= 'Name of the model.')
    parser.add_argument('-s', '--slice', type=str, required= True, help= 'Name of the main slice.')
    parser.add_argument('-f', '--folder_path', type=str, required=False, default="data/ss_lookup_table/dict", help= 'Path to the folder containing the feature dictionaries.')
    parser.add_argument('-rtol', '--relative_tolerance', type=float, required= False, default= 0.1, help= 'Maximum relative deviation from original slice.')
    parser.add_argument('-v', '--verbose', action='store_true', required=False, help= 'Flag to print Dissimilarity scores w.r.t. candidates.')

    return parser.parse_args()

def find_similar_slices(model : str, slice : str, folder_path : str = "data/ss_lookup_table/dict", rtol : float= 0.1, verbose= False):
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
    out = []
    for file in listdir(folder_path):
        if '_'.join(os.path.splitext(file)[0].split('_')[:-1]) == model:
            with open(os.path.join(folder_path, file), 'rb') as compare_file:
                compare_fd = pickle.load(compare_file)
            co = [c for c in compare_fd.values() if c is not None]
            if len(ch) != len(co):
                continue # No need to compare slices that don't have similar geometrical properties
            closeness_per_class = np.array([np.allclose(a,b, rtol= rtol) for a,b in zip(co, ch)]) # Original slice has to be "b" parameter for np.allclose, see Notes in https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
            if closeness_per_class.all(): # Make sure that all geometrical properties are in similar positions
                if file != '_'.join([model, slice + '.pkl']):
                    if verbose:
                        print('\nFor obj: ', file)
                        print('\tDissimilarity Score = ', '{:.2f}'.format(np.sum(sum([abs(ab) for ab in [a - b for a,b in zip(co,ch)]]))))
                    out.append(file)

    return out
#%%
if __name__ == '__main__':
    args = parse_args()
    print(find_similar_slices(args.model, args.slice, args.folder_path, args.relative_tolerance, args.verbose))