import os
import sys
from utils.compatibility import listdir
from typing import Dict, List
import shutil
import copy
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, help='Which dataset to load.')
    parser.add_argument('-t', '--test_models', nargs='*', type=str, help='Name of models to be used in testing')

    return parser.parse_args()

def _get_individual_models(path:str, test_model_names:List[str] = None) -> Dict:
    objs = listdir(path)
    train_dic = {}
    test_dic = {}
    if test_model_names is None:
        test_model_names = []
    for obj in objs:
        model, ext = os.path.splitext(obj)
        if ext not in ['.obj', '.xml', '.mtl']: continue
        dic = test_dic if model in test_model_names else train_dic
        dic.setdefault(model, []).append(obj)

    return train_dic, test_dic

def _write_individual_models(in_path: str, out_path: str, models_dict: Dict, test_dict: Dict = None, test_path:str = None):
    if test_dict is None:
        test_dict = {}
    if test_path is None:
        test_path = ''
    for dictionary, path in zip([models_dict, test_dict], [out_path, test_path]):
        for model, files in dictionary.items():
            for file in files:
                model_out_path = os.path.join(path, model.replace(' ', '-'))
                if not os.path.exists(model_out_path):
                    os.makedirs(model_out_path)
                shutil.copyfile(os.path.join(in_path, file), os.path.join(model_out_path.replace(' ', '-'), file.replace(' ', '-')))




def split_models(in_path: str, out_path: str, test_path:str = None, test_model_names: List[str] = None) -> bool:
    models_dict, test_dict = _get_individual_models(in_path, test_model_names)
    _write_individual_models(in_path, out_path, models_dict, test_dict, test_path)
    return True


if __name__ == '__main__':
    args = parse_args()
    os.system('mv data data_last')
    if args.dataset in ['1', '2']:
        split_models(f'Dataset/welding_objects_ds'+args.dataset, 'data/train/models', 'data/test/models', args.test_models)
    elif args.dataset == 'trailer':
        split_models(f'Dataset/ds_Trailer', 'data/train/models', 'data/test/models', args.test_models)
    else:
        split_models(args.dataset, 'data/train/models', 'data/test/models', args.test_models)



