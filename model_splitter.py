import os
import sys
from utils.compatibility import listdir
from typing import Dict
import shutil
import copy

def _get_individual_models(path:str) -> Dict:
    objs = listdir(path)
    dic = {}
    for obj in objs:
        model = os.path.splitext(obj)[0]
        dic.setdefault(model, []).append(obj)

    return dic

def _write_individual_models(in_path: str, out_path: str, models_dict: Dict):
    prev_working_dir = os.getcwd()

    for model, files in models_dict.items():
        for file in files:
            model_out_path = os.path.join(out_path, model.replace(' ', '-'))
            if not os.path.exists(model_out_path):
                os.makedirs(model_out_path)
            shutil.copyfile(os.path.join(in_path.replace(' ', '\ '), file.replace(' ', '\ ')), os.path.join(model_out_path.replace(' ', '-'), file.replace(' ', '-')))




def split_models(in_path: str, out_path: str) -> bool:
    models_dict = _get_individual_models(in_path)
    _write_individual_models(in_path, out_path, models_dict)
    return True


if __name__ == '__main__':
    if sys.argv[1] in ['1', '2']:
        dataset_nr = int(sys.argv[1])
    else:
        raise ValueError('First argument has to be "1" or "2"')
    os.system('mv data data_last')
    split_models(f'Dataset/welding_objects_ds{dataset_nr}', 'data/train/models')



