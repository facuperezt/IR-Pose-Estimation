import os
import sys
from utils.compatibility import listdir
from typing import Dict
import shutil

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
            model_out_path = os.path.join(out_path, model)
            if not os.path.exists(model_out_path):
                os.makedirs(model_out_path)
            shutil.copyfile(os.path.join(in_path, file), os.path.join(model_out_path, file))




def split_models(in_path: str, out_path: str) -> bool:
    models_dict = _get_individual_models(in_path)
    _write_individual_models(in_path, out_path, models_dict)
    return True


if __name__ == '__main__':
    os.system('mv data data_last')
    split_models('Dataset/welding_objects_ds1', 'data/train/models')



