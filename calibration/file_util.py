from log_util import *
from PIL import Image
import pickle
import numpy as np
import shutil
import os
import json


def save_png_image(path, image):
    '''
        save a png image to a path
    '''
    assert path[-4:] == ".png"
    with open(path, 'wb') as f:
        img = Image.fromarray(image)
        img.save(path)
        debug_print(f"save png image to {path}")


def load_png_image(path):
    assert os.path.exists(path) == True
    image = Image.open(path)
    image = np.asarray(image, dtype=np.float32)
    assert len(image.shape) == 2
    return image


def save_pkl(path, obj):
    assert type(path) is str, "path is the first param"
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        log_print(f"save pkl to {path}")


def clear_and_create_dir(dir_path):
    if os.path.exists(dir_path) == True:
        shutil.rmtree(dir_path)
    ret = os.makedirs(dir_path)
    print(f"create {dir_path}")


def not_clear_and_create_dir(dir_path):
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)


def load_pkl(path):
    assert os.path.exists(path)
    with open(path, 'rb') as f:
        cont = pickle.load(f)
    return cont


def convert_nparray_to_json(value):
    return json.dumps(np.squeeze(value).tolist())


def get_basename(fullname):
    filename = os.path.split(fullname)[-1]
    basename = filename[:filename.find(".")]
    return basename


def get_subdirs(root_dir):
    all_dirs = [
        os.path.join(root_dir, i) for i in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, i))
    ]
    all_dirs.sort(key=os.path.getctime)
    for _idx in range(len(all_dirs)):
        all_dirs[_idx] = os.path.split(all_dirs[_idx])[-1]
    return all_dirs


def load_json(path):
    assert os.path.join(path)
    with open(path) as f:
        return json.load(f)


def save_json(path, cur_dict):
    with open(path, 'w') as f:
        json.dump(cur_dict, f)


def get_subfiles(root_dir):
    all_files = [
        os.path.join(root_dir, i) for i in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, i))
    ]
    all_files.sort(key=os.path.getctime)
    for _idx in range(len(all_files)):
        all_files[_idx] = os.path.split(all_files[_idx])[-1]
    return all_files
