from log_util import *
from PIL import Image
import pickle
import shutil
import os


def save_png_image(path, image):
    '''
        save a png image to a path
    '''
    assert path[-4:] == ".png"
    with open(path, 'wb') as f:
        img = Image.fromarray(image)
        img.save(path)
        debug_print(f"save png image to {path}")


def save_pkl(path, obj):
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
