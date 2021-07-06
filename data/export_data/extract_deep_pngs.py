from shutil import copyfile
import numpy as np
import sys
from tqdm import tqdm

sys.path.append("../../calibration")
from file_util import *
'''
Copy the image to the first level of the folder
'''

if __name__ == "__main__":
    origin_dir = "test_data"
    target_dir = "test_data_exposed"
    clear_and_create_dir(target_dir)

    for mesh_idx, mesh_dir in enumerate(tqdm(get_subdirs(origin_dir))):
        # 1. for each init rot
        origin_dir1 = os.path.join(origin_dir, mesh_dir)
        for init_rot_idx, init_rot_dir in enumerate(get_subdirs(origin_dir1)):
            origin_dir2 = os.path.join(origin_dir1, init_rot_dir)
            # 2. for each cam, a group of pngs
            for cam_idx, cam_dir in enumerate(get_subdirs(origin_dir2)):
                origin_dir3 = os.path.join(origin_dir2, cam_dir)
                png_files = get_subfiles(origin_dir3)
                for file_idx, cur_file in enumerate(png_files):
                    full_origin_path = os.path.join(origin_dir3, cur_file)
                    full_target_path = f"mesh{mesh_idx}_initrot_{init_rot_idx}_cam{cam_idx}_file{file_idx}.png"
                    full_target_path = os.path.join(target_dir, full_target_path) 
                    copyfile(full_origin_path, full_target_path)
