from device_util import get_depth_mode_str
from matplotlib.pyplot import plot
import numpy as np
import os
from file_util import load_pkl, save_pkl, not_clear_and_create_dir
import process_data_scene
from take_depth_image_and_compare import cut_depth_image_by_given_window


def get_scene():
    config_path = "./config/data_process.json"
    scene = process_data_scene.process_data_scene()
    scene.Init(config_path)
    return scene


if __name__ == "__main__":
    origin_dir = "calibration/current_dir.log"
    target_dir = "calibration/cutted_dir.log"
    not_clear_and_create_dir(target_dir)
    files = [os.path.join(origin_dir, i) for i in os.listdir(origin_dir)]
    scene = get_scene()
    # 1. begin to load the pkl files
    multiview_lst = [load_pkl(pkl) for pkl in files]

    for _idx, cur_multiview in enumerate(multiview_lst):
        for __idx, cur_view in enumerate(cur_multiview):
            cur_multiview[__idx] = cut_depth_image_by_given_window(
                scene, cur_view)
        # save current pkl
        filename = os.path.split(files[_idx])[-1]
        filename = filename[:filename.find(".")]
        name = f"{target_dir}/{filename}-cutted.pkl"
        save_pkl(name, cur_multiview)
        print(f"current save pkl to {name}")

        # cut_depth_image_by_given_window(scene, j)

        # for j in depth_image:
        #     ploter.add(j)
        # ploter.show()