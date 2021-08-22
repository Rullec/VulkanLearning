'''
Now lots of labels in sampled data is still in old format [500, 5000]
Convert them to new format [0, 100] by a segmented formula
'''
import sys
from tqdm import tqdm

sys.path.append("../../calibration")
from file_util import get_subdirs, get_subfiles, load_json, save_json
import os
import numpy as np

# def get_stretch_map_gui2sim():
#     stretch_map_guitosim = {
#         0.0: 1e3,
#         9.0: 1e4,
#         27.0: 1e5,
#         57.0: 4e5,
#         93.0: 4e6,
#         100.0: 1e7
#     }
#     return stretch_map_guitosim


def get_bending_map_gui2sim():
    bending_map_guitosim = {
        0.0: 0e0,
        10.0: 1e2,
        28.0: 1e3,
        48.0: 3e3,
        65.0: 2e4,
        83.0: 2e5,
        100.0: 2e6
    }
    return bending_map_guitosim


def get_bending_map_sim2gui():
    gui2sim = get_bending_map_gui2sim()
    sim2gui = {}
    for i in gui2sim:
        sim2gui[gui2sim[i]] = i
    return sim2gui


def convert_bending_key2value(key_value, cur_dict):
    last_id = len(cur_dict) - 1
    threshold = 1e-6
    dict_keys = list(cur_dict.keys())
    dict_values = list(cur_dict.values())
    # 1. exceed the boundary
    if key_value > dict_keys[last_id] - threshold:
        return dict_values[last_id]
    if key_value < dict_keys[0] + threshold:
        return dict_values[0]
    # 2. find the interval
    for st in range(0, len(cur_dict) - 1):
        st_key = dict_keys[st]
        ed_key = dict_keys[st + 1]
        st_value = dict_values[st]
        ed_value = dict_values[st + 1]
        gap = ed_key - st_key
        if (st_key <= key_value) and (ed_key >= key_value):
            return (1.0 - (key_value - st_key) / gap) * st_value + (
                1.0 - (ed_key - key_value) / gap) * ed_value

    assert False


def convert_bending_sim2gui(sim_value):
    sim2gui_dict = get_bending_map_sim2gui()
    gui_value = convert_bending_key2value(sim_value, sim2gui_dict)
    return gui_value


def convert_bending_gui2sim(gui_value):
    gui2sim_dict = get_bending_map_gui2sim()
    sim_value = convert_bending_key2value(gui_value, gui2sim_dict)
    return sim_value


def test_convert():
    gui_value_lst = np.linspace(0, 100, 1000)
    for gui_value in gui_value_lst:
        sim_value = convert_bending_gui2sim(gui_value)
        gui_value_re = convert_bending_sim2gui(sim_value)
        # print(f"gui value {gui_value}, sim_value {sim_value} gui_value_re {gui_value_re}")
        assert np.abs(
            gui_value_re - gui_value
        ) < 1e-3, f"gui value {gui_value}, gui_value_re {gui_value_re}"
    print(f"convert test succ")


# 1. convert for a depth image folder
def handle_old_depth_image_folder(folder):
    def bending_feature_old2new(value):
        assert type(value) == dict
        FEATURE_KEY = "feature"
        FEATURE_SIZE = 3
        assert FEATURE_KEY in value
        assert len(value[FEATURE_KEY]) == FEATURE_SIZE
        for i in range(FEATURE_SIZE):
            old_value = value[FEATURE_KEY][i]
            if old_value > 400:
                value[FEATURE_KEY][i] = convert_bending_sim2gui(old_value)
            # else:
            #     print(
            #         "[warn] old value must be greater than 400, do not convert it again"
            #     )
        return value

    subdirs = [os.path.join(folder, i) for i in get_subdirs(folder)]
    feature_basename = "feature.json"
    features = [os.path.join(i, feature_basename) for i in subdirs]
    assert all([os.path.exists(i) for i in features]) is True

    for cur_fea in features:
        now_value = load_json(cur_fea)
        now_value = bending_feature_old2new(now_value)
        save_json(cur_fea, now_value)
        # print(f"handle {cur_fea} done")
    print(f"handle depth image dir {folder} done")


def handle_old_mesh_data_folder(mesh_data_dir):
    def bending_output_old2new(value):
        assert type(value) == dict
        FEATURE_KEY = "output"
        FEATURE_SIZE = 3
        assert FEATURE_KEY in value
        assert len(value[FEATURE_KEY]) == FEATURE_SIZE
        for i in range(FEATURE_SIZE):
            old_value = value[FEATURE_KEY][i]
            if old_value > 400:
                value[FEATURE_KEY][i] = convert_bending_sim2gui(old_value)
            # else:
            #     print(
            #         "[warn] old value must be greater than 400, do not convert it again"
            #     )
        return value

    files = [
        os.path.join(mesh_data_dir, file) for file in os.listdir(mesh_data_dir)
        if file.find("json") != -1
    ]

    for file in tqdm(files):
        now_value = load_json(file)
        now_value = bending_output_old2new(now_value)
        save_json(file, now_value)
    print(f"handle mesh data dir {mesh_data_dir} done")


def get_mesh_dir_and_depth_image_dir(root_dir):
    assert os.path.exists(root_dir) == True

    def judge_is_mesh_dir(dir):
        subfiles = get_subfiles(dir)
        json_subfiles = [i for i in subfiles if i.find("json") != -1]
        num_of_json_files = len(json_subfiles)
        return num_of_json_files > 10

    def judge_is_depth_dir(dir):
        subdirs = get_subdirs(dir)
        num_of_dirs = len(subdirs)
        return num_of_dirs > 10

    subdirs = [os.path.join(root_dir, i) for i in get_subdirs(root_dir)]
    mesh_dir_lst = []
    depth_dir_lst = []
    for cur_dir in subdirs:
        is_mesh_dir = judge_is_mesh_dir(cur_dir)
        is_depth_dir = judge_is_depth_dir(cur_dir)
        if is_mesh_dir == is_depth_dir:
            f"[warn] directory {cur_dir} is unrecognizable"
            continue
        else:
            print(
                f"{cur_dir} is {'mesh_dir' if is_mesh_dir == True else 'depth_dir'}"
            )
            if is_mesh_dir == True:
                mesh_dir_lst.append(cur_dir)
            else:
                depth_dir_lst.append(cur_dir)
    return mesh_dir_lst, depth_dir_lst


if __name__ == "__main__":
    data_dir = r"D:\SimpleClothSimulator\data\export_data"
    mesh_dirs, depth_dirs = get_mesh_dir_and_depth_image_dir(data_dir)
    for i in mesh_dirs:
        print(f"mesh dir {i}")
    for i in depth_dirs:
        print(f"depth dir {i}")

    from multiprocessing import Pool
    pool = Pool(8)
    pool.map(handle_old_mesh_data_folder, mesh_dirs)
    print("mesh data handle done")
    pool.map(handle_old_depth_image_folder, depth_dirs)
    print("depth image data handle done")
    
    # for mesh_dir in mesh_dirs:
    #     try:
    #         handle_old_mesh_data_folder(mesh_dir)
    #     except Exception as e:
    #         print(e)
    # for depth_dir in depth_dirs:
    #     try:
    #         handle_old_depth_image_folder(mesh_dir)
    #     except Exception as e:
    #         print(e)