'''
Inference and evalution code

Config-base inference... need to create a lot of tmp file
'''
import os
import shutil
import numpy as np
from datetime import datetime
from param_net import ParamNet
from data_loader import DataLoader
import json
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

TEMP_DIR = "./log/infer_tmp_" + datetime.now().strftime("%m-%d-%H_%M_%S")
# cpu version
device = torch.device("cpu", 0)


def recreate_dir(dir):
    import shutil
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def file_exist(file):
    assert os.path.exists(file) == True, f"{file}"


def change_load_model_path_in_config(tmp_file, target_model):
    '''
    Given a tempory config path, change its "load_model" key to "target_model"
    :param tmp_file: tempory file path
    '''
    # 1. write done
    with open(tmp_file, "r") as f:
        print(f"[debug] begin to load {tmp_file}")
        cont = json.load(f)
        cont[ParamNet.MODEL_OUTPUT_DIR_KEY] = target_model

    with open(tmp_file, "w") as f:
        json.dump(cont, f, indent=4)

    # 2. verification
    with open(tmp_file, 'r') as f:
        cont = json.load(f)
        assert cont[
            ParamNet.
            MODEL_OUTPUT_DIR_KEY] == target_model, f"verify {tmp_file} failed for {cont[ParamNet.MODEL_OUTPUT_DIR_KEY]}"


def build_net(template_file, network_weight_file):
    '''
    Build network by given a template config file, and weight file
    '''
    # 1. create tmp config file
    temp_config_file = os.path.basename(template_file)
    temp_config_file = os.path.join(TEMP_DIR, temp_config_file)
    # exit(0)
    shutil.copyfile(template_file, temp_config_file)
    file_exist(temp_config_file)
    # 2. change this temp config
    change_load_model_path_in_config(temp_config_file, network_weight_file)

    # 3. build the net
    return ParamNet(temp_config_file, device)


def load_feature(feature_file):
    with open(feature_file) as f:
        cont = json.load(f)
    return cont["input"], cont["output"]


def convert_to_plot3d_format(x):
    x_lst, y_lst, z_lst = [], [], []
    for i in range(int(len(x) / 3)):
        raw_x = x[3 * i]
        raw_y = x[3 * i + 1]
        raw_z = x[3 * i + 2]
        new_x = raw_x
        new_y = -raw_z
        new_z = raw_y
        x_lst.append(new_x)
        y_lst.append(new_y)
        z_lst.append(new_z)
    return x_lst, y_lst, z_lst


def draw_3d(v, ax):
    x_lst, y_lst, z_lst = convert_to_plot3d_format(v)
    # ax.plot3D(x_lst, y_lst, z_lst)
    ax.scatter(x_lst, y_lst, z_lst, alpha = 0.7)
    # ax.sca


def create_new_simulation_config(template_file, new_result_path,
                                 cloth_property):
    # 1. copy the template config to tmp config
    file_exist(template_file)
    temp_config_file = os.path.basename(template_file)
    temp_config_file = os.path.join(TEMP_DIR, temp_config_file)
    # exit(0)
    shutil.copyfile(template_file, temp_config_file)
    file_exist(temp_config_file)

    with open(temp_config_file, 'r') as f:
        cont = json.load(f)
        print(f"temp simulation config: cont {cont}")

    # 2. change temp config's cloth property
    if True:
        num_of_props = len(cloth_property)
        given_prop = cont["cloth_property"]
        cont["cloth_property"]["stretch_warp"] = float(cloth_property[0])
        cont["cloth_property"]["stretch_weft"] = float(cloth_property[1])
        cont["cloth_property"]["bending_warp"] = float(cloth_property[2])
        cont["cloth_property"]["bending_weft"] = float(cloth_property[3])

    # 3. change temp config's output simulation result path
    cont["network_inference_output_path"] = new_result_path

    # 4. change temp config's "enable_network_inference_mode"
    cont["enable_network_inference_mode"] = True

    with open(temp_config_file, 'w') as f:
        json.dump(cont, f)
    return temp_config_file


if __name__ == "__main__":
    # 1. input network weight, input geometry file
    # network_weight_file = "../output/04-21-22_17_46-0.001.pkl"
    network_weight_file = "../output/04-22-14_23_27-0.001.pkl"
    geo_file = r"..\data\\export_data\\1296items\\1000.json"
    template_network_config_file = r"../config/train_configs/train_config.json"
    template_se_config_file = r"../config/se_config.json"
    new_geo_file = os.path.join(os.getcwd(), TEMP_DIR, "new_geo.json")
    file_exist(network_weight_file)
    file_exist(geo_file)
    file_exist(template_network_config_file)
    recreate_dir(TEMP_DIR)
    X, gt_Y = load_feature(geo_file)
    X = torch.from_numpy(np.array(X)).type(torch.float)
    # 2. load the network, do inference
    net = build_net(template_network_config_file, network_weight_file)
    pred = net.infer(X)
    print(f"final pred {pred}")
    print(f"gt {gt_Y}")
    print(f"diff {pred - gt_Y}")

    # 3. put new prediction parameters into the simulation world, get the result
    ## 3.1 get the template se_config, copy this config, change its setting, return the new config path
    file_exist(template_se_config_file)
    temp_se_config = create_new_simulation_config(template_se_config_file,
                                                  new_geo_file, pred)
    temp_se_config = os.path.join(os.getcwd(), temp_se_config)
    print(f"temp se config {temp_se_config}")

    # exit(0)
    ## 3.2 call the C++ ./main.exe with the parameter new config, wait until it is finished,
    import subprocess

    # args = 'powershell', '-noprofile', '-command', 'set-location ../; ./main.exe ./config/se_config.json'
    args = 'powershell', '-noprofile', '-command', f'set-location ../; ./main.exe {temp_se_config}'
    subprocess.call(args)

    ## 3.3 done, begin to compare the result
    new_X, _ = load_feature(new_geo_file)

    print(f"new X shape {len(new_X)}")
    print(f"old X shape {len(X)}")

    fig1 = plt.figure()
    ax = Axes3D(fig1)
    draw_3d(X, ax)

    fig2 = plt.figure()
    ax = Axes3D(fig2)
    draw_3d(new_X, ax)
    plt.show()
