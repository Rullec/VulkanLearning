import os
import sys

from file_util import load_pkl
import numpy as np
from drawer_util import *

sys.path.append("../learning")

from train import build_net, init_env


def load_image_data(data_dir):
    assert os.path.exists(data_dir) == True

    data_lst = []
    for file in [os.path.join(data_dir, i) for i in os.listdir(data_dir)]:
        img_data = load_pkl(file)
        # unit: mm
        img_data = np.uint8(np.array(img_data) * 1e-3 * 255.99)
        data_lst.append(img_data)
    return data_lst


def compare_result(train_data, real_data):
    size = 1 + 1 + real_data.shape[0]
    rows, cols = calculate_subplot_size(size)
    plotter = DynaPlotter(rows, cols, iterative_mode=False)
    # enable_train = train_data[0] > 130
    # enable_real = real_data[0] > 130
    # enable_mask = np.logical_and(enable_train, enable_real)
    diff = np.squeeze(train_data[0] - real_data[0])
    plotter.add(diff, f"diff")
    plotter.add(np.squeeze(train_data[0]), f"train data")
    # diff[enable_mask == False] = 0
    for i in range(real_data.shape[0]):
        plotter.add(np.squeeze(real_data[i]), f"real data {i}")
    plotter.show()
    exit()
    # plotter = DynaPlotter(2, 4, iterative_mode=False)
    # for i in range(4):
    #     plotter.add(train_data[i], f"train-{i}")
    # for i in range(4):
    #     plotter.add(real_data[i], f"real-{i}")
    # plotter.show()


if __name__ == "__main__":
    # 1. load the image data
    data_dir = "manual_fix_dir.log"
    img_lst = load_image_data(data_dir)

    # 2. begin to build network, load the agent, load the mean and standard

    device = init_env()
    conf_path = "..\config\\train_configs\\conv_conf.json"
    net_type = build_net(conf_path)
    net = net_type(conf_path, device, only_load_statistic_data=False)

    # 3. do inference
    input_mean = net.data_loader.input_mean
    input_std = net.data_loader.input_std
    output_X, output_Y = next(net.data_loader.get_validation_data())
    # output_X = net.data_loader.get_train_data()

    print(f"input mean shape {input_mean.shape}")
    print(f"input std shape {input_std.shape}")
    train_input_example = net.data_loader.unnormalize_input_data(output_X)

    print(f"train input example shape {train_input_example.shape}")
    # exit(0)
    # data_input = np.expand_dims(img_lst[0][0], axis=0)
    # data_input = np.expand_dims(data_input, axis=0)
    # train_input_example = np.expand_dims(train_input_example, axis=0)
    real_input_example = np.array(img_lst)
    print(f"real data shape {real_input_example.shape}")
    # print(data_input.shape)
    # exit()
    real_output = net.infer(real_input_example)
    print(f"real_output {real_output}")

    train_output = net.infer(train_input_example)
    print(f"train_output {train_output}")

    compare_result(train_input_example[0], real_input_example)