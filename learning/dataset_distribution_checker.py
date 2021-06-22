'''
    check whether the test data is in the same distribution with the train data
'''
import sys
from data_loader import DataLoader
import numpy as np
sys.path.append("../calibration")
from drawer_util import DynaPlotter, calculate_subplot_size
import os
from file_util import load_pkl


def load_dataset(pkl_filename):
    assert os.path.exists(pkl_filename) == True
    data = load_pkl(pkl_filename)
    assert type(data) is dict
    print(data.keys())
    X_lst = data[DataLoader.X_KEY]
    Y_lst = data[DataLoader.Y_KEY]
    input_mean = data[DataLoader.INPUT_MEAN_KEY]
    input_std = data[DataLoader.INPUT_STD_KEY]
    output_mean = data[DataLoader.OUTPUT_MEAN_KEY]
    output_std = data[DataLoader.OUTPUT_STD_KEY]
    return X_lst, Y_lst, input_mean, input_std, output_mean, output_std
    # dataset = load_pkl(pkl_path)
    # print(type(dataset))
    # print(dataset.keys())


def load_captured_image(pkl_path):
    assert os.path.exists(pkl_path) == True
    data = load_pkl(pkl_path)
    data = np.uint8(np.array(data) * 1e-3 * 255.99)
    return data


def eye_metric(real_data_lst, captured_image):
    print(f"real data lst {len(real_data_lst)}")
    print(f"captured_image {type(captured_image)}")
    print(captured_image[0].shape)
    num_of_imgs = len(real_data_lst) + 1
    rows, cols = calculate_subplot_size(num_of_imgs)
    plot = DynaPlotter(rows, cols, iterative_mode=False)
    for _idx, data in enumerate(real_data_lst):
        plot.add(data[0][0], f"train {_idx}")
    plot.add(captured_image[0], "captured")
    plot.show()

    return


if __name__ == "__main__":
    # 1. set the config
    dataset_path = "D:\\SimpleClothSimulator\\data\\export_data\\1view_smallset\\train_data_1percent.pkl"
    captured_image_path = "D:\\SimpleClothSimulator\\calibration\\no_background_dir.log\\0.pkl"

    metric = "eye"
    # metric = "l2"
    # metric = "pca"
    # metric = "perpixel_gaussian"  # calculate the distance between the normalized real data and the zero origin

    # 2. load the data
    X_lst, Y_lst, input_mean, input_std, output_mean, output_std = load_dataset(
        dataset_path)
    captured_image = load_captured_image(captured_image_path)

    # 3.
    if metric == "eye":
        eye_metric(X_lst, captured_image)
