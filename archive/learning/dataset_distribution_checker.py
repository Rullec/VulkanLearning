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


def eye_metric(real_data_lst, label_lst, captured_image):
    num_of_imgs = len(real_data_lst)
    batch_images = 32
    cur_st = 0

    while num_of_imgs > 0:
        cur_image_num = num_of_imgs
        if num_of_imgs > batch_images:
            cur_image_num = batch_images

        rows, cols = calculate_subplot_size(cur_image_num + 1)
        cur_data_lst = real_data_lst[cur_st:cur_st + cur_image_num]

        plot = DynaPlotter(rows, cols, iterative_mode=False)
        for _idx, data in enumerate(cur_data_lst):

            prop_mean = f"{np.mean(label_lst[cur_st + _idx]):4.1f}"
            plot.add(cur_data_lst[_idx][0][0], f"mean {prop_mean}")
        plot.add(captured_image[0], "captured")
        plot.show()
        cur_st += cur_image_num
        num_of_imgs -= cur_image_num

    return


from sklearn.metrics.pairwise import pairwise_distances


def l2_metric(train_data_lst, label_lst, captured_image):
    train_data_lst = np.array(train_data_lst)
    train_data_lst = train_data_lst.reshape((-1, ) + train_data_lst.shape[2:])
    # print(f"train_data_lst.shape {train_data_lst.shape}")
    captured_image = captured_image[0, :]

    captured_image = np.expand_dims(np.expand_dims(captured_image, axis=0),
                                    axis=0)
    # print(f"captured_image.shape {captured_image.shape}")
    train_data_lst = np.vstack([train_data_lst, captured_image])

    train_data_lst = train_data_lst.reshape((train_data_lst.shape[0], -1))
    # print(f"final train_data.shape {train_data_lst.shape}")
    print(f"begin to calc pairwise")
    dist_mat = pairwise_distances(train_data_lst)
    size = dist_mat.shape[0]
    samples = 40
    rows, cols = calculate_subplot_size(samples + 3)
    plot = DynaPlotter(rows, cols, iterative_mode=False)
    plot.add_histogram(dist_mat[size - 1, :], "captured data")
    for _idx, i in enumerate(range(0, size - 1, int(size / samples) + 1)):
        # print(_idx)
        plot.add_histogram(dist_mat[i, :], f"train data {i}")
    plot.show()

    capture_feature = dist_mat[dist_mat.shape[0] - 1, :]
    capture_feature[dist_mat.shape[0] - 1] = np.max(capture_feature)
    capture_feature_mean = np.mean(capture_feature)
    capture_feature_min = np.min(capture_feature)
    capture_feature_max = np.max(capture_feature)
    capture_feature_std = np.std(capture_feature)
    print(f"captured mean {capture_feature_mean}, std {capture_feature_std}, min {capture_feature_min} max {capture_feature_max}")

    for _idx, i in enumerate(range(0, size - 1, int(size / 10) + 1)):
        train_dist = dist_mat[i, :]
        train_dist[i] = np.mean(train_dist)
        train_dist[-1] = np.mean(train_dist)
        train_dist_mean = np.mean(train_dist)
        train_dist_min = np.min(train_dist)
        train_dist_max = np.max(train_dist)
        train_dist_std = np.std(train_dist)
        print(f"train data mean {train_dist_mean}, std {train_dist_std}, min {train_dist_min} max {train_dist_max}")
    print(f"end to calc pairwise, dist_mat shape {dist_mat.shape}")


if __name__ == "__main__":
    # 1. set the config
    dataset_path = "D:\\SimpleClothSimulator\\data\\export_data\\1view_smallset\\train_data_1tenth.pkl"
    # dataset_path = "D:\\SimpleClothSimulator\\data\\export_data\\1view_smallset\\train_data.pkl"
    captured_image_path = "D:\\SimpleClothSimulator\\calibration\\manual_fix_dir.log\\new.pkl"

    # metric = "eye"
    metric = "l2"
    # metric = "pca"
    # metric = "perpixel_gaussian"  # calculate the distance between the normalized real data and the zero origin

    # 2. load the data
    X_lst, Y_lst, input_mean, input_std, output_mean, output_std = load_dataset(
        dataset_path)
    captured_image = load_captured_image(captured_image_path)

    # 3.
    if metric == "eye":
        eye_metric(X_lst, Y_lst, captured_image)
    elif metric == "l2":
        l2_metric(X_lst, Y_lst, captured_image)
