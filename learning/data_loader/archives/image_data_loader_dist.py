from ntpath import join
import os
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import sys
sys.path.append("../calibration")
from file_util import get_subdirs
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as torchDataLoader


def load_single_data(png_files, feature_path, enable_log_pred):
    '''
    Given a pair of png path and feature json path, return the image and feature in np.ndarray
    '''
    for i in png_files:
        assert os.path.exists(i), f"{i}"
    assert os.path.exists(feature_path), f"{feature_path}"

    try:
        # 1. load the image
        img_lst = []
        for png_file in png_files:
            image = Image.open(png_file)
            image = np.asarray(image, dtype=np.float32)
            assert len(image.shape) == 2, f"{png_file} {image.shape}"
            img_lst.append(image)
        img_lst = np.stack(img_lst, axis=0)

        # 2. load the feature
        with open(feature_path) as f:
            cont = json.load(f)
            feature = np.array(cont["feature"])
            if enable_log_pred == True:
                feature = np.log(feature)
            feature = feature.astype(np.float32)
    except Exception as e:
        print(e)
        img_lst = None
        feature = None
    return img_lst, feature




class MeshData:
    def __init__(self, png_files, feature_file):
        self.png_files = png_files
        self.feature_file = feature_file


def get_mesh_data(root_dir):
    feature_file = os.path.join(root_dir, "feature.json")
    # 1. level0: init rotation angle
    mesh_lst = []
    for init_rot_subdir in get_subdirs(root_dir):
        for cam_sub_dir in get_subdirs(os.path.join(root_dir,
                                                    init_rot_subdir)):
            png_dir = os.path.join(root_dir, init_rot_subdir, cam_sub_dir)
            png_files = [os.path.join(png_dir, i) for i in os.listdir(png_dir)]
            mesh_lst.append(MeshData(png_files, feature_file))
    return mesh_lst


def get_pngs_and_features(data_root_dir: str):
    '''
    Given a data directory, fetch all filenames and split them into png files & json feature files
    '''
    # print(f"[debug] begin to get pngs and features data_dir")

    mesh_data_lst = []

    for mesh_data in get_subdirs(data_root_dir):
        mesh_data_lst = mesh_data_lst + get_mesh_data(
            os.path.join(data_root_dir, mesh_data))
    # print(len(mesh_data_lst))
    # for i in mesh_data_lst:
    #     print(f"feature file {i.feature_file} png files {i.png_files}")
    return mesh_data_lst


def default_loader(png_files, feature_file):
    # print(
    #     f"[debug] default loader, png files {png_files}, feature file {feature_file}"
    # )
    image, feature = load_single_data(png_files,
                                      feature_file,
                                      enable_log_pred=True)
    return image, feature


class dataset(Dataset):
    def __init__(self, inputs, output, loader=default_loader):
        #定义好 image 的路径
        self.images = inputs
        self.target = output
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.images[index]
        feature_path = self.target[index]
        img, target = self.loader(img_path, feature_path)
        return img, target

    def __len__(self):
        return len(self.images)


class ImageDataLoaderDist(DataLoader):
    '''
    Depth image dataloader for network training 
    '''
    def __init__(self, data_dir: str, train_perc: float, test_perc: float,
                 batch_size: int, enable_log_prediction: bool,
                 only_load_statistic_data: bool) -> None:
        # print("[log] image dataloader begin")

        super().__init__(data_dir, train_perc, test_perc, batch_size,
                         enable_log_prediction, only_load_statistic_data)

    def _init_vars(self):
        '''
        initialize the train variables
        '''
        mesh_data_lst = get_pngs_and_features(self.data_dir)
        example_mesh_data = mesh_data_lst[0]
        # verify succ, pick the first one to load
        image, feature = load_single_data(example_mesh_data.png_files,
                                          example_mesh_data.feature_file,
                                          self.enable_log_predction)

        self.input_size = image.shape
        self.output_size = feature.shape
        # print(f"input size {self.input_size}")
        # print(f"output size {self.output_size}")
        # print("init vars succ")
        # exit(0)

    def calc_statistics(self):
        train_iter = iter(self.train_loader)
        test_iter = iter(self.test_loader)

        input_mean_lst = []
        output_mean_lst = []
        input_std_lst = []
        output_std_lst = []
        size_lst = []
        try:
            while True:
                X, Y = train_iter.next()
                X = np.array(X)
                Y = np.array(Y)
                # print(f"X len {len(X)}")
                # print(f"X shape {X.shape}")

                # print(f"mean(X) shape {np.mean(X, axis=0).shape}")
                # print(f"Y shape {Y.shape}")
                input_mean_lst.append(np.mean(X, axis=0))
                output_mean_lst.append(np.mean(Y, axis=0))
                input_std_lst.append(np.std(X, axis=0))
                output_std_lst.append(np.std(Y, axis=0))
                size_lst.append(X.shape[0])
        except StopIteration as e:
            pass
        try:
            while True:
                X, Y = test_iter.next()
                X = np.array(X)
                Y = np.array(Y)
                input_mean_lst.append(np.mean(X, axis=0))
                output_mean_lst.append(np.mean(Y, axis=0))
                input_std_lst.append(np.std(X, axis=0))
                output_std_lst.append(np.std(Y, axis=0))
                size_lst.append(X.shape[0])
        except StopIteration as e:
            pass

        self.input_mean = np.sum([
            input_mean_lst[i] * size_lst[i] for i in range(len(input_mean_lst))
        ],
                                 axis=0) / (np.sum(size_lst))
        self.output_mean = np.sum([
            output_mean_lst[i] * size_lst[i]
            for i in range(len(output_mean_lst))
        ],
                                  axis=0) / (np.sum(size_lst))

        self.input_std = np.sqrt(
            np.sum([
                size_lst[i] * np.square(input_std_lst[i])
                for i in range(len(size_lst))
            ],
                   axis=0) / np.sum(size_lst))
        self.output_std = np.sqrt(
            np.sum([
                size_lst[i] * np.square(output_std_lst[i])
                for i in range(len(size_lst))
            ],
                   axis=0) / np.sum(size_lst))
        np.clip(self.input_std, 1e-2, None, self.input_std)
        np.clip(self.output_std, 1e-2, None, self.output_std)

        print(f"input mean shape {self.input_mean.shape}")
        print(f"input std shape {self.input_std.shape}")
        print(f"output mean shape {self.output_mean}")
        print(f"output std shape {self.output_std}")
        # exit()

    def _load_data(self, only_load_statistic_data_):
        load_stat_succ = False
        if only_load_statistic_data_ == True:
            # 1. begin to load statisic
            load_stat_succ = self._load_statistics()
            assert load_stat_succ, True
        else:
            # load the data and the statistics together
            mesh_data_lst = get_pngs_and_features(self.data_dir)
            size = len(mesh_data_lst)
            train_size = int(self.train_perc * size)
            perm = np.random.permutation(size)
            train_id = perm[:train_size]
            test_id = perm[train_size:]
            png_files_lst = []
            feature_file_lst = []

            for i in mesh_data_lst:
                png_files_lst.append(i.png_files)
                feature_file_lst.append(i.feature_file)

            # begin to split
            from operator import itemgetter

            train_X_file = list(itemgetter(*train_id)(png_files_lst))
            train_Y_file = list(itemgetter(*train_id)(feature_file_lst))
            test_X_file = list(itemgetter(*test_id)(png_files_lst))
            test_Y_file = list(itemgetter(*test_id)(feature_file_lst))

            train_set = dataset(inputs=train_X_file, output=train_Y_file)
            test_set = dataset(inputs=test_X_file, output=test_Y_file)
            self.train_loader = torchDataLoader(train_set,
                                                batch_size=self.batch_size,
                                                shuffle=True)

            self.test_loader = torchDataLoader(test_set,
                                               batch_size=self.batch_size,
                                               shuffle=True)

            self.calc_statistics()
            self._dump_statistics()

    def get_validation_data(self):
        # while True:
        for i, data in enumerate(self.test_loader):
            X, Y = data
            X = (X - self.input_mean) / self.input_std
            Y = (Y - self.output_mean) / self.output_std
            # print(f"get valid data, X shape {X.shape}")
            # exit()
            yield X, Y

    def get_train_data(self):
        # while True:
        for i, data in enumerate(self.train_loader):
            X, Y = data
            X = (X - self.input_mean) / self.input_std
            Y = (Y - self.output_mean) / self.output_std
            yield X, Y

    def __shuffle_train_data(self):
        pass

    def shuffle(self):
        pass
        # self.__shuffle_train_data()