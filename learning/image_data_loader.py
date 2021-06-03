from data_loader import DataLoader
import os
from PIL import Image
import numpy as np
import json
from tqdm import tqdm


class ImageDataLoader(DataLoader):
    '''
    Depth image dataloader for network training 
    '''
    def __init__(self, data_dir: str, train_perc: float, test_perc: float,
                 batch_size: int, enable_log_prediction: bool,
                 only_load_statistic_data: bool) -> None:
        # print("[log] image dataloader begin")

        super().__init__(data_dir, train_perc, test_perc, batch_size,
                         enable_log_prediction, only_load_statistic_data)

    @staticmethod
    def __load_single_data(png_path, feature_path, enable_log_pred):
        '''
        Given a pair of png path and feature json path, return the image and feature in np.ndarray
        '''
        assert os.path.exists(png_path), f"{png_path}"
        assert os.path.exists(feature_path), f"{feature_path}"

        try:
            # 1. load the image
            image = Image.open(png_path)
            # print(f"image info {image.format} {image.size} {image.mode}")
            # image.show()
            image = np.asarray(image, dtype=np.float32)
            assert len(image.shape) == 2, f"{png_path} {image.shape}"
            # print(image.shape)
            # exit(0)
            # new_image = np.mean(image_array[:, :], axis)
            # print(f"old image shape {image.size}")
            # print(f"new image shape {new_image}")
            # confirm that this image is grayscale
            # new_image_verify = image_array[:, :, 0]
            # diff_norm = np.linalg.norm(new_image_verify - new_image)
            # assert diff_norm < 1e-10

            # 2. load the feature
            with open(feature_path) as f:
                cont = json.load(f)
                feature = np.array(cont["feature"])
                if enable_log_pred == True:
                    feature = np.log(feature)
                feature = feature.astype(np.float32)
        except Exception as e:
            print(e)
            image = None
            feature = None
        return image, feature

    @staticmethod
    def __get_pngs_and_features(data_dir: str):
        '''
        Given a data directory, fetch all filenames and split them into png files & json feature files
        '''
        all_names = os.listdir(data_dir)
        png_files = []
        feature_files = []

        # divide and verify the data
        def judge_suffix(name: str, suffix: str) -> bool:
            return name[-len(suffix):] == suffix

        for file in all_names:
            if judge_suffix(file, "png"):
                png_files.append(file)
            elif judge_suffix(file, "json"):
                feature_files.append(file)
        assert len(feature_files) == len(
            png_files), f"{len(feature_files)} !={len(png_files)}"
        for i in range(len(png_files)):
            png = png_files[i]
            feature = feature_files[i]
            png_base = png.split('.')[0]
            feature_base = feature.split('.')[0]
            assert png_base == feature_base, f"{png_base} != {feature_base}"
            # print(f"{png_base} == {feature_base}")
        return png_files, feature_files

    def _init_vars(self):
        '''
        initialize the train variables
        '''
        png_files, feature_files = ImageDataLoader.__get_pngs_and_features(
            self.data_dir)
        # verify succ, pick the first one to load
        image, feature = ImageDataLoader.__load_single_data(
            os.path.join(self.data_dir, png_files[0]),
            os.path.join(self.data_dir, feature_files[0]),
            self.enable_log_predction)

        self.input_size = image.shape
        self.output_size = feature.shape

    def _load_data(self, only_load_statistic_data_):
        load_stat_succ = False
        if only_load_statistic_data_ == True:
            # 1. begin to load statisic
            load_stat_succ = self._load_statistics()

        if (only_load_statistic_data_ == True
                and load_stat_succ == False) or (only_load_statistic_data_
                                                 == False):
            pkl_file = os.path.join(self.data_dir, DataLoader.PKL_FILE_NAME)
            if os.path.exists(pkl_file) == True:
                print(f"[debug] loading pkl from {pkl_file}")
                import pickle
                with open(pkl_file, 'rb') as f:
                    # print(f"begin to load pkl from {pkl_file}")
                    cont = pickle.load(f)
                    X_lst = cont[DataLoader.X_KEY].astype(np.float32)
                    Y_lst = cont[DataLoader.Y_KEY].astype(np.float32)
                    self.input_mean = cont[DataLoader.INPUT_MEAN_KEY]
                    self.input_std = cont[DataLoader.INPUT_STD_KEY]
                    self.output_mean = cont[DataLoader.OUTPUT_MEAN_KEY]
                    self.output_std = cont[DataLoader.OUTPUT_STD_KEY]

                    # print(f"done to load pkl from {pkl_file}")
            else:
                print(f"[debug] pkl not found, load pngs from {self.data_dir}")
                X_lst, Y_lst = [], []
                if os.path.exists(self.data_dir) == True:
                    png_files, feature_files = ImageDataLoader.__get_pngs_and_features(
                        self.data_dir)
                    for f, _ in enumerate(
                            tqdm(png_files,
                                 f"Loading data from {self.data_dir}")):
                        # if f[-4:] == "json":
                        X, Y = ImageDataLoader.__load_single_data(
                            os.path.join(self.data_dir, png_files[f]),
                            os.path.join(self.data_dir, feature_files[f]),
                            self.enable_log_predction)
                        if X is None or Y is None:
                            print(
                                f"[warn] data {png_files[f]} {feature_files[f]} is broken, please clear"
                            )
                            continue
                        X_lst.append(X)
                        Y_lst.append(Y)
                X_lst = np.array(X_lst)
                Y_lst = np.array(Y_lst)
                self.input_mean = X_lst.mean(axis=0)
                self.input_std = X_lst.std(axis=0)
                self.output_mean = Y_lst.mean(axis=0)
                self.output_std = Y_lst.std(axis=0)

                # exit(0)
                cont = {
                    DataLoader.X_KEY: X_lst,
                    DataLoader.Y_KEY: Y_lst,
                    DataLoader.INPUT_MEAN_KEY: self.input_mean,
                    DataLoader.INPUT_STD_KEY: self.input_std,
                    DataLoader.OUTPUT_MEAN_KEY: self.output_mean,
                    DataLoader.OUTPUT_STD_KEY: self.output_std
                }
                # print(f"ave pkl to {pkl_file} begin")
                import pickle
                with open(pkl_file, 'wb') as f:
                    pickle.dump(cont, f, protocol=4)
                # print(f"save pkl to {pkl_file} succ")

            np.clip(self.input_std, 1e-2, None, self.input_std)
            np.clip(self.output_std, 1e-2, None, self.output_std)
            self._dump_statistics()

        if only_load_statistic_data_ == False:
            print("begin to normalize data")
            for i in tqdm(range(X_lst.shape[0])):
                size = X_lst.shape[1]
                # before = X_lst[i, size/2, size/2]
                X_lst[i] = (X_lst[i] - self.input_mean) / self.input_std
                Y_lst[i] = (Y_lst[i] - self.output_mean) / self.output_std
                # after = X_lst[i, size/2, size/2]
                # print(f"{before} -> {after}")
            # print("succ to normalize the data")

            # print("begin to split the data")

            size = len(X_lst)
            train_size = int(self.train_perc * size)
            test_size = size - train_size
            perm = np.random.permutation(size)
            train_id = perm[:train_size]
            test_id = perm[train_size:]
            from operator import itemgetter

            self.test_X = np.expand_dims(list(itemgetter(*test_id)(X_lst)),
                                         axis=1)
            self.test_Y = list(itemgetter(*test_id)(Y_lst))
            self.train_Y = list(itemgetter(*train_id)(Y_lst))
            self.train_X = np.expand_dims(list(itemgetter(*train_id)(X_lst)),
                                          axis=1)
