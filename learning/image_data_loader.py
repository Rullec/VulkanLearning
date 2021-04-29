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
                 batch_size: int, enable_log_prediction: bool) -> None:
        print("[log] image dataloader begin")

        super().__init__(data_dir, train_perc, test_perc, batch_size,
                         enable_log_prediction)

    @staticmethod
    def __load_single_data(png_path, feature_path, enable_log_pred):
        '''
        Given a pair of png path and feature json path, return the image and feature in np.ndarray
        '''
        assert os.path.exists(png_path), f"{png_path}"
        assert os.path.exists(feature_path), f"{feature_path}"
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
                feature = np.log(feature).astype(np.float32)
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
        assert len(feature_files) == len(png_files)
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

    def _load_data(self):
        pkl_file = os.path.join(self.data_dir, DataLoader.PKL_FILE_NAME)
        if os.path.exists(pkl_file) == True:
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

            X_lst, Y_lst = [], []
            if os.path.exists(self.data_dir) == True:
                png_files, feature_files = ImageDataLoader.__get_pngs_and_features(
                    self.data_dir)
                for f, _ in enumerate(
                        tqdm(png_files, f"Loading data from {self.data_dir}")):
                    # if f[-4:] == "json":
                    X, Y = ImageDataLoader.__load_single_data(
                        os.path.join(self.data_dir, png_files[f]),
                        os.path.join(self.data_dir, feature_files[f]),
                        self.enable_log_predction)
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
        # print(
        #     f"input mean min {np.min(np.abs(self.input_mean))} max {np.max(np.abs( self.input_mean))}"
        # )
        # print(
        #     f"input std min {np.min(np.abs(self.input_std))} max {np.max(np.abs( self.input_std))}"
        # )
        # print(
        #     f"output mean min {np.min(np.abs(self.output_mean))} max {np.max(np.abs( self.output_mean))}"
        # )
        # print(
        #     f"output std min {np.min(np.abs(self.output_std))} max {np.max(np.max( self.output_std))}"
        # )

        # print("begin to normalize the data")
        # for i in range(X_lst.shape[0]):
        # print(f"raw res = \n{X_lst[0]}")
        # print(f"raw Y {Y_lst}")
        # print(f"output mean {self.output_mean}")
        # print(f"output std {self.output_std}")
        # exit(0)
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

        self.train_X = list(itemgetter(*train_id)(X_lst))
        self.train_Y = list(itemgetter(*train_id)(Y_lst))
        self.test_X = list(itemgetter(*test_id)(X_lst))
        self.test_Y = list(itemgetter(*test_id)(Y_lst))

        # add another dim
        self.train_X = np.expand_dims(self.train_X, axis=1)
        # self.train_Y = np.expand_dims(self.train_Y, axis=1)
        self.test_X = np.expand_dims(self.test_X, axis=1)
        # print(f"test Y {Y_lst}")
        # exit(0)
        # self.test_Y = np.expand_dims(self.test_Y, axis=1)
        # print(self.train_X.shape)
        # print(self.train_Y.shape)
        # print(self.test_X.shape)
        # print(self.test_Y.shape)
        # from PIL import Image
        # img = Image.fromarray(self.train_X[0], 'F')
        # img = Image.fromarray(self.train_X[0], 'P')
        # img.show()
        # print("succ to split the data")
        # print(f"train X shape {len(self.train_X)}")
        # print(f"train Y shape {len(self.train_Y)}")
        # print(f"new res = \n{X_lst[0]}")
        
        # print(f"mean = \n{self.input_mean}")
        # print(f"std = \n{self.input_std}")
        # exit(0)