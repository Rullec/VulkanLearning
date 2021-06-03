from ntpath import join
from data_loader import DataLoader
import os
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
from image_data_loader_dist import get_subdirs, get_mesh_data

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
    def __load_single_data(png_files, feature_path, enable_log_pred):
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

    @staticmethod
    def __get_pngs_and_features(data_root_dir: str):
        '''
        Given a data directory, fetch all filenames and split them into png files & json feature files
        '''
        print(f"[debug] begin to get pngs and features data_dir")

        mesh_data_lst = []

        for mesh_data in get_subdirs(data_root_dir):
            mesh_data_lst = mesh_data_lst + get_mesh_data(
                os.path.join(data_root_dir, mesh_data))
        # print(len(mesh_data_lst))
        # for i in mesh_data_lst:
        #     print(f"feature file {i.feature_file} png files {i.png_files}")
        return mesh_data_lst

    def _init_vars(self):
        '''
        initialize the train variables
        '''
        mesh_data_lst = ImageDataLoader.__get_pngs_and_features(self.data_dir)
        example_mesh_data = mesh_data_lst[0]
        # verify succ, pick the first one to load
        image, feature = ImageDataLoader.__load_single_data(
            example_mesh_data.png_files, example_mesh_data.feature_file,
            self.enable_log_predction)

        self.input_size = image.shape
        self.output_size = feature.shape
        # print(f"input size {self.input_size}")
        # print(f"output size {self.output_size}")
        # print("init vars succ")
        # exit(0)

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
                    mesh_data_lst = ImageDataLoader.__get_pngs_and_features(
                        self.data_dir)
                    for _id, _ in enumerate(
                            tqdm(
                                mesh_data_lst,
                                f"Loading data from {os.path.split(self.data_dir)[-1] }"
                            )):
                        # if f[-4:] == "json":
                        mesh_data = mesh_data_lst[_id]
                        X, Y = ImageDataLoader.__load_single_data(
                            mesh_data.png_files, mesh_data.feature_file,
                            self.enable_log_predction)
                        # print(f"X shape {X.shape}")
                        # print(f"Y shape {Y.shape}")
                        if X is None or Y is None:
                            print(
                                f"[warn] data {mesh_data.png_files} {mesh_data.feature_file} is broken, please clear"
                            )
                            continue
                        X_lst.append(X)
                        Y_lst.append(Y)
                # print(len(X_lst))
                X_lst = np.stack(X_lst, axis=0)
                Y_lst = np.stack(Y_lst, axis=0)
                self.input_mean = X_lst.mean(axis=0)
                self.input_std = X_lst.std(axis=0)
                self.output_mean = Y_lst.mean(axis=0)
                self.output_std = Y_lst.std(axis=0)
                # print(f"X_lst shape {X_lst.shape}")
                # print(f"Y_lst shape {Y_lst.shape}")
                # exit(0)
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
            perm = np.random.permutation(size)
            train_id = perm[:train_size]
            test_id = perm[train_size:]
            from operator import itemgetter

            # self.test_X = np.expand_dims(list(itemgetter(*test_id)(X_lst)),
            #                              axis=1)
            # self.test_Y = list(itemgetter(*test_id)(Y_lst))
            # self.train_X = np.expand_dims(list(itemgetter(*train_id)(X_lst)),
            #                               axis=1)
            # self.train_Y = list(itemgetter(*train_id)(Y_lst))
            self.test_X = np.array(list(itemgetter(*test_id)(X_lst)))
            self.test_Y = list(itemgetter(*test_id)(Y_lst))
            self.train_X = np.array(list(itemgetter(*train_id)(X_lst)))
            self.train_Y = list(itemgetter(*train_id)(Y_lst))

            print(f"test X shape {self.test_X.shape}")
            print(f"test Y shape {len(self.test_Y)}, {self.test_Y[0].shape}")
            print(f"train X shape {self.train_X.shape}")
            print(
                f"train Y shape {len(self.train_Y)}, {self.train_Y[0].shape}")
            # exit()