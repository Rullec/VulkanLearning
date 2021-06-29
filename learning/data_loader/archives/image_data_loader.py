import time
from torchvision import transforms
from data_loader import DataLoader
import os
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import sys

sys.path.append("../calibration")
from file_util import get_subdirs
from drawer_util import DynaPlotter
from image_data_loader_dist import get_mesh_data
import torch
from torchvision.transforms import RandomAffine, GaussianBlur


class ImageDataLoader(DataLoader):
    '''
    Depth image dataloader for network training 
    '''
    ENABLE_SPLIT_SAME_ROT_IMAGE_INTO_ONE_SET_KEY = "enable_split_same_rot_image_into_one_set"

    # def __init__(self, data_dir: str, train_perc: float, test_perc: float,
    #              batch_size: int, enable_log_prediction: bool,
    #              only_load_statistic_data: bool) -> None:
    # print("[log] image dataloader begin")
    def __init__(self, data_loader_config_dict,
                 only_load_statistic_data: bool):

        # super().__init__(data_dir,
        #                  train_perc,
        #                  test_perc,
        #                  batch_size,
        #                  enable_log_prediction,
        #                  only_load_statistic_data,
        #                  enable_data_augment=True,
        #                  select_validation_set_inside=True)
        self.enable_split_same_rot_image_into_one_set = data_loader_config_dict[
            self.ENABLE_SPLIT_SAME_ROT_IMAGE_INTO_ONE_SET_KEY]
        super().__init__(data_loader_config_dict, only_load_statistic_data)
        self._init_aug_torch()

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
        # print(f"data root dir {data_root_dir}")
        # exit()
        for mesh_data in get_subdirs(data_root_dir):
            mesh_data_lst.append(
                get_mesh_data(os.path.join(data_root_dir, mesh_data)))

        # each element of mesh_data_lst is all rotated data points of a property
        return mesh_data_lst

    def _init_aug_torch(self):
        ########### hard to train
        # self.affine = RandomAffine(degrees=(-10, 10), translate = (0.1, 0.1), scale = (1, 1.2), shear = None)
        # self.blur = GaussianBlur(kernel_size=5)
        # self.noise_gaussian_std = 0.02

        ########## small noise test： succ
        # self.affine = RandomAffine(degrees=(-5, 5), translate = (0.01, 0.01), scale = (1, 1), shear = None)
        # self.blur = GaussianBlur(kernel_size=1)
        # self.noise_gaussian_std = 0

        ######### bigger noise test： succ
        self.affine = RandomAffine(degrees=(-5, 5),
                                   translate=(0.07, 0.07),
                                   scale=(1.0, 1.0),
                                   shear=None)
        self.blur = GaussianBlur(kernel_size=3)
        self.data_transform_affine_blur = transforms.Compose(
            [self.affine, self.blur])
        self.noise_gaussian_std = 0.04

    def aug_torch(self, all_imgs):
        # aug_st = time.time()
        torch_all_imgs = torch.from_numpy(np.array(all_imgs))
        # aug_convert = time.time()
        # print(f"aug input convert cost {aug_convert - aug_st} s")
        # height, width = all_imgs[0].shape[1], all_imgs[0].shape[2]

        # torch_all_imgs = self.affine(torch_all_imgs)
        # torch_all_imgs = self.blur(torch_all_imgs)
        torch_all_imgs = self.data_transform_affine_blur(torch_all_imgs)
        # aug_trans = time.time()
        # print(f"aug trans cost {aug_trans - aug_convert} s")

        noise = torch.randn_like(torch_all_imgs[0]) * self.noise_gaussian_std
        # aug_noise = time.time()
        # print(f"aug rand noise cost {aug_noise - aug_trans} s")

        torch_all_imgs += noise
        # aug_addnoise = time.time()
        # print(f"aug add noise cost {aug_addnoise - aug_noise} s")
        ret = np.array(torch_all_imgs)
        # aug_copy = time.time()
        # print(f"aug ret copy cost {aug_copy - aug_addnoise} s")
        return ret

    def _init_vars(self):
        '''
        initialize the train variables
        '''
        mesh_data_lst = ImageDataLoader.__get_pngs_and_features(self.data_dir)
        example_mesh_data = mesh_data_lst[0][0]
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

    def _load_data_of_this_property(self, mesh_data_of_this_property):

        mesh_data_of_this_property_sample_X_lst, mesh_data_of_this_property_sample_Y_lst = [], []
        for mesh_data_of_this_property_roti in mesh_data_of_this_property:
            X, Y = ImageDataLoader.__load_single_data(
                mesh_data_of_this_property_roti.png_files,
                mesh_data_of_this_property_roti.feature_file,
                self.enable_log_predction)
            if X is None or Y is None:
                print(
                    f"[warn] data {mesh_data_of_this_property_roti.png_files} {mesh_data_of_this_property_roti.feature_file} is broken, please clear"
                )
                continue
            mesh_data_of_this_property_sample_X_lst.append(X)
            mesh_data_of_this_property_sample_Y_lst.append(Y)
        return mesh_data_of_this_property_sample_X_lst, mesh_data_of_this_property_sample_Y_lst

    # def _calc_statistics(self, raw_X_lst, raw_Y_lst):

    #     return self.input_mean, self.input_std, self.output_mean, self.output_std

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
                    X_lst = cont[DataLoader.X_KEY]
                    Y_lst = cont[DataLoader.Y_KEY]
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
                        # for _id, _ in enumerate(mesh_data_lst):

                        mesh_data_of_this_property = mesh_data_lst[_id]

                        # load all data points of this property
                        mesh_data_of_this_property_sample_X_lst, mesh_data_of_this_property_sample_Y_lst = self._load_data_of_this_property(
                            mesh_data_of_this_property)

                        # mesh_data_of_this_property_sample_X_lst: 4 x [6, 300, 300]
                        X_lst.append(mesh_data_of_this_property_sample_X_lst)
                        Y_lst.append(mesh_data_of_this_property_sample_Y_lst)

                self.input_mean = np.stack([
                    init_rot_angle for prop in X_lst for init_rot_angle in prop
                ],
                                           axis=0).mean(axis=0)
                self.input_std = np.stack([
                    init_rot_angle for prop in X_lst for init_rot_angle in prop
                ],
                                          axis=0).std(axis=0)
                self.output_mean = np.stack([
                    init_rot_angle for prop in Y_lst for init_rot_angle in prop
                ],
                                            axis=0).mean(axis=0)
                self.output_std = np.stack([
                    init_rot_angle for prop in Y_lst for init_rot_angle in prop
                ],
                                           axis=0).std(axis=0)

                # self.input_mean, self.input_std, self.output_mean, self.output_std = self._calc_statistics(
                #     X_lst, Y_lst)
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
                # print(f"len X lst {len(X_lst)}")
                # print(f"len X lst[0] {len(X_lst[0])}")
                # print(f"shape X lst[0][0] {X_lst[0][0].shape}")
                # exit()
                # print(f"ave pkl to {pkl_file} begin")
                import pickle
                with open(pkl_file, 'wb') as f:
                    pickle.dump(cont, f, protocol=4)
                # print(f"save pkl to {pkl_file} succ")

            np.clip(self.input_std, 1e-2, None, self.input_std)
            np.clip(self.output_std, 1e-2, None, self.output_std)
            self._dump_statistics()
        # print(f"input mean {self.input_mean}")
        # print(f"input std {self.input_std}")
        # print(f"output mean {self.output_mean}")
        # print(f"output std {self.output_std}")
        # exit()
        if only_load_statistic_data_ == False:
            print("begin to normalize data")
            for i in tqdm(range(len(X_lst))):
                # for i in range(len(X_lst)):
                for property_all_data_id in range(len(X_lst[i])):
                    # plot.add(X_lst[i][property_all_data_id][0], "old data")
                    # single_point_shape = X_lst[i][property_all_data_id].shape
                    # print(single_point_shape)
                    # exit()
                    # before = X_lst[i, size/2, size/2]
                    # print(X_lst[i][property_all_data_id].shape)
                    X_lst[i][property_all_data_id] = (
                        (X_lst[i][property_all_data_id] - self.input_mean) /
                        self.input_std).astype(np.float32)
                    Y_lst[i][property_all_data_id] = (
                        (Y_lst[i][property_all_data_id] - self.output_mean) /
                        self.output_std).astype(np.float32)
                    # plot.add(X_lst[i][property_all_data_id][0], "new data")
                    # plot.add((X_lst[i][property_all_data_id] * self.input_std +
                    #           self.input_mean)[0], "restored data")
                    # plot.show()
                    # print(X_lst[i][property_all_data_id].shape)
                    # print(self.input_mean.shape)

                    # exit()
                # after = X_lst[i, size/2, size/2]
                # print(f"{before} -> {after}")
            # print("succ to normalize the data")

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
            # exit()
            # print(f"X_lst[0] type {type(X_lst[0])}")
            # print(f"X_lst[0][0] type {type(X_lst[0][0])}")
            # print(f"train id {train_id}")
            # print(f"test id {test_id}")
            if len(train_id) == 1:
                self.train_X = [list(itemgetter(*train_id)(X_lst))]
                self.train_Y = [list(itemgetter(*train_id)(Y_lst))]
            else:
                self.train_X = list(itemgetter(*train_id)(X_lst))
                self.train_Y = list(itemgetter(*train_id)(Y_lst))
            if len(test_id) == 1:
                self.test_X = [list(itemgetter(*test_id)(X_lst))]
                self.test_Y = [list(itemgetter(*test_id)(Y_lst))]
            else:
                self.test_X = list(itemgetter(*test_id)(X_lst))
                self.test_Y = list(itemgetter(*test_id)(Y_lst))
            # print(f"self.train_X[0] type {type(self.train_X[0])}")
            # exit()
            # print(f"self.train_X[0] shape {(self.train_X[0]).shape}")
            # print(len(self.test_X))
            # print(type(self.test_X))
            # print(len(self.test_X[0]))
            # print(type(self.test_X[0]))
            # print(self.test_X[0][0].shape)
            # exit()
            self.test_X = [
                init_rot_angle for prop in self.test_X
                for init_rot_angle in prop
            ]
            self.test_Y = [
                init_rot_angle for prop in self.test_Y
                for init_rot_angle in prop
            ]
            self.train_X = [
                init_rot_angle for prop in self.train_X
                for init_rot_angle in prop
            ]
            self.train_Y = [
                init_rot_angle for prop in self.train_Y
                for init_rot_angle in prop
            ]
            # exit()
            # print(len(self.test_X))
            # print(len(self.test_Y))
            # print(len(self.train_X))
            # print(len(self.train_Y))
            # print(type(self.test_X[0]))
            # print(np.array(self.test_X[0]).shape)
            # exit()
            print(
                f"test X len {len(self.test_X)} X[0] shape {self.test_X[0].shape}"
            )
            print(
                f"test Y len {len(self.test_Y)}, Y[0] shape {self.test_Y[0].shape}"
            )
            print(
                f"train X len {len(self.train_X)} X[0] shape {self.train_X[0].shape}"
            )
            print(
                f"train Y len {len(self.train_Y)}, Y[0] shape {self.train_Y[0].shape}"
            )
            # exit()

    def get_train_data(self):

        st = 0
        while st < len(self.train_X):

            # st_time = time.time()
            incre = min(st + self.batch_size, len(self.train_X)) - st
            if incre <= 0:
                break
            output_X, output_Y = self.train_X[st:st +
                                              incre], self.train_Y[st:st +
                                                                   incre]
            # ed_time = time.time()
            # print(f"get output_X from train X list cost {ed_time - st_time} s")

            # import pickle
            # with open("img.pkl", 'wb') as f:
            #     pickle.dump(output_X, f)
            #     print("dump done")
            #     exit()
            if self.enable_data_augment == True:
                for id in range(len(output_X)):
                    # re permutate the view channels
                    low = 0
                    high = output_X[id].shape[0] - 1
                    if low < high:
                        rand = np.random.randint(0, high)
                        output_X[id] = np.roll(output_X[id], rand, axis=0)
                output_X = self.aug_torch(output_X)
            yield output_X, output_Y
            st += incre

    def get_validation_data(self):

        st = 0
        while st < len(self.test_X):
            incre = min(st + self.batch_size, len(self.test_X)) - st
            if incre <= 0:
                break
            output_X, output_Y = self.test_X[st:st +
                                             incre], self.test_Y[st:st + incre]
            yield np.array(output_X), np.array(output_Y)
            st += incre

    def get_all_data(self):
        yield np.array(self.train_X + self.test_X), np.array(self.train_Y +
                                                             self.test_Y)
