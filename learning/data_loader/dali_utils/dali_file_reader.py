from PIL import Image
import numpy as np
import time
import os
from multiprocessing import Pool
import os
import json
from .dali_helper import get_subdirs
from tqdm import tqdm


class DALIPngReader(object):
    FEATURE_PATH = "feature.json"

    def __init__(self, batch_size, data_folder_list):
        '''
            the batch size is the num of data points
            batch_size * num_of_view = num_of_imgs
        '''
        self.batch_size = batch_size
        self.data_folder_list = data_folder_list

        # 1. load all png files, 4 groups
        self.pool = Pool(12)
        self.__load_path()

    def __load_path(self):
        print(f"begin to load from {len(self.data_folder_list)} samples ")
        self.num_of_view = None
        self.input_datapoint_dir_lst = []
        self.img_path_per_datapoint_lst = []
        self.label_lst = []
        for mesh_dir in self.data_folder_list:
            label_path = os.path.join(mesh_dir, DALIPngReader.FEATURE_PATH)
            assert os.path.exists(label_path), f"{label_path} doesn't exist"
            for init_rot in get_subdirs(mesh_dir):
                for cam in get_subdirs(init_rot):
                    cur_num = len(os.listdir(cam))
                    if self.num_of_view is None:
                        self.num_of_view = cur_num
                    else:
                        assert self.num_of_view == cur_num
                    self.input_datapoint_dir_lst.append(cam)
                    self.img_path_per_datapoint_lst.append(
                        [os.path.join(cam, i) for i in os.listdir(cam)])
                    self.label_lst.append(label_path)
        self.num_of_data = len(self.input_datapoint_dir_lst)
        self.num_of_img = self.num_of_view * self.num_of_data
        self.done = False
        self.cur_idx = 0
        print(
            f"[log] load {len(self.input_datapoint_dir_lst)} data list done, real img size = {self.num_of_img}"
        )

    def shuffle(self):
        # print(f"begin to do shuffle")
        # exit()
        size = len(self.input_datapoint_dir_lst)
        perb = np.random.permutation(size)
        # self.input_datapoint_dir_lst = self.input_datapoint_dir_lst[perb]
        # self.label_lst = self.label_lst[perb]
        self.input_datapoint_dir_lst = [
            self.input_datapoint_dir_lst[i] for i in perb
        ]
        self.label_lst = [self.label_lst[i] for i in perb]

    def __iter__(self):
        self.i = 0
        self.n = len(self.input_datapoint_dir_lst)
        return self

    def load_json_feature(feature_path):
        with open(feature_path, 'r') as f:
            return np.array(json.load(f)["feature"])

    def load_png(filenames):
        return [
            np.expand_dims(np.array(Image.open(i), dtype=np.float32), -1)
            for i in filenames
        ]

    def __next__(self):
        # st = time.time()
        batch_names = []
        labels = []

        if self.cur_idx >= self.num_of_data:
            self.shuffle()
            raise StopIteration

        st = time.time()
        for i in range(self.batch_size):
            batch_names.append(
                self.img_path_per_datapoint_lst[self.cur_idx %
                                                self.num_of_data])
            cur_label = DALIPngReader.load_json_feature(
                self.label_lst[self.cur_idx % self.num_of_data])
            for _ in range(self.num_of_view):
                labels.append(cur_label)
            self.cur_idx += 1

            if self.cur_idx >= self.num_of_data:
                break
        ed = time.time()
        print(f"load json cost {ed - st}")

        # ed = time.time()
        from itertools import chain
        # print(f"load json {ed - st}")
        st = time.time()
        batch_imgs = self.pool.map(DALIPngReader.load_png, batch_names)
        batch_imgs = [i for grp in batch_imgs for i in grp]
        ed = time.time()
        print(f"load png cost {ed - st}, png images {len(batch_imgs)}")
        # exit(0)
        return (batch_imgs, labels)


import h5py


class DALIHdf5Reader(object):
    ARCHIVE_PATH = "archive.hdf5"

    def __init__(self, batch_size, grp_handles, load_all_data_into_mem,
                 input_mean, input_std, output_mean, output_std):
        '''
            the batch size is the num of data points
            batch_size * num_of_view = num_of_imgs
        '''
        self.batch_size = batch_size
        self.load_all_data_into_mem = load_all_data_into_mem
        # 1. load all png files, 4 groups
        self.grp_handles = grp_handles
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.num_of_view = self.input_mean.shape[0]

        self.__load_h5py_dataset()
        if load_all_data_into_mem is True:
            self.__load_into_mem()

    def __load_into_mem(self):
        self.input_lst = []
        self.output_lst = []
        for i in self.dst_lst:
            self.input_lst.append(i[...])
            self.output_lst.append(i.attrs["label"])

    def __load_h5py_dataset(self):
        self.num_of_data = 0
        self.cur_idx = 0
        for handle in self.grp_handles:
            self.num_of_data += len(handle.keys())
        self.dst_lst = [None for _ in range(self.num_of_data)]

        for handle in self.grp_handles:
            for key in handle.keys():
                key_id = int(key)
                self.dst_lst[key_id] = handle[key]
        for i in self.dst_lst:
            assert i is not None

        self.index_map = [i for i in range(self.num_of_data)]

    def shuffle(self):
        self.index_map = np.random.permutation(self.num_of_data)

    def __iter__(self):
        self.i = 0
        self.n = self.num_of_data
        return self

    def __len__(self):
        length = int(self.num_of_data // self.batch_size)
        if self.num_of_data % self.batch_size != 0:
            length += 1
        return length

    def __next__(self):
        
        # 1. if stop
        if self.cur_idx >= self.num_of_data:
            self.shuffle()
            self.cur_idx = 0
            raise StopIteration

        batch_imgs = []
        labels = []
        # 2. else, get batch data
        for i in range(self.batch_size):
            data_idx = self.index_map[self.cur_idx]
            dst = self.dst_lst[data_idx]
            val = dst[...]
            assert len(val.shape) == 3
            shift = np.random.randint(0, val.shape[0])
            val = np.roll(val, shift, axis=0)
            label = dst.attrs["label"]
            self.cur_idx += 1

            for id in range(val.shape[0]):
                batch_imgs.append(np.expand_dims(val[id], -1))
                labels.append(label)
            # labels.append(label)

            if self.cur_idx >= self.num_of_data:
                break
            # print(val.shape)
            # print(label.shape)
            # exit()
        # batch_imgs = np.vstack(batch_imgs)
        return (batch_imgs, labels)


import torch


def profiling_the_iter(dali_iter):
    iters = 0
    total_st = time.time()
    while True:
        st = time.time()
        try:
            res = next(dali_iter)
            iters += 1
        except StopIteration:
            dali_iter.reset()
            # exit()
            break
        ed = time.time()
        print(f"cost {ed - st}")
    total_ed = time.time()
    print(f"avg cost {(total_ed - total_st) / iters}")
    print(f"total cost {total_ed - total_st}")


def check_reshape(old_img, new_img, num_of_view):
    old_size = old_img.shape[0]
    new_size = new_img.shape[0] * new_img.shape[1]
    assert old_size == new_size
    assert new_img.shape[1] == num_of_view

    for i in range(old_size):
        row = int(i / num_of_view)
        col = i % num_of_view
        diff = old_img[i] - new_img[row, col, :]
        assert torch.norm(diff) < 1e-3
    print(f"check reshape succ")
