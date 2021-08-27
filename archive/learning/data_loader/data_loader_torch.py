from torch.utils.data import Dataset, DataLoader
import os
from multiprocessing import Pool
from tqdm import tqdm
import platform
import numpy as np
from multiprocessing import Pool
import time
import json
from itertools import chain
import cv2
class HDF5Dataset(Dataset):
    def __init__(self,
                 grp_lst,
                 input_mean,
                 input_std,
                 output_mean,
                 output_std,
                 load_all_data_into_mem=False,
                 data_aug=None):
        self.data_aug = data_aug
        self.grp_lst = grp_lst
        self.input_mean = input_mean[...]
        self.input_std = input_std[...]
        self.output_mean = output_mean[...]
        self.output_std = output_std[...]
        self.load_all_data_into_mem = load_all_data_into_mem
        self.__build_map_idx2dst()

        # check whether do we need to load all data into the mem
        if self.load_all_data_into_mem is True:
            self.__load_all_data_into_mem()

    def get_hdf5_dataset_from_group(grp):
        return [cur_grp[cur_key] for cur_key in tqdm(cur_grp)]

    def __build_map_idx2dst(self):
        self.grp_st = []
        self.grp_ed = []
        self.grp_length = []
        for cur_grp in self.grp_lst:
            idx_lst = [int(i) for i in cur_grp.keys()]
            if len(idx_lst) == 0:
                self.grp_st.append(0)
                self.grp_ed.append(0)
                self.grp_length.append(0)
            else:
                st = min(idx_lst)
                ed = max(idx_lst) + 1
                self.grp_st.append(st)
                self.grp_ed.append(ed)
                self.grp_length.append(ed - st)

    def __load_all_data_into_mem(self):
        size = len(self)
        self.inputs = []
        self.outputs = []
        for i in tqdm(range(size)):
            input, output = self.__getitem_from_disk(i)
            self.inputs.append(input)
            self.outputs.append(output)
        print(f"[dataset] load all data into mem done")
        # exit()

    def __determine_group(self, index):
        for idx in range(len(self.grp_lst)):

            if index >= self.grp_st[idx] and index < self.grp_ed[idx]:
                return idx
        assert False

    def __getitem_from_disk(self, index):
        assert index < np.sum(self.grp_length) and index >= 0
        import time
        # st = time.time()
        grp_idx = self.__determine_group(index)
        # ed1 = time.time()
        # print(f"load data1 cost {ed1 - st}")
        # print(f"determine group cost {ed - st}")
        # print(self.grp_lst[grp_idx].keys())
        dst = self.grp_lst[grp_idx][f"{index}"]

        # ed2 = time.time()
        # print(f"load data2 cost {ed2 - ed1}")
        input = dst[...]
        output = dst.attrs["label"]
        # ed3 = time.time()
        # print(f"load data3 cost {ed3 - ed2}")
        return input, output

    def __getitem_from_mem(self, index):
        input = self.inputs[index]
        output = self.outputs[index]

        return input, output

    def __getitem__(self, index):
        if self.load_all_data_into_mem == True:
            input, output = self.__getitem_from_mem(index)
        else:
            input, output = self.__getitem_from_disk(index)
        # if this data is images, rotate its axis
        if len(input.shape) == 3:
            num_of_views = input.shape[0]
            shift = np.random.randint(0, num_of_views)
            input = np.roll(input, shift, axis=0)

        # if the augmentation is enabled, apply it
        if self.data_aug is not None:
            input = self.data_aug(input)
        return input, output
        # input = np.ones([4, 360, 480], dtype=np.float32)
        # output = np.ones([3], dtype=np.float32)
        # return input, output

    def __len__(self):
        return np.sum(self.grp_length)

    def get_input_size(self):
        input, _ = self.__getitem__(0)
        return input.shape

    def get_output_size(self):
        _, output = self.__getitem__(0)
        return output.shape

    def shuffle(self):
        print(f"shuffle continue")
        pass


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batchsize):
        self.dataset = dataset
        if platform.system() == "Linux":
            workers = 6
        elif platform.system() == "Windows":
            workers = 0
        else:
            raise ValueError(f"unsupported platform {platform.system()}")
        self.input_mean = self.dataset.input_mean
        self.input_std = self.dataset.input_std
        self.output_mean = self.dataset.output_mean
        self.output_std = self.dataset.output_std

        super().__init__(self.dataset,
                         batch_size=batchsize,
                         shuffle=True,
                         num_workers=workers,
                         persistent_workers=(workers != 0),
                         prefetch_factor=2)

    def input_unnormalize(self, val):
        return val * self.input_std + self.input_mean

    def input_unnormalize(self, val):
        return val * self.input_std + self.input_mean

    def output_normalize(self, val):
        return (val - self.output_mean) / self.output_std

    def output_normalize(self, val):
        return (val - self.output_mean) / self.output_std

    def get_input_size(self):
        return self.dataset.get_input_size()

    def get_output_size(self):
        return self.dataset.get_output_size()