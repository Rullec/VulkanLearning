from torch.utils.data import Dataset
from tqdm import tqdm
import os
import json
from multiprocessing import Pool
import cv2
import numpy as np
from itertools import chain


class OpencvDataset(Dataset):
    def __init__(self, datapoint_dir_lst, input_mean, input_std, output_mean,
                 output_std, data_aug):
        self.data_aug = data_aug
        self.input_mean = input_mean
        self.input_std = input_std
        self.input_std_inv = 1.0 / input_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.output_std_inv = 1.0 / output_std
        self.__check_mesh_data(datapoint_dir_lst)

    def load_data_info(dir_name):
        json_path = os.path.join(dir_name, "feature.json")
        with open(json_path, 'r') as f:
            output = json.load(f)["feature"]
        input_dir_lst = []
        output_lst = []
        for dir1 in os.listdir(dir_name):
            all_dir1 = os.path.join(dir_name, dir1)
            if os.path.isdir(all_dir1):
                for dir2 in os.listdir(all_dir1):
                    all_dir2 = os.path.join(all_dir1, dir2)
                    input_dir_lst.append(all_dir2)
                    output_lst.append(output)
        return input_dir_lst, output_lst

    def __check_mesh_data(self, datapoint_dir_lst):
        self.input_dir_lst = []
        self.output_lst = []
        import platform
        pool = Pool(28 if platform.system() == "Linux" else 1)

        for ret in tqdm(pool.imap_unordered(
                OpencvDataset.load_data_info,
                datapoint_dir_lst,
        ),
                        "inspecting png groups",
                        total=len(datapoint_dir_lst)):
            self.input_dir_lst.append(ret[0])
            self.output_lst.append(
                (np.array(ret[1], dtype=np.float32) - self.output_mean) *
                self.output_std_inv)

        self.input_dir_lst = list(chain.from_iterable(self.input_dir_lst))
        self.output_lst = list(chain.from_iterable(self.output_lst))

    def __getitem__(self, index):
        assert index >= 0 and index < len(self.input_dir_lst)
        dir = self.input_dir_lst[index]
        pngs = [
            os.path.join(dir, i) for i in os.listdir(dir)
            if i.find(".png") != -1
        ]
        # st = time.time()
        inputs = np.array([cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in pngs],
                           dtype=np.float32)
        # raw_max = np.amax(inputs)
        # raw_min = np.amin(inputs)
        inputs = (inputs -
                  self.input_mean) * self.input_std_inv
        
        # new_data = inputs * self.input_std + self.input_mean
        # max_d = np.amax(new_data)
        # min_d = np.amin(new_data)
        # if min_d < -1 or max_d > 255:
        #     print(f"max {max_d} min {min_d} in opencv dataset")
        #     exit()
        # print(f"max {max_d} min {min_d} in opencv dataset")
        # print(f"raw max {raw_max} raw min {raw_min}")
       
        
        # re-arrange the axis
        if len(inputs.shape) == 3:
            num_of_views = inputs.shape[0]
            shift = np.random.randint(0, num_of_views)
            inputs = np.roll(inputs, shift, axis=0)

        # if self.data_aug is not None:
        #     inputs = self.data_aug(inputs)
        #     print(f"do data augmentation!")

        output = self.output_lst[index]
        return inputs, output
        # ed = time.time()
        # # print(res)
        # print(res.shape)
        # print(ed - st)
        # exit()

    def __len__(self):
        return len(self.input_dir_lst)

    def get_input_size(self):
        input, _ = self.__getitem__(0)
        return input.shape

    def get_output_size(self):
        _, output = self.__getitem__(0)
        return output.shape

    def shuffle(self):
        print(f"shuffle continue")
        pass
