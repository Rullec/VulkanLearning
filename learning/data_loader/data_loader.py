from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import platform
import numpy as np


class CustomDataset(Dataset):
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
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std,
        self.load_all_data_into_mem = load_all_data_into_mem
        self.__build_map_idx2dst()

        # check whether do we need to load all data into the mem
        if self.load_all_data_into_mem is True:
            self.__load_all_data_into_mem()

    def __build_map_idx2dst(self):
        print(f"building idx2dst map...")
        total_num = 0
        for cur_grp in self.grp_lst:
            total_num += len(cur_grp.keys())
        # print(f"total_num {total_num}")
        self.idx2dst = [None for _ in range(total_num)]

        idx = 0
        for cur_grp in self.grp_lst:
            for cur_key in cur_grp:
                self.idx2dst[idx] = cur_grp[cur_key]
                idx += 1

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

    def __getitem_from_disk(self, index):
        dst = self.idx2dst[index]
        output = dst.attrs["label"]
        input = dst[...]
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
        if self.data_aug is not None:
            assert len(input.shape) == 3
            num_of_views = input.shape[0]
            shift = np.random.randint(0, num_of_views)
            input = np.roll(input, shift, axis=0)
            # print(f"shift {shift}")

        if self.data_aug is not None:
            input = self.data_aug(input)
            # print("done dataaug, exit")
            # exit()
        return input, output

    def __len__(self):
        return len(self.idx2dst)

    def get_input_size(self):
        input, _ = self.__getitem__(0)
        return input.shape

    def get_output_size(self):
        _, output = self.__getitem__(0)
        return output.shape


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batchsize):
        self.dataset = dataset
        platform.system() == "Linux"
        if platform.system() == "Linux":
            workers = 12
        elif platform.system() == "Windows":
            workers = 0
        else:
            raise ValueError(f"unsupported platform {platform.system()}")
        super().__init__(
            self.dataset,
            batch_size=batchsize,
            shuffle=True,
            #  num_workers=workers, persistent_workers = True, prefetch_factor =2)
            num_workers=workers,
            persistent_workers=False,
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