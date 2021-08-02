import sys
sys.path.append("../")
# from .dali_file_reader_torch import CustomDataLoader, HDF5Dataset
from ..data_loader_torch import CustomDataLoader, HDF5Dataset
import numpy as np
from tqdm import tqdm
import time
import torch
from copy import deepcopy


def reshape_input(input: torch.Tensor):
    # print(f"input.shape {input.shape}")
    num_of_data = input.shape[0]
    num_of_view = input.shape[1]

    # old_input = deepcopy(input)
    # print(input.shape[2])
    # print(input.shape[3])
    # exit()
    # st = time.time()
    input = torch.reshape(input,
                          shape=(num_of_data * num_of_view, input.shape[2],
                                 input.shape[3]))
    # print(f"reshape cost {time.time() - st}")

    # for i in range(input.shape[0]):
    #     idx0 = i // num_of_view
    #     idx1 = i % num_of_view
    #     diff = old_input[idx0, idx1] - input[i]
    #     assert torch.norm(diff) < 1e-10
    #     print(f"row {i} diff {torch.norm(diff)}, verify done")

    return input


def reshape_output(output, num_of_view):
    # print(f"output shape {output.shape}")
    new_output = torch.repeat_interleave(input=output,
                                         repeats=num_of_view,
                                         dim=0)
    return new_output
    # print(new_output)
    # new_output = output.repeat(4, 1)
    # print(f"new_output shape {new_output.shape}")
    # exit()


class DALIFileReaderBasedTorch(object):
    ARCHIVE_PATH = "archive.hdf5"

    def __init__(self, batch_size, grp_handles, load_all_data_into_mem,
                 input_mean, input_std, output_mean, output_std):
        # do not apply data aug in torch
        dataset = HDF5Dataset(grp_handles,
                                input_mean,
                                input_std,
                                output_mean,
                                output_std,
                                load_all_data_into_mem,
                                data_aug=None)
        self.num_of_data = len(dataset)
        self.batch_size = batch_size
        self.torch_loader = CustomDataLoader(dataset, batch_size)
        self.num_of_view = input_mean.shape[0]
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std

        self.new_iter = self.create_iter()

    def create_iter(self):
        # st = time.time()
        ret = iter(self.torch_loader)
        # ed = time.time()
        # print(f"create iter cost {ed - st}")
        return ret


    def __iter__(self):
        return self

    def __len__(self):
        length = int(self.num_of_data // self.batch_size)
        if self.num_of_data % self.batch_size != 0:
            length += 1
        return length

    def __next__(self):
        try:

            # fetch_st = time.time()
            input, output = next(self.new_iter)
            # fetch_ed = time.time()
            # print(f"fetch cost {fetch_ed - fetch_st}")
            input = np.expand_dims(reshape_input(input).numpy(), -1)
            output = reshape_output(output, self.num_of_view).numpy()
            # ed = time.time()
            # print(f"torch based file reader cost {ed - st}")
            # print(f"input shape {input.shape}")
            # print(f"output shape {output.shape}")
            return (input, output)
        except StopIteration:
            self.new_iter = self.create_iter()
            raise StopIteration

    def shuffle(self):
        pass