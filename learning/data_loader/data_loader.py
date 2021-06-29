from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate

class CustomDataset(Dataset):
    def __init__(self,
                 hdf5_grp_handle,
                 input_mean,
                 input_std,
                 output_mean,
                 output_std,
                 load_all_data_into_mem=False):
        self.hdf5_grp_handle = hdf5_grp_handle
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std,
        self.load_all_data_into_mem = load_all_data_into_mem

        # check whether do we need to load all data into the mem
        if self.load_all_data_into_mem is True:
            self.__load_all_data_into_mem()

    def __load_all_data_into_mem(self):
        size = self.__len__()
        self.inputs = []
        self.outputs = []
        for i in tqdm(range(size)):
            input, output = self.__getitem_from_disk(i)
            self.inputs.append(input)
            self.outputs.append(output)
        print(f"[dataset] load all data into mem done")
        # exit()

    def __getitem_from_disk(self, index):
        dst = self.hdf5_grp_handle[f"{index}"]
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
        # print(f"input shape {input.shape}")
        # print(f"output shape {output.shape}")
        return input, output

    def __len__(self):
        return len(self.hdf5_grp_handle.keys())

    def get_input_size(self):
        input, _ = self.__getitem__(0)
        return input.shape

    def get_output_size(self):
        _, output = self.__getitem__(0)
        return output.shape


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batchsize, data_aug=None):
        self.dataset = dataset
        self.data_aug = data_aug
        super().__init__(self.dataset,
                         batch_size=batchsize,
                         shuffle=True,
                         collate_fn=self.custom_collate)

    def custom_collate(self, batch):
        batch = default_collate(batch)
        if self.data_aug is not None:
            batch = self.data_aug(batch)
        return batch
        

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