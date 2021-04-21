from operator import itemgetter
import numpy as np
import os
import json
import torch
# from torch.utils.data.dataset import Dataset as DatasetTorch
# from torch.utils.data.dataloader import DataLoader as DataLoaderTorch

# class DataSet():
#     def __init__(self, root_dir, files, input_mean, input_std, output_mean,
#                  output_std) -> None:
#         '''
#         DataSet, mimic torch's dataset
#         :param root_dir: The path of root directory of data
#         :param files: a list of filenames of data samples
#         :param input_mean/input_std/output_mean/output_std: The statistic variables of input data, optional
#         '''
#         # super().__init__()
#         self.root_dir = root_dir
#         self.files = files
#         self.input_mean = input_mean
#         self.input_std = input_std
#         self.output_mean = output_mean
#         self.output_std = output_std
#         self.data_lst = []
#         self.__validate()
#         self.__load_all()

#     def __load_all(self):
#         from multiprocessing import Pool
#         pool = Pool(10)

#         def handle(filename, input_mean, input_std, output_mean, output_std):
#             with open(filename) as f:
#                 cont = json.load(f)
#             for i in range(len(cont["output"])):
#                 cont["output"][i] = np.log(cont["output"][i])

#             if (input_mean is not None) and (input_std is not None):
#                 cont["input"] = (cont["input"] - input_mean) / input_std
#             if (output_mean is not None) and (output_std is not None):
#                 cont["output"] = (cont["output"] - output_mean) / output_std
#             return cont

#         params = [(os.path.join(self.root_dir, i), self.input_mean,
#                    self.input_std, self.output_mean, self.output_std)
#                   for i in self.files]
#         self.data_lst = pool.map(handle, params)
#         # from tqdm import tqdm
#         # for idx in tqdm(range(len(self.files)), "Loading dataset..."):

#             # self.data_lst.append(cont)

#     def __validate(self):
#         '''
#         Validate whether the loaded files exist in the given directory
#         '''
#         assert type(self.files) == list
#         for i in self.files:
#             assert True == os.path.exists(os.path.join(self.root_dir, i))

#     def __len__(self):
#         return len(self.files)

#     def __normalize(self, data, mean, std):
#         '''
#         normalize the given data
#         '''
#         return list((np.array(data) - mean) / std)

#     def __getitem__(self, idx):
#         '''
#         load "idx" data from the file
#         '''
#         return self.data_lst[idx]


class DataLoader():
    PKL_FILE_NAME = "train_data.pkl"

    def __init__(self, data_dir: str, train_perc: float, test_perc: float,
                 batch_size: int, enable_log_prediction: bool) -> None:
        '''
        DataLoader inherited from the torch dataloader
        :param data_dir: data root directory
        :train_perc: the split percentage of trainset
        :test_perc: 1-train_perc
        :batch_size:
        :enable_log_prediction: if True, apply np.log on the data output vector before the normalization.
            It can make the resulted distribution better
        '''
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.train_perc = train_perc / (train_perc + test_perc)
        self.test_perc = test_perc / (train_perc + test_perc)
        self.enable_log_predction = enable_log_prediction
        self.__init_vars()
        self.__load_data()
        # self.__split_data()

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size

    def __init_vars(self):
        assert os.path.exists(self.data_dir), f"{self.data_dir}"
        self.num_of_data = len(os.listdir(self.data_dir))
        tmp_name = os.listdir(self.data_dir)[0]
        X, Y = DataLoader.__load_single_data(
            os.path.join(self.data_dir, tmp_name), self.enable_log_predction)
        self.input_size = len(X)

        self.output_size = len(Y)

    def get_input_mean(self):
        return self.input_mean

    def get_input_std(self):
        return self.input_std

    def get_output_mean(self):
        return self.output_mean

    def get_output_std(self):
        return self.output_std

    @staticmethod
    def __load_single_data(file_path, enable_log_pred):
        assert os.path.exists(file_path), f"{file_path}"

        with open(file_path) as f:
            cont = json.load(f)
            if ("input" in cont) and ("output" in cont):
                X = np.array(cont["input"], dtype=np.float32)
                if enable_log_pred == True:
                    Y = np.log(np.array(cont["output"], dtype=np.float32))
                else:
                    Y = np.array(cont["output"], dtype=np.float32)
                return X, Y
            else:
                return None

    def __load_data(self):
        '''
        deprecated API without the usage of torch generator
        '''
        pkl_file = os.path.join(self.data_dir, DataLoader.PKL_FILE_NAME)
        if os.path.exists(pkl_file) == True:
            import pickle
            with open(pkl_file, 'rb') as f:
                cont = pickle.load(f)
                X_lst = cont["X"]
                Y_lst = cont["Y"]
        else:
            from tqdm import tqdm
            X_lst, Y_lst = [], []
            if os.path.exists(self.data_dir) == True:
                for f in tqdm(os.listdir(self.data_dir),
                              f"Loading data from {self.data_dir}"):
                    # if f[-4:] == "json":
                    tar_f = os.path.join(self.data_dir, f)
                    X, Y = DataLoader.__load_single_data(
                        tar_f, self.enable_log_predction)
                    X_lst.append(X)
                    Y_lst.append(Y)
            X_lst = np.array(X_lst)
            Y_lst = np.array(Y_lst)
            cont = {"X": X_lst, "Y": Y_lst}
            import pickle
            with open(pkl_file, 'wb') as f:
                pickle.dump(cont, f)

        self.input_mean = X_lst.mean(axis=0)
        self.input_std = X_lst.std(axis=0)
        self.output_mean = Y_lst.mean(axis=0)
        self.output_std = Y_lst.std(axis=0)

        X_lst = (X_lst - self.input_mean) / self.input_std
        Y_lst = (Y_lst - self.output_mean) / self.output_std
        # print(f"raw Y = {Y_lst}")
        # print(f"exp Y = {np.exp(Y_lst)}")
        # print(f"output mean = {self.output_mean}")
        # print(f"output std = {self.output_std}")
        # exit(0)

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

    def get_validation_data(self):
        st = 0
        while st < len(self.test_X):
            incre = min(st + self.batch_size, len(self.test_X)) - st
            if incre <= 0:
                break
            yield self.test_X[st:st + incre], self.test_Y[st:st + incre]
            st += incre

    def get_train_data(self):
        st = 0
        while st < len(self.train_X):
            incre = min(st + self.batch_size, len(self.train_X)) - st
            if incre <= 0:
                break
            yield self.train_X[st:st + incre], self.train_Y[st:st + incre]
            st += incre

    def __shuffle_train_data(self):
        train_size = len(self.train_X)
        perm = np.random.permutation(train_size)
        self.train_X = list(itemgetter(*perm)(self.train_X))
        self.train_Y = list(itemgetter(*perm)(self.train_Y))

    def shuffle(self):
        self.__shuffle_train_data()
        # print(f"X lst shape {X_lst.shape}")
        # print(f"Y lst shape {Y_lst.shape}")
        # assert len(X_lst.shape) == 2
        # assert len(Y_lst.shape) == 2
        # assert X_lst.shape[0] == Y_lst.shape[0]
        # return X_lst, Y_lst

    # def get_torch_dataloader(self):
    #     '''
    #     return the train & test generator, which can be used to train
    #     '''
    #     # 1. check whether the split info is supplied
    #     train_names, test_names = self._get_split_info()

    #     # 2. calculate the mean and standard
    #     if True:
    #         print(
    #             "[debug] begin to calculate mean & std of input & out features..."
    #         )
    #         all_set = DataSet(self.data_dir, train_names + test_names, None,
    #                           None, None, None)
    #         loader = DataLoaderTorch(all_set,
    #                                  batch_size=len(all_set),
    #                                  num_workers=1)
    #         data = next(iter(loader))
    #         input = torch.vstack(data["input"]).transpose(0, 1)
    #         output = torch.vstack(data["output"]).transpose(0, 1)

    #         min_std = 1e-3
    #         self.input_std = torch.maximum(
    #             self.input_std,
    #             torch.ones_like(self.input_std) * min_std).numpy()
    #         self.output_std = torch.maximum(
    #             self.output_std,
    #             torch.ones_like(self.output_std) * min_std).numpy()

    #     # create the train dataset loader
    #     self.train_loader = DataLoaderTorch(DataSet(self.data_dir, train_names,
    #                                                 self.input_mean,
    #                                                 self.input_std,
    #                                                 self.output_mean,
    #                                                 self.output_std),
    #                                         batch_size=self.batch_size,
    #                                         shuffle=True,
    #                                         num_workers=0)
    #     # create the validation dataset loader
    #     self.validation_loader = DataLoaderTorch(DataSet(
    #         self.data_dir, test_names, self.input_mean, self.input_std,
    #         self.output_mean, self.output_std),
    #                                              batch_size=self.batch_size,
    #                                              shuffle=True,
    #                                              num_workers=0)
    #     return self.train_loader, self.validation_loader