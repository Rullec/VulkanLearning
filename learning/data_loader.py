from operator import itemgetter
import numpy as np
import os
import json

class DataLoader():
    PKL_FILE_NAME = "train_data.pkl"
    X_KEY = "X"
    Y_KEY = "Y"
    INPUT_MEAN_KEY = "input_mean"
    INPUT_STD_KEY = "input_std"
    OUTPUT_MEAN_KEY = "output_mean"
    OUTPUT_STD_KEY = "output_std"

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
        # print(self.train_perc)
        # print(self.test_perc)
        # exit()
        self.enable_log_predction = enable_log_prediction
        self._init_vars()
        self._load_data()
        # self.__split_data()

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size

    def _init_vars(self):
        assert os.path.exists(self.data_dir), f"{self.data_dir}"
        self.num_of_data = len(os.listdir(self.data_dir))
        tmp_name = os.listdir(self.data_dir)[0]
        X, Y = DataLoader._load_single_data(
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
    def _load_single_data(file_path, enable_log_pred):
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

    def _load_data(self):
        '''
        load the data and compress to a pkl file
        '''
        pkl_file = os.path.join(self.data_dir, DataLoader.PKL_FILE_NAME)
        if os.path.exists(pkl_file) == True:
            import pickle
            with open(pkl_file, 'rb') as f:
                cont = pickle.load(f)
                X_lst = cont[DataLoader.X_KEY]
                Y_lst = cont[DataLoader.Y_KEY]
                self.input_mean = cont[DataLoader.INPUT_MEAN_KEY]
                self.input_std = cont[DataLoader.INPUT_STD_KEY]
                self.output_mean = cont[DataLoader.OUTPUT_MEAN_KEY]
                self.output_std = cont[DataLoader.OUTPUT_STD_KEY]
        else:
            from tqdm import tqdm
            X_lst, Y_lst = [], []
            if os.path.exists(self.data_dir) == True:
                for f in tqdm(os.listdir(self.data_dir),
                              f"Loading data from {self.data_dir}"):
                    # if f[-4:] == "json":
                    tar_f = os.path.join(self.data_dir, f)
                    X, Y = DataLoader._load_single_data(
                        tar_f, self.enable_log_predction)
                    X_lst.append(X)
                    Y_lst.append(Y)
            X_lst = np.array(X_lst, dtype=np.float32)
            Y_lst = np.array(Y_lst, dtype=np.float32)
            self.input_mean = X_lst.mean(axis=0)
            self.input_std = X_lst.std(axis=0)
            self.output_mean = Y_lst.mean(axis=0)
            self.output_std = Y_lst.std(axis=0)

            cont = {
                DataLoader.X_KEY: X_lst,
                DataLoader.Y_KEY: Y_lst,
                DataLoader.INPUT_MEAN_KEY: self.input_mean,
                DataLoader.INPUT_STD_KEY: self.input_std,
                DataLoader.OUTPUT_MEAN_KEY: self.output_mean,
                DataLoader.OUTPUT_STD_KEY: self.output_std
            }
            import pickle
            with open(pkl_file, 'wb') as f:
                pickle.dump(cont, f)

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
        # print(f"train id {train_id}")
        # print(f"test id {test_id}")
        # exit()
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