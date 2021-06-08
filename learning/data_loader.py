from operator import itemgetter
import numpy as np
import os
import json


class DataLoader():
    PKL_FILE_NAME = "train_data.pkl"
    STATISTIC_FILE_NAME = "statistic.pkl"
    X_KEY = "X"
    Y_KEY = "Y"
    INPUT_MEAN_KEY = "input_mean"
    INPUT_STD_KEY = "input_std"
    OUTPUT_MEAN_KEY = "output_mean"
    OUTPUT_STD_KEY = "output_std"

    BATCH_SIZE_KEY = "batch_size"
    DATA_DIR_KEY = "data_dir"
    TRAIN_PERC_KEY = "train_perc"
    ENABLE_LOG_PREDICTION_KEY = "enable_log_prediction"
    ENABLE_DATA_AUGMENT_KEY = "enable_data_augment"
    BATCH_SIZE_KEY = "batch_size"
    ENABLE_SELECT_VALIDATION_SET_INSIDE_KEY = "enable_select_validation_set_inside"

    # def __init__(self, data_dir: str, train_perc: float, test_perc: float,
    #              batch_size: int, enable_log_prediction: bool,
    #              only_load_statistic_data: bool, enable_data_augment: bool,
    #              select_validation_set_inside: bool) -> None:
    def __init__(self, data_loader_config_dict,
                 only_load_statistic_data) -> None:
        '''
        DataLoader inherited from the torch dataloader
        :param data_dir: data root directory
        :train_perc: the split percentage of trainset
        :test_perc: 1-train_perc
        :batch_size:
        :enable_log_prediction: if True, apply np.log on the data output vector before the normalization.
            It can make the resulted distribution better
        '''
        self.batch_size = data_loader_config_dict[self.BATCH_SIZE_KEY]
        self.data_dir = data_loader_config_dict[self.DATA_DIR_KEY]
        self.train_perc = data_loader_config_dict[self.TRAIN_PERC_KEY]
        self.test_perc = 1 - self.train_perc
        assert self.test_perc > 0
        # print(self.train_perc)
        # print(self.test_perc)
        # exit()
        self.enable_log_predction = data_loader_config_dict[
            self.ENABLE_LOG_PREDICTION_KEY]
        self.enable_data_augment = data_loader_config_dict[
            self.ENABLE_DATA_AUGMENT_KEY]
        self.select_validation_set_inside = data_loader_config_dict[
            self.ENABLE_SELECT_VALIDATION_SET_INSIDE_KEY]

        self._init_vars()
        self._load_data(only_load_statistic_data)
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

    def _dump_statistics(self):
        stat_file = os.path.join(self.data_dir, self.STATISTIC_FILE_NAME)
        cont = {
            DataLoader.INPUT_MEAN_KEY: self.input_mean,
            DataLoader.INPUT_STD_KEY: self.input_std,
            DataLoader.OUTPUT_MEAN_KEY: self.output_mean,
            DataLoader.OUTPUT_STD_KEY: self.output_std
        }

        import pickle
        with open(stat_file, 'wb') as f:
            pickle.dump(cont, f)

    def _load_statistics(self):
        stat_file = os.path.join(self.data_dir, self.STATISTIC_FILE_NAME)
        if os.path.exists(stat_file) is True:
            import pickle
            with open(stat_file, 'rb') as f:
                cont = pickle.load(f)
                self.input_mean = cont[DataLoader.INPUT_MEAN_KEY]
                self.input_std = cont[DataLoader.INPUT_STD_KEY]
                self.output_mean = cont[DataLoader.OUTPUT_MEAN_KEY]
                self.output_std = cont[DataLoader.OUTPUT_STD_KEY]
            return True
        else:
            return False

    def _load_data(self, only_load_statistic_data_):
        '''
        load the data and compress to a pkl file
        '''
        load_stat_succ = False
        if only_load_statistic_data_ == True:
            # 1. begin to load statisic
            load_stat_succ = self._load_statistics()

            # 2. if succ, set X_lst and Y_lst, set flag = true

        # if we load statistic failed
        if (only_load_statistic_data_ == True
                and load_stat_succ == False) or (only_load_statistic_data_
                                                 == False):

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
                np.clip(self.input_std, 1e-2, None, self.input_std)
                np.clip(self.output_std, 1e-2, None, self.output_std)
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
            self._dump_statistics()

        if only_load_statistic_data_ == False:
            X_lst = (X_lst - self.input_mean) / self.input_std
            Y_lst = (Y_lst - self.output_mean) / self.output_std
            # print(f"raw Y = {Y_lst}")
            # print(f"exp Y = {np.exp(Y_lst)}")
            # print(f"output mean = {self.output_mean}")
            # print(f"output std = {self.output_std}")
            # exit(0)

            size = len(X_lst)
            train_size = int(self.train_perc * size)
            # test_size = size - train_size
            perm = np.random.permutation(size)

            train_id = None
            test_id = None
            if self.select_validation_set_inside is False:
                train_id = perm[:train_size]
                test_id = perm[train_size:]
            else:
                # 1. find the max and min value of each feature channel
                label_max = np.max(np.array(Y_lst), axis=0)
                label_min = np.min(np.array(Y_lst), axis=0)

                for i in range(len(label_max)):
                    if np.abs(label_max[i] - label_min[i]) < 1e-5:
                        # the min max is the same
                        label_min[i] -= 1e-5
                        label_max[i] += 1e-5
                print(f"label max {label_max}")
                print(f"label min {label_min}")
                # 2. iterate over the set: if one chanel meet the limit, put it inside the train_id, else put it inside the test_id
                test_size = size - train_size
                test_id = []
                train_id = []
                for idx in list(perm):
                    cur_Y = Y_lst[idx]
                    # if inside the dataset
                    if all(cur_Y > label_min) and all(cur_Y < label_max):
                        # if the test set is filled
                        if len(test_id) >= test_size:
                            train_id.append(idx)
                        else:

                            test_id.append(idx)
                    else:
                        train_id.append(idx)
                assert len(
                    test_id
                ) == test_size, f"ideal test size {test_size} real test_size {len(test_id)}"
                assert len(train_id) == train_size
            # print(f"--------- begin to check test id --------------")
            # for _idx, i in enumerate(test_id):
            #     print(f"test feature {_idx}: {self.output_mean + self.output_std * Y_lst[i]}")

            # exit()

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
            output_X, output_Y = self.test_X[st:st +
                                             incre], self.test_Y[st:st + incre]

            yield output_X, output_Y
            st += incre

    def apply_noise(self, inputs):
        assert type(inputs) is list
        assert type(inputs[0]) is np.ndarray
        assert len(inputs[0].shape) == 1

        size = inputs[0].shape[0]
        for _idx in range(len(inputs)):
            noise = (np.random.rand(3) - 0.5) / 10  # +-5cm
            noise_all = np.tile(noise, size // 3)
            inputs[_idx] += noise_all

    def get_train_data(self):
        st = 0
        while st < len(self.train_X):
            incre = min(st + self.batch_size, len(self.train_X)) - st
            if incre <= 0:
                break
            output_X, output_Y = self.train_X[st:st +
                                              incre], self.train_Y[st:st +
                                                                   incre]

            if self.enable_data_augment is True:
                assert len(output_X[0].shape) == 1
                size = output_X[0].shape[0]
                for _idx in range(len(output_X)):
                    noise = (np.random.rand(3) - 0.5) / 10  # +-5cm
                    noise_all = np.tile(noise, size // 3)
                    output_X[_idx] += noise_all

                # print(f"old norm {np.linalg.norm(np.array(output_X))}")
                # self.apply_noise(output_X)
                # print(f"new norm {np.linalg.norm(np.array(output_X))}")

            yield output_X, output_Y
            st += incre

    def __shuffle_train_data(self):
        train_size = len(self.train_X)
        perm = np.random.permutation(train_size)
        self.train_X = list(itemgetter(*perm)(self.train_X))
        self.train_Y = list(itemgetter(*perm)(self.train_Y))

    def shuffle(self):
        self.__shuffle_train_data()