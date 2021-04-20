import numpy as np
import os
import json


class DataLoader():
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.init_vars()

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size

    def init_vars(self):
        assert os.path.exists(self.data_dir)
        self.num_of_data = len(os.listdir(self.data_dir))
        tmp_name = os.listdir(self.data_dir)[0]
        X, Y = DataLoader.__load_single_data(
            os.path.join(self.data_dir, tmp_name))
        self.input_size = len(X)
        self.output_size = len(Y)

    @staticmethod
    def __load_single_data(file_path):
        assert os.path.exists(file_path), f"{file_path}"

        with open(file_path) as f:
            cont = json.load(f)

            X = np.array(cont["input"])
            Y = np.array(cont["output"])

            return X, Y

    def load_data(self):
        X_lst, Y_lst = [], []
        if os.path.exists(self.data_dir) == True:
            for f in os.listdir(self.data_dir):
                tar_f = os.path.join(self.data_dir, f)
                X, Y = DataLoader.__load_single_data(tar_f)
                X_lst.append(X)
                Y_lst.append(Y)
        X_lst = np.array(X_lst)
        Y_lst = np.array(Y_lst)
        print(f"X lst shape {X_lst.shape}")
        print(f"Y lst shape {Y_lst.shape}")
        assert len(X_lst.shape) == 2
        assert len(Y_lst.shape) == 2
        assert X_lst.shape[0] == Y_lst.shape[0]
        return X_lst, Y_lst
