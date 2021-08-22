from torch.utils.data import Dataset
import pickle
import json
import os
import numpy as np
import cv2


class ToyDataset(Dataset):
    def __init__(self, data_dir, phase_train=True, transform=None):
        assert data_dir is not None
        self.phase_train = phase_train
        # 1. load train.txt
        self.transform = transform
        self.root_dir = data_dir
        self.train_file = os.path.join(data_dir, "train.txt")
        self.train_depth_lst = []
        self.train_label_lst = []

        self.test_file = os.path.join(data_dir, "test.txt")
        self.test_depth_lst = []
        self.test_label_lst = []

        with open(self.train_file, 'r') as f:
            for line in f.readlines():
                line = line.strip().split()
                self.train_depth_lst.append(
                    os.path.join(self.root_dir, line[0]))
                self.train_label_lst.append(
                    os.path.join(self.root_dir, line[1]))
        with open(self.test_file, 'r') as f:
            for line in f.readlines():
                line = line.strip().split()
                self.test_depth_lst.append(os.path.join(
                    self.root_dir, line[0]))
                self.test_label_lst.append(os.path.join(
                    self.root_dir, line[1]))

        self.stats_file = os.path.join(data_dir, "stats.pkl")
        with open(self.stats_file, 'rb') as f:
            cont = pickle.load(f)
            self.input_mean = cont['depth_mean']
            self.input_std = cont['depth_std']
            self.output_mean = cont['label_mean']
            self.output_std = cont['label_std']

    def __getitem__(self, index):
        if self.phase_train:
            depth = np.array([
                cv2.imread(
                    os.path.join(self.train_depth_lst[index], f"{i}.png"),
                    cv2.IMREAD_GRAYSCALE) for i in range(4)
            ],
                             dtype=np.float32)
            with open(self.train_label_lst[index], 'r') as f:
                label = np.array(json.load(f)["feature"], dtype=np.float32)

            # albu aug:
            if self.transform is not None:
                depth = self.transform(image=depth)
                if type(depth) == dict:
                    depth = depth['image']
            return depth, label
        else:
            depth = np.array([
                cv2.imread(
                    os.path.join(self.test_depth_lst[index], f"{i}.png"),
                    cv2.IMREAD_GRAYSCALE) for i in range(4)
            ],
                             dtype=np.float32)
            with open(self.test_label_lst[index], 'r') as f:
                label = np.array(json.load(f)["feature"], dtype=np.float32)
            return depth, label

    def __len__(self):
        if self.phase_train:
            return len(self.train_depth_lst)
        else:
            return len(self.test_depth_lst)