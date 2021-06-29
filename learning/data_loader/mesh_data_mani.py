from operator import itemgetter
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
from multiprocessing import Pool
from itertools import repeat
import h5py
from .data_loader import CustomDataLoader
from abc import ABC


class MeshDataManipulator(ABC):
    BATCH_SIZE_KEY = "batch_size"  # all dataloader share
    DATA_DIR_KEY = "data_dir"  #
    TRAIN_PERC_KEY = "train_perc"
    ENABLE_DATA_AUGMENT_KEY = "enable_data_augment"
    BATCH_SIZE_KEY = "batch_size"
    ARCHIVE_NAME = "archive.hdf5"
    LOAD_ALL_DATA_INTO_MEM_KEY = "load_all_data_into_mem"
    INPUT_MEAN_KEY = "input_mean"
    INPUT_STD_KEY = "input_std"
    OUTPUT_MEAN_KEY = "output_mean"
    OUTPUT_STD_KEY = "output_std"

    def __init__(self, conf_dict):
        self.conf_dict = conf_dict
        self._parse_config(conf_dict)

        # if the archive doesn't exist
        if self._check_archive_exists() == False:
            train_files, test_files = self._load_mesh_data()
            stats = self._calc_statistics_distributed(train_files + test_files)
            self._save_archive(self.get_archive_path(), train_files,
                               test_files, stats)

        self._build_data_augmentation()
        # begin to init dataloader
        self._create_dataloader()

    def _build_data_augmentation(self):
        print(f"begin to do data augmentation")
        from .data_aug import apply_mesh_data_noise
        if self.enable_data_aug == False:
            self.data_aug = None
        else:
            self.data_aug = apply_mesh_data_noise

    def get_archive_path(self):
        return os.path.join(self.data_dir, MeshDataManipulator.ARCHIVE_NAME)

    def _check_archive_exists(self):
        return (os.path.exists(self.get_archive_path()) == True)

    # def __load_archive(self):
    #     f = h5py.File(self.get_archive_path(), mode='r')
    #     print(f.keys())
    #     print(f["train_set"].keys())
    #     exit()

    def get_dataloader(self):
        assert (self.train_dataloader is not None) and (self.val_dataloader
                                                        is not None)
        return self.train_dataloader, self.val_dataloader

    def _parse_config(self, config_dict):
        self.batch_size = config_dict[MeshDataManipulator.BATCH_SIZE_KEY]
        self.data_dir = config_dict[MeshDataManipulator.DATA_DIR_KEY]
        self.train_perc = config_dict[MeshDataManipulator.TRAIN_PERC_KEY]

        self.load_all_data_into_mem = config_dict[
            MeshDataManipulator.LOAD_ALL_DATA_INTO_MEM_KEY]
        self.enable_data_aug = config_dict[
            MeshDataManipulator.ENABLE_DATA_AUGMENT_KEY]
        self.train_dataloader = None
        self.val_dataloader = None
        pass

    def save_files_to_grp(res):
        group_name, idx, input, output, archive_path = res
        f = h5py.File(archive_path, 'a')
        grp = f[group_name]
        dst = grp.create_dataset(f"{idx}",
                                 shape=input.shape,
                                 data=input,
                                 dtype=np.float32)
        dst.attrs["label"] = output.astype(np.float32)
        f.close()
        # input, output = MeshDataManipulator.load_single_json_mesh_data(cur_file)
        # dst = grp.create_dataset(f"{idx}", shape=input.shape, data=input)
        # dst.attrs["label"] = output
        # pass

    def load_single_json_mesh_data_for_archive(_idx, group_name, filepath,
                                               archive_path, input_mean,
                                               input_std, output_mean,
                                               output_std):
        input, output = MeshDataManipulator.load_single_json_mesh_data(
            filepath)
        input = (input - input_mean) / input_std
        output = (output - output_mean) / output_std
        return (group_name, _idx, input, output, archive_path)

    def _save_archive(self, output_file, train_files, test_files, stats):
        assert type(stats) is dict
        f = h5py.File(output_file, 'w')
        # output train data and test data
        train_grp = f.create_group("train_set")
        test_grp = f.create_group("test_set")
        f.close()

        input_mean = stats[MeshDataManipulator.INPUT_MEAN_KEY]
        input_std = stats[MeshDataManipulator.INPUT_STD_KEY]
        output_mean = stats[MeshDataManipulator.OUTPUT_MEAN_KEY]
        output_std = stats[MeshDataManipulator.OUTPUT_STD_KEY]
        pool = Pool(4)
        for _idx, cur_file in enumerate(train_files):
            pool.apply_async(
                MeshDataManipulator.load_single_json_mesh_data_for_archive,
                (_idx, "train_set", cur_file, output_file, input_mean,
                 input_std, output_mean, output_std),
                callback=MeshDataManipulator.save_files_to_grp)
        for _idx, cur_file in enumerate(test_files):
            pool.apply_async(
                MeshDataManipulator.load_single_json_mesh_data_for_archive,
                (_idx, "test_set", cur_file, output_file, input_mean,
                 input_std, output_mean, output_std),
                callback=MeshDataManipulator.save_files_to_grp)
        pool.close()
        pool.join()
        # output statistics
        f = h5py.File(output_file, 'a')
        for i in list(stats.keys()):
            f.create_dataset(i, stats[i].shape, data=stats[i])
        print(f"[log] save archive succ")
        # print(f.keys())

        # print(list(stats.keys()))
        # iterave over all files and begin to write them down to the hdf5 file
        # exit()

    def _get_train_and_test_id(num, train_perc):
        num_train = int(num * train_perc)
        num_test = num - num_train
        assert num_train > 0 and num_test > 0, f"{num_train} {num_test}"
        perm = np.random.permutation(num)
        train_id = perm[:num_train]
        test_id = perm[num_train:]
        return train_id, test_id

    def __split_mesh_data(self, files):
        num = len(files)
        train_id, test_id = self._get_train_and_test_id(num, self.train_perc)
        train_files = list(itemgetter(*train_id)(files))
        test_files = list(itemgetter(*test_id)(files))
        return train_files, test_files

    def _load_mesh_data(self):
        # 1. the data dir is filled with the results
        print(f"begin to load mesh data from {self.data_dir}")
        filenames = []
        for cur_file in os.listdir(self.data_dir):
            if cur_file.find(".json") != -1:
                filenames.append(os.path.join(self.data_dir, cur_file))

        # 2. begin to split train set and test set
        train_files, test_files = self.__split_mesh_data(filenames)
        return train_files, test_files

    @staticmethod
    def load_single_json_mesh_data(cur_file):
        assert os.path.exists(cur_file) is True, f"{cur_file}"
        with open(cur_file) as f:
            cont = json.load(f)
        return np.array(cont["input"]), np.array(cont["output"])

    @staticmethod
    def calc_mean(files):
        input_mean = None
        output_mean = None

        for cur_file in files:
            # print(f"cur file {cur_file}")
            feature, output = MeshDataManipulator.load_single_json_mesh_data(
                cur_file)

            if input_mean is None:
                input_mean = feature
                output_mean = output
            else:
                input_mean += feature
                output_mean += output
        return input_mean, output_mean

    @staticmethod
    def calc_x_minus_xbar(files, input_mean, output_mean):
        assert type(files) == list
        assert type(input_mean) == np.ndarray
        assert type(output_mean) == np.ndarray
        input_sum = None
        output_sum = None
        for cur_file in files:
            input, output = MeshDataManipulator.load_single_json_mesh_data(
                cur_file)
            input_diff_squ = np.square(input - input_mean)
            output_diff_squ = np.square(output - output_mean)
            if input_sum is None:
                input_sum = input_diff_squ
                output_sum = output_diff_squ
            else:
                input_sum += input_diff_squ
                output_sum += output_diff_squ

        return input_sum, output_sum

    def _calc_statistics_distributed(self, all_files):
        num_of_files = len(all_files)
        num_of_thread = 6 if num_of_files > 6 else num_of_files

        # 1. calculate mean statistics
        params = [[] for i in range(num_of_thread)]
        for _idx, file in enumerate(all_files):
            params[_idx % num_of_thread].append(file)

        pool = Pool(num_of_thread)
        total_input_mean = None
        total_output_mean = None
        for subbatch_input_mean, subbatch_output_mean in pool.map(
                MeshDataManipulator.calc_mean, params):
            if total_input_mean is None:
                total_input_mean = subbatch_input_mean
                total_output_mean = subbatch_output_mean
            else:
                total_input_mean += subbatch_input_mean
                total_output_mean += subbatch_output_mean
        input_mean = total_input_mean / num_of_files
        output_mean = total_output_mean / num_of_files

        # 2. calculate std statistics
        # ret = pool.starmap(MeshDataManipulator.calc_x_minus_xbar, zip(params, repeat(mean)))

        input_std = None
        output_std = None
        for input_sum, output_sum in pool.starmap(
                MeshDataManipulator.calc_x_minus_xbar,
                zip(params, repeat(input_mean), repeat(output_mean))):
            if input_std is None:
                input_std = input_sum
                output_std = output_sum
            else:
                input_std += input_sum
                output_std += output_sum
        input_std = np.sqrt(input_std / (num_of_files))
        output_std = np.sqrt(output_std / (num_of_files))

        np.clip(input_std, 1e-2, None, input_std)
        np.clip(output_std, 1e-2, None, output_std)
        stats = {
            MeshDataManipulator.INPUT_MEAN_KEY: input_mean.astype(np.float32),
            MeshDataManipulator.INPUT_STD_KEY: input_std.astype(np.float32),
            MeshDataManipulator.OUTPUT_MEAN_KEY:
            output_mean.astype(np.float32),
            MeshDataManipulator.OUTPUT_STD_KEY: output_std.astype(np.float32)
        }
        return stats

    def __create_dataset(self):
        from .data_loader import CustomDataset

        f = h5py.File(self.get_archive_path(), mode='r')
        f[MeshDataManipulator.INPUT_MEAN_KEY],
        f[MeshDataManipulator.INPUT_STD_KEY],
        f[MeshDataManipulator.OUTPUT_MEAN_KEY],
        f[MeshDataManipulator.OUTPUT_STD_KEY]

        train_dataset = CustomDataset(f["train_set"],
                                      f[MeshDataManipulator.INPUT_MEAN_KEY],
                                      f[MeshDataManipulator.INPUT_STD_KEY],
                                      f[MeshDataManipulator.OUTPUT_MEAN_KEY],
                                      f[MeshDataManipulator.OUTPUT_STD_KEY],
                                      self.load_all_data_into_mem)
        test_dataset = CustomDataset(f["test_set"],
                                     f[MeshDataManipulator.INPUT_MEAN_KEY],
                                     f[MeshDataManipulator.INPUT_STD_KEY],
                                     f[MeshDataManipulator.OUTPUT_MEAN_KEY],
                                     f[MeshDataManipulator.OUTPUT_STD_KEY],
                                     self.load_all_data_into_mem)

        return train_dataset, test_dataset

    def _create_dataloader(self):
        train_dataset, test_dataset = self.__create_dataset()
        self.train_dataloader = CustomDataLoader(train_dataset,
                                                 self.batch_size,
                                                 self.data_aug)
        self.val_dataloader = CustomDataLoader(test_dataset, self.batch_size)
        print(f"create dataloader succ")

    # def __calc_statistics(self, all_files):
    #     input_lst, output_lst = [], []
    #     for file in all_files:
    #         input, output = MeshDataManipulator.load_single_json_mesh_data(file)
    #         input_lst.append(input)
    #         output_lst.append(output)
    #     input_lst = np.array(input_lst)
    #     output_lst = np.array(output_lst)
    #     input_mean = np.mean(input_lst, axis=0)
    #     input_std = np.std(input_lst, axis=0)
    #     output_mean = np.mean(output_lst, axis=0)
    #     output_std = np.std(output_lst, axis=0)

    #     return input_mean, input_std, output_mean, output_std
