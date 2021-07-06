from operator import itemgetter
import h5py
from PIL.Image import Image
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("../calibration")
from file_util import get_subdirs, load_json, load_png_image
from .mesh_data_mani import MeshDataManipulator
from itertools import repeat
import os
from multiprocessing import Pool, Value

from enum import Enum


class NORMALIZE_MODE(Enum):
    PER_PIXEL = 0
    ALL_FLOAT = 1

    @staticmethod
    def build_mode_from_str(mode_str):
        if mode_str == "per_pixel":
            return NORMALIZE_MODE.PER_PIXEL
        elif mode_str == "all_float":
            return NORMALIZE_MODE.ALL_FLOAT
        else:
            raise Value(mode_str)

    @staticmethod
    def build_str_from_mode(mode):
        if mode == NORMALIZE_MODE.PER_PIXEL:
            return "per_pixel"
        elif mode == NORMALIZE_MODE.ALL_FLOAT:
            return "all_float"
        else:
            raise Value(mode)


class ImageDataManipulator(MeshDataManipulator):
    FEATURE_JSON_NAME = "feature.json"
    INPUT_NORMALZE_MODE_KEY = "input_normalize_mode"
    NORMALIZE_MODE_HDF5_KEY = "normalize_mode"

    def __init__(self, conf_dict):
        self.conf_dict = conf_dict
        self._parse_config(conf_dict)

        if self.enable_test == True:
            self._test()

        # if the archive doesn't exist
        if self._check_archive_exists() == False or self._validate_archive(
        ) == False:
            train_dirs, test_dirs = self._load_mesh_data()
            stats = self._calc_statistics_distributed(train_dirs + test_dirs)

            self._save_archive(self.get_archive_path(), train_dirs, test_dirs,
                               stats)

        self._build_data_augmentation()
        # begin to init dataloader
        self._create_dataloader()

    def _parse_config(self, conf_dict):
        super()._parse_config(conf_dict)
        self.normalize_mode = NORMALIZE_MODE.build_mode_from_str(
            conf_dict[ImageDataManipulator.INPUT_NORMALZE_MODE_KEY])

    def _validate_archive(self):
        valid = super()._validate_archive()
        f = h5py.File(self.get_archive_path(), 'r')
        all_keys = f.keys()
        valid = (ImageDataManipulator.NORMALIZE_MODE_HDF5_KEY
                 in f.attrs) and valid

        if valid is True:
            hdf5_mode_str = f.attrs[
                ImageDataManipulator.NORMALIZE_MODE_HDF5_KEY]
            cur_str = NORMALIZE_MODE.build_str_from_mode(self.normalize_mode)
            valid = (hdf5_mode_str == cur_str)

        return valid

    def _build_data_augmentation(self):
        if self.enable_data_aug is True:
            from .data_aug import apply_depth_albumentation, apply_depth_aug
            self.data_aug = apply_depth_aug
            print("[log] do torch aug")
            
            # self.data_aug = apply_depth_albumentation
            # print("[log] do albumentation aug")
        else:
            self.data_aug = None

    def _load_mesh_data(self):
        print(f"begin to load depth data from {self.data_dir}")
        # 1. find all images dirs
        all_img_dirs = []
        for cur_obj in os.listdir(self.data_dir):
            full = os.path.join(self.data_dir, cur_obj)
            if os.path.isdir(full) == True:
                all_img_dirs.append(full)
        # 2. get all dirs
        train_dirs, test_dirs = self._split_mesh_data(all_img_dirs)
        return train_dirs, test_dirs

    def _get_directory_png_dirs_and_feature_file(cur_dir):
        assert os.path.exists(cur_dir) == True, f"{cur_dir}"
        feature_filename = os.path.join(cur_dir,
                                        ImageDataManipulator.FEATURE_JSON_NAME)
        assert os.path.exists(feature_filename) == True, f"{feature_filename}"

        data_dir_lst = []
        for init_rot_dir in get_subdirs(cur_dir):
            cur_dir1 = os.path.join(cur_dir, init_rot_dir)
            for cam_dir in get_subdirs(cur_dir1):
                data_dir_lst.append(os.path.join(cur_dir1, cam_dir))
        return data_dir_lst, feature_filename

    def _load_many_png(dir_path):
        files = os.listdir(dir_path)
        png_lst = []
        for i in files:
            png_lst.append(load_png_image(os.path.join(dir_path, i)))
        png_lst = np.stack(png_lst, axis=0)
        assert len(png_lst.shape) == 3
        return png_lst

    def _load_image_parent_dir(cur_dir):
        # 1. judge the feature.json
        assert os.path.exists(cur_dir) == True, f"{cur_dir}"
        feature_filename = os.path.join(cur_dir,
                                        ImageDataManipulator.FEATURE_JSON_NAME)
        assert os.path.exists(feature_filename) == True
        from file_util import load_json
        label = np.array(load_json(feature_filename)["feature"])

        input_lst = []
        # 2. find results per mesh
        for init_rot_dir in get_subdirs(cur_dir):
            cur_dir1 = os.path.join(cur_dir, init_rot_dir)
            for cam_dir in get_subdirs(cur_dir1):
                cur_dir2 = os.path.join(cur_dir1, cam_dir)
                input = ImageDataManipulator._load_many_png(cur_dir2)
                input_lst.append(input)
        input_array = np.array(input_lst)
        # 3. return feature vector & data list
        return input_array, label

    def _calc_x_minus_xbar_perpixel(dir_path_lst, input_mean, output_mean):
        total_num = 0
        input_sum = None
        output_sum = None
        # for cur_dir in tqdm(dir_path_lst):
        for cur_dir in tqdm(dir_path_lst):
            input_array, output = ImageDataManipulator._load_image_parent_dir(
                cur_dir)
            assert len(input_array.shape) == 4
            num = input_array.shape[0]
            total_num += num

            # handle output label squared diff sum
            for i in range(num):
                assert input_array[i].shape == input_mean.shape
                input_diff_squ = np.square(input_array[i] - input_mean)
                output_diff_squ = np.square(output - output_mean)
                # print(f"dir {cur_dir} id {i} input diff squ sum {np.sum(input_diff_squ)}")
                # input_sum_lst.append(input_diff_squ)
                # output_sum_lst.append(output_diff_squ)
                if input_sum is None:
                    input_sum = input_diff_squ
                    output_sum = output_diff_squ
                else:
                    input_sum += input_diff_squ
                    output_sum += output_diff_squ

        return total_num, total_num, input_sum, output_sum

    def _calc_sum_per_pixel(dir_path_lst):
        # 1. given a batch of dir, read all image and calculate the sum
        input_sum = None
        output_sum = None
        total_num = 0
        for cur_dir in tqdm(dir_path_lst):
            input_array, output = ImageDataManipulator._load_image_parent_dir(
                cur_dir)
            assert len(input_array.shape) == 4, f"{input_array.shape}"
            cur_num = input_array.shape[0]
            total_num += cur_num
            if input_sum is None:
                input_sum = np.sum(input_array, axis=0)
                output_sum = output * cur_num
            else:
                input_sum += np.sum(input_array, axis=0)
                output_sum += output * cur_num

        return total_num, total_num, input_sum, output_sum

    def _calc_sum_allfloat(dir_path_lst):
        input_sum = None
        output_sum = None
        num_input_total = 0
        num_output_total = 0
        for cur_dir in tqdm(dir_path_lst):
            input_array, output = ImageDataManipulator._load_image_parent_dir(
                cur_dir)
            assert len(input_array.shape) == 4, f"{input_array.shape}"
            num_input_total += input_array.size
            num_output_total += output.size
            if input_sum is None:
                input_sum = np.sum(input_array)
                output_sum = np.sum(output)
            else:
                input_sum += np.sum(input_array)
                output_sum += np.sum(output)

        return num_input_total, num_output_total, input_sum, output_sum

    def _calc_x_minus_xbar_allfloat(dir_path_lst, input_mean, output_mean):

        input_sum = 0.0
        output_sum = 0.0
        num_input_total = 0
        num_output_total = 0
        # for cur_dir in tqdm(dir_path_lst):
        for cur_dir in tqdm(dir_path_lst):
            input_array, output = ImageDataManipulator._load_image_parent_dir(
                cur_dir)
            assert len(input_array.shape) == 4
            num_input_total += input_array.size
            num_output_total += output.size
            input_sum += np.sum(np.square(input_array - input_mean))
            output_sum += np.sum(np.square(output - output_mean))
        return num_input_total, num_output_total, input_sum, output_sum

    def _calc_statistics_distributed_perpixel(all_dirs):
        stats = MeshDataManipulator._calc_statistics_distributed(
            all_dirs, ImageDataManipulator._calc_sum_per_pixel,
            ImageDataManipulator._calc_x_minus_xbar_perpixel)
        return stats

    def _calc_statistics_distributed_allfloat(all_dirs):
        stats = MeshDataManipulator._calc_statistics_distributed(
            all_dirs, ImageDataManipulator._calc_sum_allfloat,
            ImageDataManipulator._calc_x_minus_xbar_allfloat)
        return stats

    def _calc_statistics_distributed(self, all_dirs):
        if self.normalize_mode == NORMALIZE_MODE.PER_PIXEL:
            stats = ImageDataManipulator._calc_statistics_distributed_perpixel(
                all_dirs)
        elif self.normalize_mode == NORMALIZE_MODE.ALL_FLOAT:
            stats = ImageDataManipulator._calc_statistics_distributed_allfloat(
                all_dirs)
        else:
            raise ValueError(f"{self.normalize_mode} unsupported")
        input_mean, input_std, output_mean, output_std = MeshDataManipulator._unpack_statistics(
            stats)
        input_std, output_std = ImageDataManipulator._std_clip(
            input_std, output_std)
        stats = MeshDataManipulator._pack_statistics(
            input_mean.astype(np.float32), input_std.astype(np.float32),
            output_mean.astype(np.float32), output_std.astype(np.float32))

        return stats

    def _calc_statistics_allmem_perpixel(all_dirs):
        input_lst = []
        output_lst = []
        # for cur_dir in all_dirs:
        for cur_dir in tqdm(all_dirs):
            input, output = ImageDataManipulator._load_image_parent_dir(
                cur_dir)
            input_lst.append(input)
            output_lst.append(output)

        input_lst = np.vstack(input_lst)
        output_lst = np.vstack(output_lst)
        assert len(input_lst.shape) == 4, f"{input_lst.shape}"
        assert len(output_lst.shape) == 2
        input_mean = np.mean(input_lst, axis=0)
        input_std = np.std(input_lst, axis=0)
        output_mean = np.mean(output_lst, axis=0)
        output_std = np.std(output_lst, axis=0)
        input_std, output_std = MeshDataManipulator._std_clip(
            input_std, output_std)
        return MeshDataManipulator._pack_statistics(input_mean, input_std,
                                                    output_mean, output_std)

    def _calc_statistics_allmem_allfloat(all_dirs):
        input_lst = []
        output_lst = []
        # for cur_dir in all_dirs:
        for cur_dir in tqdm(all_dirs):
            input, output = ImageDataManipulator._load_image_parent_dir(
                cur_dir)
            input_lst.append(input)
            output_lst.append(output)

        input_lst = np.vstack(input_lst)
        output_lst = np.vstack(output_lst)
        input_mean = np.mean(input_lst)
        input_std = np.std(input_lst)
        output_mean = np.mean(output_lst)
        output_std = np.std(output_lst)
        input_std, output_std = MeshDataManipulator._std_clip(
            input_std, output_std)
        return MeshDataManipulator._pack_statistics(input_mean, input_std,
                                                    output_mean, output_std)

    def load_datadir_and_feature_file_for_archive(_idx, group_name, data_dir,
                                                  feature_file, archive_path,
                                                  input_mean, input_std,
                                                  output_mean, output_std):
        # noth that now the "data_dir" should be the very low level dir
        input = ImageDataManipulator._load_many_png(data_dir).astype(
            np.float32)
        output = np.array(load_json(feature_file)["feature"], dtype=np.float32)

        input = (input - input_mean) / input_std
        output = (output - output_mean) / output_std

        return (group_name, _idx, input, output, archive_path)

    def _save_archive(self, output_file, train_parent_dirs, test_parent_dirs,
                      stats):
        print(f"begin to save depth archive to {output_file}...")
        assert type(stats) is dict
        MeshDataManipulator._create_hdf5_archive_empty_file(output_file)
        input_mean, input_std, output_mean, output_std = MeshDataManipulator._unpack_statistics(
            stats)

        # 1. given a list of directory, expand them to the file list
        train_data_dirs = [
        ]  # filled with tuples (datapoint dir, feature file)
        for p in train_parent_dirs:
            data_dirs, feature_file = ImageDataManipulator._get_directory_png_dirs_and_feature_file(
                p)
            for i in range(len(data_dirs)):
                train_data_dirs.append((data_dirs[i], feature_file))

        test_data_dirs = []  # filled with tuples (datapoint dir, feature file)
        for p in test_parent_dirs:
            data_dirs, feature_file = ImageDataManipulator._get_directory_png_dirs_and_feature_file(
                p)
            for i in range(len(data_dirs)):
                test_data_dirs.append((data_dirs[i], feature_file))

        import platform
        if platform.system() == "Linux":
            pool = Pool(12)
        elif platform.system() == "Windows":
            pool = Pool(4)
        else:
            raise ValueError("unsupported platform {platform.system()}")
            
        print("begin to save train set...")
        for _idx, value in enumerate(train_data_dirs):
            # MeshDataManipulator.save_files_to_grp(
            #     ImageDataManipulator.load_datadir_and_feature_file_for_archive(
            #         _idx, "train_set", data_dir, feature_file, output_file,
            #         input_mean, input_std, output_mean, output_std))
            data_dir, feature_file = value
            pool.apply_async(
                ImageDataManipulator.load_datadir_and_feature_file_for_archive,
                (_idx, "train_set", data_dir, feature_file, output_file,
                 input_mean, input_std, output_mean, output_std),
                callback=MeshDataManipulator.save_files_to_grp)

        for _idx, value in enumerate(test_data_dirs):
            data_dir, feature_file = value
            pool.apply_async(
                ImageDataManipulator.load_datadir_and_feature_file_for_archive,
                (_idx, "test_set", data_dir, feature_file, output_file,
                 input_mean, input_std, output_mean, output_std),
                callback=MeshDataManipulator.save_files_to_grp)
        pool.close()
        pool.join()
        f = h5py.File(output_file, 'a')
        for i in list(stats.keys()):
            f.create_dataset(i, stats[i].shape, data=stats[i])

        f.attrs["normalize_mode"] = NORMALIZE_MODE.build_str_from_mode(
            self.normalize_mode)
        print(f"[log] save archive succ")

    def _test(self):
        if self.normalize_mode == NORMALIZE_MODE.PER_PIXEL:
            self._test_perpixel()
        elif self.normalize_mode == NORMALIZE_MODE.ALL_FLOAT:
            self._test_allfloat()
        exit()

    def _test_perpixel(self):
        self._remove_archive()
        train_dirs, test_dirs = self._load_mesh_data()
        dist_stats = ImageDataManipulator._calc_statistics_distributed_perpixel(
            train_dirs + test_dirs)
        print(f"_calc_statistics_distributed perpixel done")
        mem_stats = ImageDataManipulator._calc_statistics_allmem_perpixel(
            train_dirs + test_dirs)

        print(f"_calc_statistics_allmem perpixel ldone")
        # begin to compare
        for key in dist_stats.keys():
            diff = np.abs(dist_stats[key] - mem_stats[key])
            max_diff = np.max(diff)
            if max_diff > 1e-2:
                raise ValueError(f"{key} test failed, max diff {max_diff}")

            else:
                print(f"test {key} succ, max diff {max_diff}")

    def _test_allfloat(self):
        self._remove_archive()
        train_dirs, test_dirs = self._load_mesh_data()
        dist_stats = ImageDataManipulator._calc_statistics_distributed_allfloat(
            train_dirs + test_dirs)
        print(f"_calc_statistics_distributed allfloat done")
        mem_stats = ImageDataManipulator._calc_statistics_allmem_allfloat(
            train_dirs + test_dirs)

        print(f"_calc_statistics_allmem allfloat ldone")
        # begin to compare
        for key in dist_stats.keys():
            diff = np.abs(dist_stats[key] - mem_stats[key])
            max_diff = np.max(diff)
            if max_diff > 1e-2:
                raise ValueError(f"{key} test failed, max diff {max_diff}")

            else:
                print(f"test {key} succ, max diff {max_diff}")
        print(dist_stats)
        pass