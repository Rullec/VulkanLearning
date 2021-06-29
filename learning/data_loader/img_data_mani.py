from operator import itemgetter

from tqdm import utils
from .mesh_data_mani import MeshDataManipulator
import os
from multiprocessing import Pool


class ImageDataManipulator(MeshDataManipulator):
    def __init__(self, conf_dict):
        super(ImageDataManipulator, self).__init__(conf_dict)

    def __split_img_dirs(self, all_img_dirs):
        num = len(all_img_dirs)
        train_id, test_id = self._get_train_and_test_id(num, self.train_perc)
        train_dirs = list(itemgetter(*train_id)(all_img_dirs))
        test_dirs = list(itemgetter(*test_id)(all_img_dirs))

        return train_dirs, test_dirs

    def _load_image_dir(dir_path):
        assert os.path.exists(dir_path) == True

    def _load_mesh_data(self):
        print(f"begin to load depth data from {self.data_dir}")
        # 1. find all images dirs
        all_img_dirs = []
        for cur_obj in os.listdir(self.data_dir):
            if os.path.isdir(cur_obj) == True:
                all_img_dirs.append(cur_obj)
        # 2. get all dirs
        train_dirs, test_dirs = self.__split_img_dirs(all_img_dirs)
        return train_dirs, test_dirs

    def calc_mean(dir_path):

        return

    def _calc_statistics_distributed_perpixel(self, all_dirs):
        # calculate the mean for all images
        num_of_dirs = len(all_dirs)
        num_of_threads = 6
        params = [[] for i in range(num_of_threads)]

        for _idx, dir in enumerate(all_dirs):
            params[_idx % num_of_threads].append(dir)

        pool = Pool(num_of_threads)
        total_input_mean = None
        total_output_mean = None
        for subbatch_input_mean, subbatch_output_mean in pool.map(
                ImageDataManipulator.calc_mean, params):
            if total_input_mean is None:
                total_input_mean = subbatch_input_mean
                total_output_mean = subbatch_output_mean
            else:
                total_input_mean += subbatch_input_mean
                total_output_mean = subbatch_output_mean

        input_mean = total_input_mean / num_of_dirs
        output_mean = total_output_mean / num_of_dirs
        print(f"input mean {input_mean.shape}")
        print(f"output mean {output_mean.shape}")
        exit()
        # calculate the std for all images

        return

    def _calc_statistics_distributed(self, all_dirs):
        return self._calc_statistics_distributed_perpixel(all_dirs)