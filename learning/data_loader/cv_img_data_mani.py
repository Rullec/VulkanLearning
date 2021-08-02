from .img_data_mani import HDF5ImageDataManipulator
from .mesh_data_mani import MeshDataManipulator
from .data_loader_torch import OpencvDataset
import h5py
from .data_loader_torch import CustomDataLoader
class OpencvImageDataManipulator(HDF5ImageDataManipulator):
    def __init__(self, conf_dict):
        self.conf_dict = conf_dict
        self._parse_config(conf_dict)
        if self._check_archive_exists() == False:
            # begin to save archive
            # only the statistics: input & output / mean & std
            assert False
        else:
            ultimate_f = h5py.File(self.get_default_archive_path(), mode='r')
            self.input_mean = ultimate_f[MeshDataManipulator.INPUT_MEAN_KEY][...]
            self.input_std = ultimate_f[MeshDataManipulator.INPUT_STD_KEY][...]
            self.output_mean = ultimate_f[MeshDataManipulator.OUTPUT_MEAN_KEY][...]
            self.output_std = ultimate_f[MeshDataManipulator.OUTPUT_STD_KEY][...]
            # read the statistics

        # build the data augmentation
        self._build_data_augmentation()

        # create the dataloader
        train_dirs, test_dirs = self._load_mesh_data()
        
        train_dst = OpencvDataset(train_dirs, self.input_mean ,
            self.input_std ,
            self.output_mean, 
            self.output_std, self.data_aug)
        test_dst = OpencvDataset(test_dirs, self.input_mean ,
            self.input_std ,
            self.output_mean, 
            self.output_std, data_aug= None)
        # print(self.data_aug)
        # exit()
        self.train_dataloader = CustomDataLoader(train_dst,
                                                 self.batch_size)
        self.val_dataloader = CustomDataLoader(test_dst, self.batch_size)

    def _parse_config(self,conf_dict):
        super()._parse_config(conf_dict)

