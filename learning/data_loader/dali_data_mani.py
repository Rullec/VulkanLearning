from .img_data_mani import ImageDataManipulator
# from img_data_mani import ImageDataManipulator

# only import dali on Linux platform
try:
    import platform
    if platform.system() != "Linux":
        raise ImportError
    from .dali_utils.dali_torch_wrapper import build_dali_torch_wrapper
    # from dali_utils.dali_torch_wrapper import build_dali_torch_wrapper
except ImportError as e:
    pass
from .dali_utils.dali_file_reader import DALIHdf5Reader
# from dali_utils.dali_file_reader import DALIHdf5Reader
import h5py


class DALIDataManipulator(ImageDataManipulator):
    '''
    '''
    def __init__(self, conf_dict):
        print(f"init DALIDataManipulator")
        self.conf_dict = conf_dict
        self._parse_config(conf_dict)
        if self._check_archive_exists() == False or self._validate_archive(
        ) == False:
            train_dirs, test_dirs = self._load_mesh_data()
            self.stats = self._calc_statistics_distributed(train_dirs +
                                                           test_dirs)

            self._save_archive(self.get_default_archive_path(), train_dirs,
                               test_dirs, self.stats)

        self._create_dataloader()

    def _parse_config(self, config_dict):
        super()._parse_config(config_dict)

    def _create_dataloader(self):
        # open all group handles
        train_grp_lst, test_grp_lst = [], []
        all_archives = self.get_all_hdf5_archive()
        for arc in all_archives:
            f = h5py.File(arc, 'r')
            train_grp_lst.append(f["train_set"])
            test_grp_lst.append(f["test_set"])

        input_mean, input_std, output_mean, output_std = self._load_statistics_from_archive()
        train_reader = DALIHdf5Reader(self.batch_size, train_grp_lst,
                                      self.conf_dict["load_all_data_into_mem"],
                                      input_mean, input_std, output_mean,
                                      output_std)
        val_reader = DALIHdf5Reader(self.batch_size, test_grp_lst,
                                    self.conf_dict["load_all_data_into_mem"],
                                    input_mean, input_std, output_mean,
                                    output_std)
        self.train_dataloader = build_dali_torch_wrapper(
            file_reader=train_reader, batch_size=self.batch_size)
        self.val_dataloader = build_dali_torch_wrapper(
            file_reader=val_reader, batch_size=self.batch_size)
        # print(f"please check the train pipe: it must be augmented")
        # exit()
        # cur_epoch = 0
        # while cur_epoch < 3:
        #     print(f"begin to test train dataloader")
        #     for _idx, batched in enumerate(self.train_dataloader):
        #         print(f"epoch {cur_epoch} idx {_idx}")
        #         input, output = batched
        #         print(f"epoch {cur_epoch} input shape {input.shape}")
        #         print(f"epoch {cur_epoch} output shape {output.shape}")
        #     print(f"done to test train dataloader")
        #     cur_epoch += 1
        # exit()
        # cur_epoch = 0
        # while cur_epoch < 3:
        #     print(f"begin to test val dataloader")
        #     for _idx, batched in enumerate(self.val_dataloader):
        #         print(f"epoch {cur_epoch} idx {_idx}")
        #         input, output = batched
        #         print(f"epoch {cur_epoch} input shape {input.shape}")
        #         print(f"epoch {cur_epoch} output shape {output.shape}")
        #     print(f"done to test val dataloader")
        #     cur_epoch += 1

        # train_dataiter = HDF5InputIterator(self.batch_size, self.data_dir,
        #                                    "train_set")
        # test_dataiter = HDF5InputIterator(self.batch_size, self.data_dir,
        #                                   "test_set")

        # input_mean, input_std, output_mean, output_std = self._get_statistics()
        # self.train_dataloader = build_variables(train_dataiter,
        #                                         self.batch_size, input_mean,
        #                                         input_std, output_mean,
        #                                         output_std)
        # self.val_dataloader = build_variables(test_dataiter, self.batch_size,
        #                                       input_mean, input_std,
        #                                       output_mean, output_std)
        print(f"create dataloader succ")


if __name__ == "__main__":
    import time
    print("begin to test the dataloader")
    conf_dict = {
        "batch_size": 64,
        "enable_log_prediction": False,
        "data_dir":
        "../../data/export_data/uniform_sample10_noised2_4camnoised_2rot_4view",
        "train_perc": 0.8,
        "enable_data_augment": True,
        "load_all_data_into_mem": False,
        "enable_test": False,
        "input_normalize_mode": "per_pixel"
    }

    mani = DALIDataManipulator(conf_dict)
    train_loader, val_loader = mani.get_dataloader()
    st = time.time()

    for _idx, batched in enumerate(train_loader):
        print(_idx)
    ed = time.time()
    print(f"cost {ed - st}")