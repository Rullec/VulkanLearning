import numpy as np
import sys

sys.path.append("../calibration")

from train import init_env, build_net
from image_data_loader import ImageDataLoader
from drawer_util import DynaPlotter


def load_test_data():
    dataloader_config = {
        "batch_size": 64,
        "enable_log_prediction": False,
        "data_dir":
        "D:\\SimpleClothSimulator\\data\\export_data\\isotropic_200prop_16samples_amp_0.05_gen_1view",
        "train_perc": 0.8,
        "enable_data_augment": True,
        "enable_select_validation_set_inside": True,
        "enable_split_same_rot_image_into_one_set": True
    }
    dataloader_config[
        "data_dir"] = r"D:\SimpleClothSimulator\data\export_data\sample_mesh_data_gen"

    # print(f"good dir")
    new_data_loader = ImageDataLoader(dataloader_config,
                                      only_load_statistic_data=False)
    x, y = next(new_data_loader.get_all_data())
    x = new_data_loader.unnormalize_input_data(x)
    y = new_data_loader.unnormalize_output_data(y)
    return x, y
    # print(len(x))
    # print(len(y))


if __name__ == "__main__":
    real_x, real_y = load_test_data()
    print(f"x shape {real_x.shape}")
    # print(f"pred shape {pred.shape}")
    # exit()
    device = init_env()
    conf_path = "..\config\\train_configs\\conv_conf.json"
    net_type = build_net(conf_path)
    net = net_type(conf_path, device, only_load_statistic_data=False)
    dataloader = net.data_loader
    train_X, train_Y = next(dataloader.get_validation_data())
    train_X = dataloader.unnormalize_input_data(train_X)
    train_Y = dataloader.unnormalize_output_data(train_Y)
    train_pred = net.infer(train_X)
    print(f"train data pred {train_pred}")

    real_pred = net.infer(real_x)
    print(f"real data pred {real_pred}")
    print(f"real data label {real_y}")
    plot = DynaPlotter(2, 3, iterative_mode=False)
    plot.add(train_X[0][0], f"a train data sample")
    for i in range(real_pred.shape[0]):
        plot.add(
            real_x[i][0],
            f"real_data{i}/pred={np.mean(real_pred[i]) : 2.1f}/label={np.mean(real_y[i]) : 2.1f}"
        )

    plot.show()