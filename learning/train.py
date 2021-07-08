import numpy as np
import torch
import json
from agents.agent_builder import build_net


def init_env():
    np.set_printoptions(suppress=True)
    # from tqdm import tqdm
    # lst = list(np.random.rand(100))
    # for id, _ in enumerate(tqdm(lst)):
    #     print(id)

    # exit()
    np.random.seed(0)
    torch.manual_seed(0)
    is_cuda_avaliable = torch.cuda.is_available()
    assert is_cuda_avaliable == True

    if is_cuda_avaliable == True:
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu", 0)
    return device


if __name__ == "__main__":

    # net = ParamNet(conf_path, device)
    device = init_env()

    # conf_path = "../config/train_configs/conv_conf.json"
    conf_path = "../config/train_configs/fc_conf.json"

    with open(conf_path) as f:
        mode = json.load(f)["mode"]
    net_type = build_net(conf_path)
    net = net_type(conf_path, device)

    if mode == "test":
        net.test()
    elif mode == "train":
        net.train(max_epochs=10000)
    else:
        raise ValueError(mode)