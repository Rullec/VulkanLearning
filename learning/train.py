import numpy as np
import torch
from param_net import ParamNet
import json
from res_param_net import CNNParamNet


def build_net(conf):
    with open(conf) as f:
        cont = json.load(f)
    assert ParamNet.NAME_KEY in cont, f"no {ParamNet.NAME_KEY} key in given config {conf}"
    param_name = cont[ParamNet.NAME_KEY]

    if param_name == ParamNet.NAME:
        print(f"[log] build {param_name} net succ")
        return ParamNet
    elif param_name == CNNParamNet.NAME:
        print(f"[log] build {param_name} net succ")
        return CNNParamNet
    else:
        raise ValueError(f"unsupport net type {param_name}")
    # print(param_name)


if __name__ == "__main__":
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
    # net = ParamNet(conf_path, device)

    conf_path = "..\config\\train_configs\\conv_conf.json"
    # conf_path = "..\config\\train_configs\\fc_conf.json"
    net_type = build_net(conf_path)
    net = net_type(conf_path, device, only_load_statistic_data=False)
    net.train(max_epochs=10000)
    # net.test()