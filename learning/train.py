import numpy as np
import torch
from param_net import ParamNet

np.random.seed(0)
torch.manual_seed(0)
is_cuda_avaliable = torch.cuda.is_available()
assert is_cuda_avaliable == True

if is_cuda_avaliable == True:
    device = torch.device("cuda", 0)
else:
    device = torch.device("cpu", 0)

conf_path = "..\config\\train_configs\\train_config.json"
net = ParamNet(conf_path, device)
net.train()