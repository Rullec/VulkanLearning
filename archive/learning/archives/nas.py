import nevergrad as ng
import json
import os
from shutil import copyfile
from res_param_net import CNNParamNet
import numpy as np
import torch
from train import build_net

samples = 0


def train(filepath):
    assert os.path.exists(filepath) == True
    np.random.seed(0)
    torch.manual_seed(0)
    is_cuda_avaliable = torch.cuda.is_available()
    assert is_cuda_avaliable == True

    if is_cuda_avaliable == True:
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu", 0)
    # net = ParamNet(conf_path, device)

    # conf_path = "..\config\\train_configs\\fc_conf.json"
    net_type = build_net(filepath)
    net = net_type(filepath, device)
    validation_error = float(net.train(max_epochs=301))
    return validation_error
    # print(f"validation error {validation_error}")
    # exit(0)


def fake_training(learning_rate: float, lr_decay: float, weight_decay: float,
                  layer0: int, layer1: int, layer2: int, layer3: int,
                  dropout: float, batch_size: int) -> float:
    # 1. create new config
    template_config = "..\config\\train_configs\\conv_conf.json"
    filename = os.path.split(template_config)[-1]
    target_config = "tmp_conf.json"
    copyfile(template_config, target_config)

    with open(target_config) as f:
        cont = json.load(f)
        cont[CNNParamNet.LEANING_RATE_KEY] = learning_rate
        cont[CNNParamNet.LEANING_RATE_DECAY_KEY] = lr_decay
        cont[CNNParamNet.WEIGHT_DECAY_KEY] = weight_decay
        cont[CNNParamNet.LAYERS_KEY] = [layer0, layer1, layer2, layer3]
        cont[CNNParamNet.DROPOUT_KEY] = dropout
        cont[CNNParamNet.BATCH_SIZE_KEY] = batch_size
    with open(target_config, 'w') as f:
        json.dump(cont, f)
    error = train(target_config) * 1000
    print(f"[sample] sample {cont} val_err {error}")
    return error
    # print(filename)
    # exit(0)
    # 2. write down new config
    # 3. train 1000 epoch, get validation error. error * 1000, logout then return

    res = learning_rate + lr_decay + weight_decay + layer0 + layer1 + layer2 + layer3 + dropout + batch_size
    global samples
    samples += 1
    print(f"[sample] {samples} {res}")
    # res *= -1
    return res


# Instrumentation class is used for functions with multiple inputs
# (positional and/or keywords)
parametrization = ng.p.Instrumentation(
    learning_rate=ng.p.Log(lower=1e-5, upper=1e-3),
    lr_decay=ng.p.Scalar(lower=0.997, upper=1),
    weight_decay=ng.p.Log(lower=1e-4, upper=1e-1),
    layer0=ng.p.Choice([16, 32, 64, 128, 256, 512, 1024, 2048]),
    layer1=ng.p.Choice([8, 16, 32, 64, 128, 256, 512]),
    layer2=ng.p.Choice([4, 8, 16, 32, 64, 128]),
    layer3=ng.p.Choice([2, 4, 8, 16, 32]),
    dropout=ng.p.Scalar(lower=0, upper=0.5),
    batch_size=ng.p.Scalar(lower=16, upper=128).set_integer_casting(),
)

budgets = 200
optimizer = ng.optimizers.registry["CMA"](parametrization=parametrization,
                                          budget=budgets)
recommendation = optimizer.minimize(fake_training)

# show the recommended keyword arguments of the function
print(recommendation.kwargs)

# for i in range()