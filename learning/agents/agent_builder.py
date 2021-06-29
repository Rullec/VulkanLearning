import json
from .res_param_net import CNNParamNet
from .param_net import ParamNet


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
