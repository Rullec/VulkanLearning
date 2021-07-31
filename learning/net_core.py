from numpy import true_divide
import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding


class fc_net(nn.Module):
    '''
    Full net
    '''
    def __init__(self, input_size, layers, output_size, device, dropout=0):
        super(fc_net, self).__init__()
        # define layers
        self.device = device
        if len(layers) > 0:
            self.input = nn.Linear(input_size, layers[0])
            self.middle_layers = []
            for j in range(len(layers) - 1):
                self.middle_layers.append(nn.Linear(layers[j], layers[j + 1]))
            self.middle_layers = torch.nn.ModuleList(self.middle_layers)
            self.output = nn.Linear(layers[-1], output_size)
        else:
            self.input = nn.Linear(input_size, output_size)
            self.output = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.output is not None:
            x = self.dropout(self.input(x)).to(self.device)
            x = F.relu(x).to(self.device)

            for i in self.middle_layers:
                x = self.dropout(i(x)).to(self.device)
                x = F.relu(x).to(self.device)

            x = self.dropout(self.output(x)).to(self.device)
        else:
            x = self.dropout(self.input(x))
        # x = F.relu(x).to(self.device)

        return x


from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3


def build_conv2d_block(config):
    KERNEL_SIZE_KEY = "kernel_size"
    STRIDE_KEY = "stride"
    INPLANE_KEY = "inplanes"
    OUTPLANE_KEY = "outplanes"
    PADDING_KEY = "padding"
    kernel_size = config[KERNEL_SIZE_KEY]
    stride_size = config[STRIDE_KEY]
    inplanes = config[INPLANE_KEY]
    outplanes = config[OUTPLANE_KEY]
    padding = config[PADDING_KEY]

    return nn.Conv2d(inplanes,
                     outplanes,
                     kernel_size=kernel_size,
                     stride=stride_size,
                     padding=padding)


def build_conv1d_block(config):
    KERNEL_SIZE_KEY = "kernel_size"
    STRIDE_KEY = "stride"
    INPLANE_KEY = "inplanes"
    OUTPLANE_KEY = "outplanes"
    PADDING_KEY = "padding"
    kernel_size = config[KERNEL_SIZE_KEY]
    stride_size = config[STRIDE_KEY]
    inplanes = config[INPLANE_KEY]
    outplanes = config[OUTPLANE_KEY]
    padding = config[PADDING_KEY]

    return nn.Conv1d(inplanes,
                     outplanes,
                     kernel_size=kernel_size,
                     stride=stride_size,
                     padding=padding)


def build_avg_pool_block(config):
    OUTPUT_KEY = "output"
    size = config[OUTPUT_KEY]
    return nn.AdaptiveAvgPool1d(size)


def build_fc_block(config):
    INPUT_KEY = "input"
    OUTPUT_KEY = "output"
    input_size = config[INPUT_KEY]
    output_size = config[OUTPUT_KEY]
    return nn.Linear(input_size, output_size)


def build_end_fc_block(config, output_size):
    INPUT_KEY = "input"
    input_size = config[INPUT_KEY]
    return nn.Linear(input_size, output_size)


class cnn_net(nn.Module):
    '''
        common CNN implemention for depth image(c=1)
    '''
    def __init__(self, layers_config, output_size, dropout):
        super(cnn_net, self).__init__()
        assert dropout == 0.0, "dropout must be 0.0"

        self.type_lst = []
        self.layer_lst = []
        for config in layers_config:
            type = config["type"]
            self.type_lst.append(type)
            if type == "conv2d":
                block = build_conv2d_block(config)
            elif type == "conv1d":
                block = build_conv1d_block(config)
            elif type == "avgpool":
                block = build_avg_pool_block(config)
            elif type == "fc":
                block = build_fc_block(config)
            elif type == "end_fc":
                block = build_end_fc_block(config, output_size)
            self.layer_lst.append(block)
        # self.dropout = dropout
        self.layer_lst = torch.nn.ModuleList(self.layer_lst)

    def forward(self, x):

        enable_logging = False
        if enable_logging:
            print(f"input shape {x.shape}")
        for _idx in range(len(self.type_lst)):
            try:
                layer = self.layer_lst[_idx]
                type = self.type_lst[_idx]
                if type == "avgpool":
                    x = torch.flatten(x, 1)
                    x = torch.unsqueeze(x, 1)
                    # print(f"after flatten shape {x.shape}")
                    if enable_logging:
                        print(f"after flatten shape {x.shape}")
                x = layer(x)
                if type == "fc":
                    x = F.relu(x)
                if enable_logging:
                    print(
                        f"after layer {_idx} {type} output shape = {x.shape}")
            except Exception as e:
                print(f"[error] got exception in layer {_idx} {type}, {e}")
                exit(1)
        x = torch.squeeze(x)
        if enable_logging:
            print(f"output shape {x.shape}")
        return x
        # x = torch.flatten(x, 1)
        # x = self.dropout(self.input(x))
        # x = F.relu(x)

        # for i in self.middle_layers:
        #     x = self.dropout(i(x))
        #     x = F.relu(x)

        # x = self.dropout(self.output(x))

        # return x
# import torchvision

# class res_net(nn.Module):
#     '''
#         common CNN implemention for depth image(c=1)
#     '''
#     def __init__(self, fc_layers, output_size):
#         super(res_net, self).__init__()
#         # if backbone_name == 'resnet_18':

#         # use a fixed resnet18 backbone
#         resnet_net = torchvision.models.resnet18(pretrained=True)
#         # resnet_net = torchvision.models.resnet50(pretrained=True)
#         modules = list(resnet_net.children())[:-1]
#         self.backbone = nn.Sequential(*modules)
#         self.backbone.out_channels = fc_layers[0]
#         for i in self.backbone.parameters():
#             i.requires_grad = False

#         # add fc head
#         self.fc_layer_lst = []
#         for j in range(len(fc_layers) - 1):
#             self.fc_layer_lst.append(nn.Linear(fc_layers[j], fc_layers[j + 1]))

#         self.fc_layer_lst = torch.nn.ModuleList(self.fc_layer_lst)

#         self.output = nn.Linear(fc_layers[-1], output_size)

#     def forward(self, x):
#         x = x.repeat(1, 3, 1, 1)
#         # print(f"input {x.shape}")
#         x = torch.squeeze(self.backbone(x))
#         # print(f"resnet {x.shape}")

#         for fc_layer in self.fc_layer_lst:
#             x = fc_layer(x)
#             x = F.relu(x)

#         x = self.output(x)
#         # print(f"output {x.shape}")
#         # exit(0)

#         return x