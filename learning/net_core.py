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

    return 


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
    return 


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
        norm_layer = nn.BatchNorm2d
        self.inplanes = 4
        self.conv1 = nn.Conv2d(self.inplanes, 32, kernel_size=7, stride=2)
        self.bn1 = norm_layer(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = norm_layer(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn3 = norm_layer(128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.bn4 = norm_layer(64)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn5 = norm_layer(32)
        self.conv6 = nn.Conv2d(32, 8, kernel_size=3, stride=1)
        self.bn6 = norm_layer(8)
        self.conv7 = nn.Conv2d(8, 4, kernel_size=3, stride=1)
        self.bn7 = norm_layer(4)
        # self.avg_pool = nn.AdaptiveAvgPool1d(8192)
        self.avg_pool = nn.AdaptiveAvgPool2d((32, 32))

        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 64)
        self.output_fc = nn.Linear(64, output_size)
        self.act = F.relu
        
    def forward(self, x):

        enable_logging = False
        if enable_logging:
            print(f"input shape {x.shape}")
       
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)

        # print(output.shape)
        # output = torch.flatten(output, 1)
        # print(output.shape)
        # output = torch.squeeze(output, 1)
        # print(output.shape)
        output = self.avg_pool(output)
        output = torch.flatten(output, 1)
        # print(output.shape)
        output = self.act(self.fc1(output))
        output = self.act(self.fc2(output))
        output = self.act(self.fc3(output))
        output = self.act(self.fc4(output))
        output = self.act(self.fc5(output))
        output = self.output_fc(output)

        x = torch.squeeze(output)
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