import torch.nn as nn
import torch


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    #inplanes其实就是channel,叫法不同
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes),
            )
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #把shortcut那的channel的维度统一
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MultiviewNet(nn.Module):
    def __init__(self, input, output):
        super(MultiviewNet, self).__init__()
        self.input = input
        
        self.total_conv = nn.Sequential(
            nn.Conv2d(
                1,
                64,
                kernel_size=7,
                stride=2,
                padding=3,  #因为mnist为（1，28，28）灰度图，因此输入通道数为1
                bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
             nn.Linear(4096, 64),
             nn.ReLU(inplace=True),
             nn.BatchNorm1d(64),
             nn.Linear(64, output)
        )

    def forward(self, x):
        x0 = torch.unsqueeze(x[:, 0, :, :], dim = 1)
        x0 = self.total_conv(x0)
        # print(x0.shape)
        # exit()
        x0 = self.pool(x0)
        x1 = torch.unsqueeze(x[:, 1, :, :], dim = 1)
        x1 = self.total_conv(x1)
        x1 = self.pool(x1)
        x2 = torch.unsqueeze(x[:, 2, :, :], dim = 1)
        x2 = self.total_conv(x2)
        x2 = self.pool(x2)
        x3 = torch.unsqueeze(x[:, 3, :, :], dim = 1)
        x3 = self.total_conv(x3)
        x3 = self.pool(x3)
            
        
        output = torch.cat([x0, x1, x2, x3], dim = 1)
        output = torch.squeeze(output)
        
        output = self.fc(output)
        
        return output