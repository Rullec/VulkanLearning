import torch
import torch.nn as nn
import torch.nn.functional as F


class fc_net(nn.Module):
    '''
    Full net
    '''
    def __init__(self, input_size, layers, output_size, device):
        super(fc_net, self).__init__()

        # define layers
        self.device = device
        self.input = nn.Linear(input_size, layers[0])
        self.middle_layers = []
        for j in range(len(layers) - 1):
            self.middle_layers.append(nn.Linear(layers[j], layers[j + 1]))
        self.middle_layers = torch.nn.ModuleList(self.middle_layers)
        self.output = nn.Linear(layers[-1], output_size)

    def forward(self, x):
        x = self.input(x).to(self.device)
        x = F.relu(x).to(self.device)

        for i in self.middle_layers:
            x = i(x).to(self.device)
            x = F.relu(x).to(self.device)

        x = self.output(x).to(self.device)

        # x = F.relu(x).to(self.device)

        return x


from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3


class GrayscaleResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 planes,
                 last_fc_input=64,
                 last_fc_output=32,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(GrayscaleResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        channels = 1
        assert len(planes) == 4
        self.inplanes = planes[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(channels,
                               self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes[1], layers[0])
        self.layer2 = self._make_layer(block,
                                       planes[2],
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       planes[3],
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       last_fc_input,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_fc_input * block.expansion, last_fc_output)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # print(f"input x shape {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(f"before fc x shape {x.shape}")
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class res_net(GrayscaleResNet):
    '''
        Resnet implemention for depth image(c=1)
    '''
    def __init__(self, layers, output_size):

        # change the default resnet18 network
        # super().__init__(block=BasicBlock, layers=[2, 2, 2, 2])
        # planes = [64, 64, 128, 256]
        # planes = [16, 16, 32, 64]
        planes = [4, 4, 8, 32]
        super().__init__(block=BasicBlock,
                         layers=[1, 1, 1, 1],
                         planes=planes,
                         last_fc_input=16,
                         last_fc_output=layers[0])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # output fc layers
        self.middle_layers = []
        for j in range(len(layers) - 1):
            self.middle_layers.append(nn.Linear(layers[j], layers[j + 1]))
        self.middle_layers = torch.nn.ModuleList(self.middle_layers)
        self.output = nn.Linear(layers[-1], output_size)

    def forward(self, x):
        x = super().forward(x)
        # add an relu after the FC
        x = F.relu(x)

        for i in self.middle_layers:
            x = i(x)
            x = F.relu(x)
        x = self.output(x)
        return x