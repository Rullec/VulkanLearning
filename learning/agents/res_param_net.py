from .param_net import ParamNet
import sys

sys.path.append("../data_loader/")
from data_loader.img_data_mani import ImageDataManipulator
from data_loader.dali_data_mani import DALIDataManipulator
from net_core import cnn_net
import torch
import time
from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
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
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class CNNParamNet(ParamNet):
    '''
    Resnet Neural Network 
    receiving image input
    '''
    NAME = "CNNParamNet"
    IMAGE_DATALOADER_TYPE_KEY = "image_dataloader_type"
    INPUT_NORMALZE_MODE_KEY = "input_normalize_mode"

    def __init__(self, config_path, device):
        super().__init__(config_path, device)

    def _load_param(self):
        super()._load_param()
        self.image_dataloader_type = self.conf[
            CNNParamNet.IMAGE_DATALOADER_TYPE_KEY]

    def _build_dataloader(self):
        data_mani = ImageDataManipulator(self.conf[self.DATA_LOADER_KEY])
        # data_mani = DALIDataManipulator(self.conf[self.DATA_LOADER_KEY])
        self.train_dataloader, self.test_dataloader = data_mani.get_dataloader(
        )
        self.input_size = self.train_dataloader.get_input_size()
        self.output_size = self.train_dataloader.get_output_size()[0]

    def _build_net(self):

        # print(self.output_size)
        # exit(0)
        # self.net = cnn_net(self.layers, self.output_size,
        #                    self.dropout).to(self.device)
        # self.net = ResModel(self.output_size).to(self.device)


        ## resnet 50
        # self.net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes = self.output_size).to(self.device)
        ## resnet 18
        self.net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes = self.output_size).to(self.device)

        self.criterion = torch.nn.MSELoss()
        total = 0
        for i in self.net.parameters():
            total += i.numel()
        print(f"[debug] build resnet succ, total param {total}")
        # exit()

    def train(self, max_epochs=1000):
        st_time = time.time()
        # print("[debug] begin training epoch")
        for epoch in range(max_epochs):
            epoch_st_time = time.time()
            # have an iteration
            cur_epoch_train_loss = 0
            iters = 0
            total_num = 0

            total_train_cost_time = 0
            total_datafetch_cost_time = 0
            # dataload_start = time.time()
            st_epoch = time.time()
            for i_batch, sampled_batched in enumerate(
                    tqdm(self.train_dataloader,
                         total=len(self.train_dataloader))):
                # profiling
                # dataload_finish = time.time()
                # total_datafetch_cost_time += dataload_finish - dataload_start
                # train_start = time.time()
                # begin to train
                self.net.train()
                inputs, outputs = sampled_batched
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)
                num = inputs.shape[0]
                if num == 1:
                    continue
                self.optimizer.zero_grad()
                pred = self.net(inputs)
                loss = self.criterion(pred, outputs).to(self.device)
                if np.isnan(loss.cpu().detach()) == True:

                    print(f"input has Nan: {np.isnan(inputs.cpu()).any()}")
                    print(
                        f"pred has Nan: {np.isnan(pred.cpu().detach()).any()}")
                    print(f"gt has Nan: {np.isnan(outputs.cpu()).any()}")
                    print(f"loss: {loss.cpu()}")
                    for i in self.net.parameters():
                        print(
                            f"weight has Nan: {np.isnan( i.cpu().detach()).any()}"
                        )
                    exit(0)
                loss.backward()
                self.optimizer.step()
                train_end = time.time()
                # total_train_cost_time += train_end - train_start

                # print(f"batch train cost {ed_epoch - st_epoch} s")
                # print(f"[train] single mse {loss} num {inputs.shape[0]}")
                cur_epoch_train_loss += loss.item() * num
                iters += 1
                total_num += num
                # dataload_start = time.time()
                # st6 = time.time()
                # print(f"4 {st6 - st5}")
            ed_epoch = time.time()
            print(f"train epoch cost {ed_epoch - st_epoch}")
            mean_train_loss = cur_epoch_train_loss / total_num
            # print(f"[train] total err {mean_train_loss} num {total_num}")
            # logging
            if epoch % self.iters_logging == 0:
                print(f"begin to do validation...")
                step = epoch / self.iters_logging
                validation_err = self._calc_validation_error()
                print(
                    f"iter {epoch} train loss {mean_train_loss:5.5f} validation loss {validation_err:5.5f}, cost {(time.time() - epoch_st_time):2.1f}, device {self.device}"
                )
                self.writer.add_scalar("train_loss", mean_train_loss, step)
                self.writer.add_scalar("validation_error", validation_err,
                                       step)
                self.writer.add_scalar("lr", self._get_lr(), step)
                # if validation_err < self.covg_threshold:
                #     break

            # saving model
            if epoch % self.iters_save_model == 0:
                print(f"begin to save model...")
                name = self._get_model_save_name(float(validation_err))
                self.save_model(name)
                print(f"save model done")
                # print(f"name {name}")

            # update hyper parameters
            self._set_lr(max(self.lr_decay * self._get_lr(), self.min_lr))
            print(f"done an epoch")
        return validation_err