import torch
from .param_net import ParamNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from data_loader.img_data_mani import HDF5ImageDataManipulator
"""
A Convolutional Variational Autoencoder
"""
import math


class VAE(nn.Module):
    def _calc_pad_info_and_feature_info(self, image_shape):
        # 5 layers, stride = 2, 2^5 = 32
        layers = 5
        gap = 2**layers
        print(f"image shape {image_shape}")
        height = image_shape[1]
        width = image_shape[2]
        height_total_padding = math.ceil(height / gap) * gap - height
        width_total_padding = math.ceil(width / gap) * gap - width
        print(f"height {height}, pad {height_total_padding}")
        print(f"width {width}, pad {width_total_padding}")
        assert (height_total_padding % 2 == 0) and (width_total_padding % 2
                                                    == 0)
        return int(height_total_padding / 2), int(
            width_total_padding / 2), math.ceil(height / gap), math.ceil(
                width / gap)

    def __init__(self, image_shape, zDim=1024):
        super(VAE, self).__init__()
        self.height_pad_2, self.width_pad_2, self.height_feature_dim, self.width_feature_dim = self._calc_pad_info_and_feature_info(
            image_shape)

        # left, right, top, bottom
        self.preprocess_padding = torch.nn.ConstantPad2d(
            (self.width_pad_2, self.width_pad_2, self.height_pad_2,
             self.height_pad_2), 0)

        imgChannels = image_shape[0]
        self.feature_dim_lst = [
            16, self.height_feature_dim, self.width_feature_dim
        ]
        self.feature_dim = 1
        for i in self.feature_dim_lst:
            self.feature_dim *= int(i)
        # print(self.feature_dim_lst)
        # print(self.feature_dim)
        # exit()
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 8, 3, 2,
                                  padding=1)  # output [71,
        self.encConv2 = nn.Conv2d(8, 16, 3, 2, padding=1)
        self.encConv3 = nn.Conv2d(16, 16, 3, 2, padding=1)
        self.encConv4 = nn.Conv2d(16, 16, 3, 2, padding=1)
        self.encConv5 = nn.Conv2d(16, 16, 3, 2, padding=1)

        self.encFC = nn.Linear(self.feature_dim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, self.feature_dim)
        self.decConv1 = nn.ConvTranspose2d(16, 16, kernel_size = 1, output_padding=1, stride=2)
        self.decConv2 = nn.ConvTranspose2d(16, 16, kernel_size = 1, output_padding=1, stride=2)
        self.decConv3 = nn.ConvTranspose2d(16, 16, kernel_size = 1, output_padding=1, stride=2)
        self.decConv4 = nn.ConvTranspose2d(16, 8, kernel_size = 1, output_padding=1, stride=2)
        self.decConv5 = nn.ConvTranspose2d(8,
                                           imgChannels,
                                           kernel_size = 1, output_padding=1,
                                           stride=2)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        # print(f"[shape] {x.shape}")
        x = F.relu(self.encConv1(x))
        # print(f"[shape] {x.shape}")
        x = F.relu(self.encConv2(x))
        # print(f"[shape] {x.shape}")
        x = F.relu(self.encConv3(x))
        # print(f"[shape] {x.shape}")
        x = F.relu(self.encConv4(x))
        # print(f"[shape] {x.shape}")
        x = F.relu(self.encConv5(x))
        # print(f"[shape] {x.shape}")
        # print(x.shape)
        x = x.view(-1, self.feature_dim)
        x = self.encFC(x)
        return x

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, self.feature_dim_lst[0], self.feature_dim_lst[1],
                   self.feature_dim_lst[2])
        # print(f"[shape] {x.shape}")
        x = F.relu(self.decConv1(x))
        # print(f"[shape] {x.shape}")
        x = F.relu(self.decConv2(x))
        # print(f"[shape] {x.shape}")
        x = F.relu(self.decConv3(x))
        # print(f"[shape] {x.shape}")
        x = F.relu(self.decConv4(x))
        # print(f"[shape] {x.shape}")
        x = self.decConv5(x)
        # print(f"[shape] {x.shape}")
        return x

    def forward(self, x):
        # preprocess
        x = self.preprocess_padding(x)
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        x = self.encoder(x)
        out = self.decoder(x)

        # clip
        out = out[:, :, self.height_pad_2: out.shape[2] - self.height_pad_2 , self.width_pad_2: out.shape[3] - self.width_pad_2]
        return out


class VAENet(ParamNet):

    NAME = "VAENet"

    def __init__(self, config_path, device):
        super().__init__(config_path, device)

    def _load_param(self):
        super()._load_param()

    def _build_dataloader(self):
        mani = HDF5ImageDataManipulator(self.conf[self.DATA_LOADER_KEY])
        self.train_dataloader, self.test_dataloader = mani.get_dataloader()
        self.input_size = self.train_dataloader.get_input_size()
        self.output_size = self.train_dataloader.get_output_size()[0]

    def _build_net(self):
        image_shape = [4, 360, 480]
        self.net = VAE(image_shape=image_shape).to(self.device)
        total = 0
        for i in self.net.parameters():
            total += i.numel()
        print(f"[debug] build resnet succ, total param {total}")
        self.criterion = torch.nn.BCELoss()
        # exit()

    def train(self, max_epochs=100):
                
        for epoch in range(max_epochs):
            for idx, data in enumerate(tqdm(self.train_dataloader), 0):
                imgs, _ = data
                imgs = imgs.to(self.device)
                # Feeding a batch of images into the network to obtain the output image, mu, and logVar
                out = self.net(imgs)
                # print(f"out {out}")
                # print(f"mu {mu}")
                # print(f"logVar {logVar}")
                # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
                # kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) +
                #                                 logVar.exp())
                # print(f"kl div {kl_divergence}")
                # loss = F.binary_cross_entropy(
                #     out, imgs, size_average=False) + kl_divergence
                loss = self.criterion(out, imgs)
                # print(loss)
                # exit()
                # Backpropagation based on the loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('Epoch {}: Loss {}'.format(epoch, loss))
