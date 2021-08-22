import argparse
from tqdm import tqdm
import torchvision
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataset import ToyDataset
from resnet_core import ToyModel
from dataaug import get_torch_transform, get_albumentation_aug

device = torch.device("cuda", 0)


def get_dataset(datadir):
    # transform = get_torch_transform()
    transform = get_albumentation_aug()
    # transform = None
    trainset = ToyDataset(datadir, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=128,
                                              num_workers=12)
    return trainloader


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=None, type=str)

args = parser.parse_args()
data_dir = args.data_dir

trainloader = get_dataset(data_dir)

input_mean = trainloader.dataset.input_mean
input_std = trainloader.dataset.input_std
output_mean = trainloader.dataset.output_mean
output_std = trainloader.dataset.output_std

model = ToyModel(input_mean, input_std, output_mean, output_std).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

loss_func = nn.MSELoss().to(device)

model.train()

for epoch in range(1):
    cur = 0
    for data, label in tqdm(trainloader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        optimizer.step()
        cur += 1