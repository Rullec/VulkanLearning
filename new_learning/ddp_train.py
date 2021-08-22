import argparse
from torch.distributed.distributed_c10d import init_process_group
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from resnet_core import ToyModel
from dataset import ToyDataset
from dataaug import get_torch_transform, get_albumentation_aug
import numpy as np
import time

torch.manual_seed(0)
np.random.seed(0)


def get_dataset(datadir):
    transform = get_albumentation_aug()
    # transform = None
    trainset = ToyDataset(datadir, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=128,
                                              num_workers=6,
                                              sampler=train_sampler)
    return trainloader


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=None, type=str)
parser.add_argument("--local_rank", default=-1, type=int)

args = parser.parse_args()
data_dir = args.data_dir
local_rank = args.local_rank
device = local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

trainloader = get_dataset(data_dir)
input_mean = trainloader.dataset.input_mean
input_std = trainloader.dataset.input_std
output_mean = trainloader.dataset.output_mean
output_std = trainloader.dataset.output_std
model = ToyModel(input_mean,
                 input_std,
                 output_mean,
                 output_std,
                 load_pretrained_weights=False).to(device)
# model._freeze_backbone_except_1st_conv()

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
cur_lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=cur_lr, weight_decay=1e-3)
from torch.optim.lr_scheduler import LambdaLR

loss_func = nn.MSELoss().to(device)

model.train()
lr_decay = 0.992
min_lr = 1e-6

for epoch in range(1000):
    train_loss = 0
    num_data = 0
    st = time.time()
    for data, label in trainloader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        single_loss = loss_func(prediction, label)
        single_loss.backward()
        optimizer.step()

        # stats
        train_loss += single_loss.detach().cpu() * data.shape[0]
        num_data += data.shape[0]

    train_loss /= num_data

    if local_rank == 0:
        print(f"epoch {epoch} train loss {train_loss:.4f} lr {cur_lr:.5e} cost {time.time() - st} s")
    
    cur_lr = max(optimizer.param_groups[0]['lr'] * lr_decay, min_lr)
    optimizer.param_groups[0]['lr'] = cur_lr
