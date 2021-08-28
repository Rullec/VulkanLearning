import argparse
import datetime
from torch.distributed.distributed_c10d import init_process_group
import os
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
    trainset = ToyDataset(datadir, transform=transform, phase_train=True)
    testset = ToyDataset(datadir, transform=transform, phase_train=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=128,
                                              num_workers=6,
                                              sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=128,
                                             num_workers=6,
                                             sampler=test_sampler)

    return trainloader, testloader


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=None, type=str)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--model_path", default=None, type=str)

args = parser.parse_args()
data_dir = args.data_dir
local_rank = args.local_rank
init_model_path = args.model_path

device = local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

trainloader, testloader = get_dataset(data_dir)
input_mean = trainloader.dataset.input_mean
input_std = trainloader.dataset.input_std
output_mean = trainloader.dataset.output_mean
output_std = trainloader.dataset.output_std

if init_model_path is None:
    model = ToyModel(input_mean,
                     input_std,
                     output_mean,
                     output_std,
                     load_pretrained_weights=False).to(device)
else:
    assert os.path.exists(
        init_model_path), f"init model path {init_model_path} doesn't exist"
    model = torch.load(init_model_path).to(device)
    print(f"[log] load model from {init_model_path}")

# model._freeze_backbone_except_1st_conv()

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
cur_lr = 5e-4
optimizer = torch.optim.Adam(model.parameters(), lr=cur_lr, weight_decay=1e-4)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import datetime

tfb_writer = SummaryWriter("../log/tensorboard_log/" +
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +
                           "/")
loss_func = nn.MSELoss().to(device)

model.train()
lr_decay = 0.995
min_lr = 3e-6
saving_epoch = 30
for epoch in range(10000):
    train_loss = []
    st = time.time()
    model.train()
    for data, label in trainloader:
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        prediction = model(data)

        single_loss = loss_func(prediction, label)
        single_loss.backward()
        optimizer.step()
        train_loss.append(float(single_loss))

    # ----------------
    test_loss = []
    model.eval()
    for data, label in testloader:
        data, label = data.to(device), label.to(device)
        prediction = model(data)
        single_loss = loss_func(prediction, label)

        # stats
        test_loss.append(float(single_loss))
        # print(f"test {data.shape}")

    if local_rank == 0:
        med_train_loss = np.median(train_loss)
        med_test_loss = np.median(test_loss)
        print(
            f"epoch {epoch} rank0 train loss {med_train_loss:.4f} test loss {med_test_loss:.4f} lr {cur_lr:.5e} cost {time.time() - st:.3f} s"
        )
        tfb_writer.add_scalar("lr", cur_lr, epoch)
        tfb_writer.add_scalar("test_loss", med_test_loss, epoch)
        tfb_writer.add_scalar("train_loss", med_train_loss, epoch)

        if epoch % saving_epoch == 0 and epoch != 0:
            path = f"../log/epoch{epoch}-{med_test_loss:.3f}.pth"
            torch.save(model, path)
            print(f"save model {path}")

    cur_lr = max(optimizer.param_groups[0]['lr'] * lr_decay, min_lr)
    optimizer.param_groups[0]['lr'] = cur_lr
