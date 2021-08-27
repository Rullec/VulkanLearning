import argparse
import torch
from resnet_core import ToyModel
from dataset import ToyDataset
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


def get_dataset(datadir):
    # transform = get_albumentation_aug()
    transform = None
    testset = ToyDataset(datadir, transform=transform, phase_train=False)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=128,
                                             num_workers=6,
                                             sampler=None)
    return testloader


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--model_path", type=str)

args = parser.parse_args()
data_dir = args.data_dir
model_path = args.model_path
assert model_path is not None
assert data_dir is not None

test_loader = get_dataset(data_dir)

input_mean = test_loader.dataset.input_mean
input_std = test_loader.dataset.input_std
output_mean = test_loader.dataset.output_mean
output_std = test_loader.dataset.output_std

# model = ToyModel(input_mean,
#                  input_std,
#                  output_mean,
#                  output_std,
#                  load_pretrained_weights=False)

device = torch.device("cpu")
model = torch.load(model_path).to(device)
print(f"load model from {model_path}")

model.eval()
# print(next(test_loader))
# exit()
for data, label in test_loader:
    data = data[:1]
    label = label[:1]
    # print(f"data shape {data.shape}")
    pred = model(data).detach().cpu()

    pred = np.array(pred)[0]
    data = np.array(data)[0]
    label = np.array(label)[0]

    # print(f"label shape {label.shape}")
    print(f"label {label}")
    print(f"pred {pred}")
    print(f"diff {label - pred}")
