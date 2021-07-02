import sys

sys.path.append("../../learning")
import os
from image_data_loader import ImageDataLoader, get_pngs_and_features, load_single_data

dataset_dir = "500_to_5000_mesh_small_gen"
mesh_data_lst = get_pngs_and_features(dataset_dir)
png_files_lst = []
feature_file_lst = []

for i in mesh_data_lst:
    png_files_lst.append(i.png_files)
    feature_file_lst.append(i.feature_file)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def default_loader(png_files, feature_file):
    # print(
    #     f"[debug] default loader, png files {png_files}, feature file {feature_file}"
    # )
    image, feature = load_single_data(png_files,
                                                      feature_file,
                                                      enable_log_pred=True)
    return image, feature
    # exit(1)
    # img_pil = Image.open(path)
    # img_pil = img_pil.resize((224, 224))
    # img_tensor = preprocess(img_pil)
    # return None


class trainset(Dataset):
    def __init__(self, inputs, output, loader=default_loader):
        #定义好 image 的路径
        self.images = inputs
        self.target = output
        self.loader = loader

    def __getitem__(self, index):
        img_path = self.images[index]
        feature_path = self.target[index]
        img, target = self.loader(img_path, feature_path)
        return img, target

    def __len__(self):
        return len(self.images)

png_files_lst = png_files_lst[:3000]
feature_file_lst = feature_file_lst[:3000]
train_data = trainset(inputs=png_files_lst, output=feature_file_lst)
trainloader = DataLoader(train_data, batch_size=1024, shuffle=True)
print(len(train_data))
iter = iter(trainloader)
import time
while True:
    try:
        st = time.time()
        images, labels = iter.next()
        print(f"cost {time.time() - st} s")
        print(images.shape)
        print(labels.shape)
    except StopIteration as e:
        print(f"finished {e}")
        break