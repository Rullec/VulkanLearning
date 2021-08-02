import numpy as np
import torch
from torch.functional import Tensor
from torchvision.transforms import RandomAffine, GaussianBlur
from torchvision import transforms
from torchvision.transforms.transforms import RandomRotation, ToTensor


def apply_mesh_data_noise(inputs):
    size = inputs.shape[0]
    noise = (np.random.rand(3).astype(np.float32) - 0.5) / 10  # +-5cm, 3x1

    noise_all = np.tile(noise, size // 3)
    # print(f"noise all shape {noise_all.shape} {noise_all}")
    # exit()
    inputs += noise_all
    # for _idx in range(inputs.shape[0]):
    #     # print(f"noise = {noise}")
    #     inputs[_idx] += noise_all
    return inputs


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.mean, self.std)


affine = RandomAffine(degrees=(-5, 5),
                      translate=(0.07, 0.07),
                      scale=(0.9, 1.1))
blur = GaussianBlur(kernel_size=3)
gaussian_noise = AddGaussianNoise(mean=0, std=0.1)
trans_aug = transforms.Compose([affine, blur, gaussian_noise])

import time


def apply_depth_aug(inputs):
    inputs = torch.from_numpy(inputs)
    # st = time.time()
    inputs = trans_aug(inputs)
    # ed = time.time()
    # print(f"torch aug cost {ed - st} s")
    # data_transform_affine_blur_noise = transforms.Compose(
    #     [affine, blur, gaussian_noise])
    # gaussian_noise = AddGaussianNoise(mean=0, std=0.04)
    # data_transform_affine_blur_noise = transforms.Compose(
    #     [affine, blur])
    # return data_transform_affine_blur_noise
    # return blur
    return inputs


import albumentations as A

a_ssr = A.ShiftScaleRotate(p=1.0)
a_gb = A.GaussianBlur(p=1.0)
a_gn = A.GaussNoise(p=1.0)
trans_aug_A = A.Compose([a_ssr, a_gb, a_gn])


def apply_depth_albumentation(inputs):
    st = time.time()
    inputs = trans_aug_A(image=inputs)["image"]
    ed = time.time()
    # print(f"albumentations aug cost = {ed - st}")
    return inputs


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if __name__ == "__main__":
    import sys
    sys.path.append("../../calibration")
    from PIL import Image
    image = Image.open("chessboard.png")
    image = image.resize((300, 300))
    image = np.array(image) / 255.9
    image = np.mean(image, axis=2)
    # print(image.shape)
    # exit()
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    from drawer_util import DynaPlotter
    res = DynaPlotter(1, 2, iterative_mode=False)
    print(image.dtype)
    res.add(image)
    trans = RandomRotation(degrees=(-5, 5))
    # print(type(image))
    # exit(1)
    image = trans(image)
    print(image.dtype)
    res.add(image)
    res.show()
