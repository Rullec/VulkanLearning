import numpy as np
import torch
from torchvision.transforms import RandomAffine, GaussianBlur
from torchvision import transforms
from torchvision.transforms.transforms import RandomRotation


def apply_mesh_data_noise(batch):
    inputs, outputs = batch
    size = inputs.shape[1]
    noise = (np.random.rand(3).astype(np.float32) - 0.5) / 10  # +-5cm
    noise_all = np.repeat([np.tile(noise, size // 3)], inputs.shape[0], axis=0)
    inputs += noise_all
    # for _idx in range(inputs.shape[0]):
    #     # print(f"noise = {noise}")
    #     inputs[_idx] += noise_all
    return (inputs, outputs)


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
                        scale=(1.0, 1.0))
blur = GaussianBlur(kernel_size=3)
gaussian_noise = AddGaussianNoise(mean=0, std=0.04)
trans_aug = transforms.Compose([affine, blur, gaussian_noise])

def apply_depth_aug(batch):
    inputs, outputs = batch
    inputs = trans_aug(inputs)
    # data_transform_affine_blur_noise = transforms.Compose(
    #     [affine, blur, gaussian_noise])
    # gaussian_noise = AddGaussianNoise(mean=0, std=0.04)
    # data_transform_affine_blur_noise = transforms.Compose(
    #     [affine, blur])
    # return data_transform_affine_blur_noise
    # return blur
    return (inputs, outputs)


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
