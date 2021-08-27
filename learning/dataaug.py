import albumentations as A
from albumentations.augmentations.transforms import ImageOnlyTransform


class AlbumentationAxisRoll(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(AlbumentationAxisRoll, self).__init__(always_apply, p)

    def apply(self, image, **params):
        num_of_views = image.shape[-1]
        assert num_of_views == 4
        shift = np.random.randint(0, num_of_views)
        if shift != 0:
            image = np.roll(image, shift, axis=0)
        return image

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ()


class AlbumentationRandomNoise(ImageOnlyTransform):
    def __init__(self, variance, always_apply=False, p=0.5):
        self.variance = variance
        super(AlbumentationRandomNoise, self).__init__(always_apply, p)

    def apply(self, image, **params):
        noise = np.random.normal(loc=0,
                                 scale=self.variance,
                                 size=image.shape[:-1]).astype(np.uint8)
        image += noise[:, :, None]
        return image

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ()


class AlbumentationCHWToHWC(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1):
        assert always_apply == True
        super(AlbumentationCHWToHWC, self).__init__(always_apply, p)

    def apply(self, image, **params):
        image = np.moveaxis(image, 0, -1)
        return image

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ()


class AlbumentationHWCToCHW(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1):
        assert always_apply == True
        super(AlbumentationHWCToCHW, self).__init__(always_apply, p)

    def apply(self, image, **params):
        image = np.moveaxis(image, 2, 0)
        return image

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ()


class AlbumentationFullBias(ImageOnlyTransform):
    def __init__(self, bias_limit, always_apply=True, p=1):
        self.bias_limit = bias_limit
        assert always_apply == True
        super(AlbumentationFullBias, self).__init__(always_apply, p)

    def apply(self, image, **params):
        # [-bias_limit, bias_limit]
        rand = np.uint8(np.random.randint(low=0, high=self.bias_limit))
        # for all nonzero pixel, add noise
        if rand % 2 == 0:
            image += (image != 0) * rand
        else:
            image -= (image != 0) * rand
        # image = image.astype(np.uint8)
        return image

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ()


def get_albumentation_aug():

    aug = A.Compose([
        AlbumentationCHWToHWC(),
        A.augmentations.geometric.transforms.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=10,
            always_apply=True),
        A.augmentations.transforms.Blur(blur_limit=3, p=0.7),
        AlbumentationRandomNoise(variance=1, always_apply=False, p=0.5),    # it make the training very very slow
        AlbumentationFullBias(bias_limit=10),
        AlbumentationAxisRoll(always_apply=True),
        AlbumentationHWCToCHW(),
    ])
    return aug


from torchvision.transforms import RandomAffine, GaussianBlur
from torchvision import transforms
import torch
import numpy as np


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.mean, self.std)


class ChangeChannel(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        assert len(tensor.shape) == 3
        num_of_views = tensor.shape[0]
        shift = np.random.randint(0, num_of_views)
        assert type(shift) == int
        # print(f"before {tensor.shape}")
        tensor = torch.roll(tensor, shift, dims=0)
        # print(f"after {tensor.shape}")
        return tensor

    def __repr__(self):
        return "depth view roll"


def get_torch_transform():
    affine = RandomAffine(degrees=(-5, 5),
                          translate=(0.07, 0.07),
                          scale=(0.9, 1.1))
    blur = GaussianBlur(kernel_size=3)
    gaussian_noise = AddGaussianNoise(mean=0, std=0.1)
    roll = ChangeChannel()
    trans_aug = transforms.Compose(
        [transforms.ToTensor(), affine, blur, gaussian_noise, roll])
    return trans_aug


if __name__ == "__main__":
    import glob
    import os
    target_dir = "../data/export_data/uniform_3c_sample25_noised16_xgpu_initrot1_cam1.small/depth/mesh0/init_rot0_cam0"
    assert os.path.exists(target_dir)
    pngs = glob.glob(f"{target_dir}/*png")
    # 1. load the png
    import cv2
    raw_img = np.array([cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in pngs])
    # print(raw_img.shape)

    from dataaug import get_albumentation_aug
    data_aug = get_albumentation_aug()
    new_img0 = data_aug(image=raw_img)['image']
    new_img1 = data_aug(image=raw_img)['image']
    new_img2 = data_aug(image=raw_img)['image']

    import matplotlib.pyplot as plt
    for i in range(4):
        plt.subplot(3, 4, 4 * 0 + i + 1)
        plt.imshow(new_img0[i])
    for i in range(4):
        plt.subplot(3, 4, 4 * 1 + i + 1)
        plt.imshow(new_img1[i])
    for i in range(4):
        plt.subplot(3, 4, 4 * 2 + i + 1)
        plt.imshow(new_img2[i])
    plt.savefig("res.png")
