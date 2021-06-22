import pickle

import numpy as np
with open("img.pkl", 'rb') as f:
    imgs = pickle.load(f)

# def rotate(raw_image):
#     from skimage.transform import rotate
#     angle = np.random.uniform(-20, 20)
#     return rotate(raw_image, angle)

samples = 10000


def affine(raw_image):
    from skimage.transform import AffineTransform
    from skimage.transform import warp
    scale = np.random.uniform(0.8, 1.1)
    rotation = np.random.uniform(-0.3, 0.3)
    height, width = raw_image.shape[0], raw_image.shape[1]
    translation = (int(np.random.uniform(-height / 6, height / 6)),
                   int(np.random.uniform(-width / 6, width / 6)))
    # translation = (0, 0)

    tform = AffineTransform(scale=(scale, scale),
                            rotation=rotation,
                            translation=translation)

    views = raw_image.shape[0]
    for i in range(views):
        raw_image[i] = warp(raw_image[i],
                            tform,
                            output_shape=raw_image[i].shape)
    return raw_image


def random_noise(raw_image):
    # noise = np.random.random(raw_image[0].shape)
    # import copy

    # new_img = copy.deepcopy(raw_image) + noise
    # return new_img
    from skimage.util import random_noise
    rand_std = np.random.uniform(0.01, 0.2)
    # rand_std = np.ones_like(raw_image) * 1e-3
    # return random_noise(raw_image, var=rand_std)
    return raw_image + rand_std


single_image = imgs[2]
print(single_image.shape)
import matplotlib.pyplot as plt

# i = 0
# while i < 3:
plt.ion()
fig = plt.figure("test")


def draw_img(fig, img, st):
    assert type(img) == np.ndarray
    assert len(img.shape) == 3
    assert img.shape[0] == 4
    for i in range(4):
        ax = fig.add_subplot(3, 4, st + i)
        ax.imshow(img[i])


def aug_cpu(all_imgs):
    from copy import deepcopy
    new_imgs = deepcopy(all_imgs)
    for i in range(len(new_imgs)):
        new_imgs[i] = affine(new_imgs[i])
        new_imgs[i] = random_noise(new_imgs[i])
    return new_imgs


import torch
from torchvision.transforms import RandomAffine, GaussianBlur
torch.manual_seed(0)
device = torch.device("cuda", 0)
# device = torch.device("cpu", 0)

affine = RandomAffine(degrees=(-5, 5), translate = (0.07, 0.07), scale = (0.9, 1.1), shear = None)
blur = GaussianBlur(kernel_size=3)
noise_gaussian_std = 0.04

def aug_torch(all_imgs):
    global noise_gaussian_std
    torch_all_imgs = torch.from_numpy(np.array(all_imgs))
    # height, width = all_imgs[0].shape[1], all_imgs[0].shape[2]

    torch_all_imgs = affine(torch_all_imgs)
    torch_all_imgs = blur(torch_all_imgs)

    noise = torch.randn_like(torch_all_imgs[0]) * noise_gaussian_std
    print(f"image mean {np.mean(np.abs(all_imgs))}")
    print(f"noise mean {torch.mean(torch.abs(noise))}")
    torch_all_imgs += noise 
    return np.array(torch_all_imgs)
    # print(torch_all_imgs.shape)
    # exit()



while True:
    # draw_img(fig, single_image, 1)
    # new_img =
    # diff = new_img - single_image
    draw_img(fig, single_image, 1)
    # draw_img(fig, affine(single_image), 5)

    import time
    st = time.time()
    # new_imgs = aug_cpu(imgs)
    new_imgs = aug_torch(imgs)
    # from copy import deepcopy
    # new_img = deepcopy(single_image)
    # # ed1 = time.time()
    # new_img = affine(new_img)
    # # ed2 = time.time()
    # new_img = random_noise(new_img)
    ed = time.time()
    print(f"cost {(ed - st ) * 1e3} ms")
    # exit()
    new_img = new_imgs[2]
    draw_img(fig, new_img, 5)
    draw_img(fig, new_img - single_image, 9)
    # draw_img(fig, single_image, 1)
    # ax1 = fig.add_subplot(3, 4, 1)
    # ax2 = fig.add_subplot(3, 4, 2)
    # ax3 = fig.add_subplot(3, 4, 3)
    # ax1.imshow(single_image[0])
    # new_img = affine(single_image[0])
    # new_img = random_noise(new_img)
    # ax2.imshow(new_img)
    # ax3.imshow(new_img - single_image[0])
    # plt.plot()
    # plt.show()
    plt.pause(0.1)