from train import build_net
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import os
from PIL import Image


def downsample(img):
    new_shape = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    return np.array(Image.fromarray(img).resize(new_shape))


def set_random_seed():
    np.random.seed(0)
    torch.manual_seed(0)


def load_agent():
    is_cuda_avaliable = torch.cuda.is_available()
    assert is_cuda_avaliable == True

    if is_cuda_avaliable == True:
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu", 0)
    # net = ParamNet(conf_path, device)

    conf_path = "..\config\\train_configs\\conv_conf.json"
    # conf_path = "..\config\\train_configs\\fc_conf.json"
    net_type = build_net(conf_path)
    net = net_type(conf_path, device, only_load_statistic_data=True)
    return net


def load_depth_image():
    target_dir = "../data/export_data/captured_depth_images"
    if os.path.exists(target_dir) is False:
        print(f"[error] target dir {target_dir} doesn't exist")
        exit()
    file_lst = [os.path.join(target_dir, i) for i in os.listdir(target_dir)]
    img_lst = []
    for file in file_lst:
        import pickle
        with open(file, 'rb') as f:
            img_lst.append(pickle.load(f))
    return img_lst


def load_example_depth_image():
    exp = "D:\\SimpleClothSimulator\\data\\export_data\\test_geodata_gen\\0_0_0.png"
    from PIL import Image
    return np.array(Image.open(exp))


def resize_real_image(real_image):
    shape = real_image.shape
    print(f"real raw {shape}")
    real_image = real_image[50:275, 160:480]
    real_image = downsample(real_image)
    real_image = (real_image * 255.99).astype(np.int32)
    return real_image


def find_hole_mask(raw_depth_image):
    '''
        canny edge, 
    '''
    holes = raw_depth_image < 100
    # raw_depth_image[raw_depth_image < 0 ] =0
    holes = holes.astype(np.uint8)
    image = raw_depth_image.astype(np.uint8)
    # edges = cv2.Canny(image, 100, 200)
    # ax1 = plt.subplot(1, 3, 1)
    # ax1.imshow(raw_depth_image)
    # ax1.title.set_text("raw_depth_image")


    # ax2 = plt.subplot(1, 3, 2)    
    # ax2.imshow(holes)
    # ax2.title.set_text("holes")

    new_image = cv2.inpaint(image, holes, 1, cv2.INPAINT_TELEA)
    return new_image
    # ax3 = plt.subplot(1, 3, 3)    
    # ax3.imshow(new_image)
    # ax3.title.set_text("new_image")
    # plt.show()
    # ax2 = plt.subplot(1, 3, 2)
    # ax2.imshow(edges)
    # ax2.title.set_text("edges")

    # kernel = np.ones((3, 3))
    # filled = cv2.erode( cv2.dilate(edges, kernel), kernel)

    # ax33 = plt.subplot(1, 3, 3)
    # ax33.imshow(filled)
    # ax33.title.set_text("filled")
    # plt.show()


# def fill_depth_hole(depth_image):
#     threshold = 90
#     import copy
#     new_image = copy.deepcopy(depth_image)
#     import cv2
#     cv2.inpaint(depth_image, )
# for x_idx in range(depth_image.shape[0]):
#     for y_idx in range(depth_image.shape[1]):
#         cur_pixel = depth_image[x_idx, y_idx]

#         # there is a hole
#         if cur_pixel < threshold:

if __name__ == "__main__":
    set_random_seed()
    # 1. load depth image
    real_img = load_depth_image()[0]
    exp_img = load_example_depth_image()
    real_img = resize_real_image(real_img)
    # print(f"real img {real_img.shape} exp img {exp_img.shape}")
    real_img = find_hole_mask(real_img)
    # 3. infer
    agent = load_agent()
    input_img = np.expand_dims(real_img, 0)
    input_img = np.expand_dims(input_img, 0)
    res = agent.infer(input_img)
    print(f"inferred param {res}")

    # 2. display
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(real_img)
    ax1.title.set_text("captured img")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(exp_img)
    ax2.title.set_text("casted img")

    diff = np.abs(exp_img - real_img)
    ax3 = plt.subplot(1, 3, 3)
    ax3.title.set_text("diff")
    ax3.imshow(diff)
    # plt.suptitle(f"inferred param {res}")
    plt.show()

    # 3.

    net.train(max_epochs=10000)
    net.test()