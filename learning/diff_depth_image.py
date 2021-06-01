import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import device_manager
from axon import get_depth_image_mm, resize, get_mtx_and_dist_sdk
import process_data_scene
import cv2

def load_capture_depth_image(cam):
    # capture depth image from the camera
    # if os.path.exists("tmp.pkl") is False:
    #     #
    
    #     with open("tmp.pkl", 'wb') as f:
    #         pickle.dump(img, f)
    # else:
    #     with open("tmp.pkl", 'rb') as f:
    #         img = pickle.load(f)
    camera_matrix, dist_coef = get_mtx_and_dist_sdk()
    img = get_depth_image_mm(cam)
    # print(f"old img {img}")
    img -= 20
    # print(f"new img {img}")
    # exit()
    img = cv2.undistort(
            img, camera_matrix, dist_coef, None, None
        )
    img *= 1e-3
    img = resize(img, 512)
    return img


def load_cast_depth_image(scene):

    shape = scene.GetDepthImageShape()

    # pos = np.array([0.00187508856, 0.42842519843, 0.55907583299, 1])
    # pos = np.array([-9.98005445, 302.31570635, 501.97957573, 1]) * 1e-3
    # center = np.array([-4.86800079, 160.0555942, 0, 1]) * 1e-3
    # pos = np.array([-8.40875573, 340.5975439, 660.68147564, 1]) * 1e-3
    # center = np.array([-10.1599905, 151.46500068, 0, 1]) * 1e-3
    # pos = np.array([-6.15960061, 302.56721469, 378.33105492, 1]) * 1e-3
    # center = np.array([-6.45289283, 153.23874287, 0, 1]) * 1e-3
    pos = np.array([0, 119, 516, 1]) * 1e-3
    center = np.array([0, 119, 0, 1]) * 1e-3
    pos[3], center[3] = 1, 1

    # fov = 49.2
    fov = 49.2
    img = scene.CalcEmptyDepthImage(pos, center, fov)
    print(f"shape {shape}")
    print(f"img shape {img.shape}")
    img = img.reshape(shape)
    print(f"img shape {img.shape}")
    return img
    # path = r"D:\SimpleClothSimulator\data\export_data\test_geodata_gen\0.png"
    # from PIL import Image
    # image = np.array(Image.open(path), dtype=np.float32)
    # image = np.mean(image, axis=2)
    # image /= 200 # convert to m
    # image *= 1000 # convert to mm

    # # read the value, divide it by 200
    # # print(image.shape)
    # return image


config_path = "./config/data_process.json"
cam = device_manager.kinect_manager()
scene = process_data_scene.process_data_scene()
scene.Init(config_path)
casted_img = load_cast_depth_image(scene)

import matplotlib.pyplot as plt

plt.ion()
fig1 = plt.figure('frame')

while True:
    # clear but do not close the figure
    fig1.clf()
    ax1 = fig1.add_subplot(1, 4, 1)
    captured_img = load_capture_depth_image(cam)

    kernel = np.ones((5, 5), dtype=np.float32)
    import cv2
    dilated_cap_img = cv2.dilate(captured_img, kernel)
    ax1.imshow(captured_img)
    ax1.title.set_text("captured")

    ax3 = fig1.add_subplot(1, 4, 2)
    ax3.imshow(dilated_cap_img)
    ax3.title.set_text("dilated")
    

    ax2 = fig1.add_subplot(1, 4, 3)

    ax2.imshow(casted_img)
    ax2.title.set_text("casted")

    ax3 = fig1.add_subplot(1, 4, 4)
    # diff = np.abs(captured_img - casted_img)
    diff = np.abs(dilated_cap_img - casted_img)
    ax3.imshow(diff)
    ax3.title.set_text("diff")

    # draw the image
    # ax1.imshow(res)
    # ax1.title.set_text("depth image (mm)")

    # pause
    plt.pause(3e-2)

# capture_depth_image = load_capture_depth_image(cam)
# cast_depth_image = load_cast_depth_image()

# print(f"capture_depth_image shape {capture_depth_image.shape}")
# print(f"cast_depth_image shape {cast_depth_image.shape}")
# plt.subplot(1, 3, 1)
# plt.imshow(capture_depth_image)
# plt.title("capture_depth_image")
# plt.subplot(1, 3, 2)
# plt.imshow(cast_depth_image)
# plt.title("cast_depth_image")
# # plt.subplot(1, 3, 3)
# # plt.imshow(cast_depth_image - capture_depth_image)
# # plt.title("diff")
# plt.show()
