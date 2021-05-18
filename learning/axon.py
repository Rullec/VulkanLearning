import numpy as np
import video_manager
import os
from PIL import Image

global_xpos = None
global_ypos = None
cam = None


def resize(image):
    # height, width
    height, width = image.shape
    mid = width / 2
    assert width % 2 == 0
    # to a square
    image = image[:, int(mid - height / 2):int(mid + height / 2)].astype(float)
    # expand this square to
    from PIL import Image
    image = Image.fromarray(image)
    image = image.resize((128, 128))
    image = np.array(image)
    # print(image.shape)
    # print(np.max(image))
    # print(np.min(image))
    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()
    # exit(0)
    return image


def mouse_move(event):
    '''
        Mouse move callback function for matplotlib
    '''
    global global_xpos, global_ypos
    global_xpos, global_ypos = event.xdata, event.ydata

    if global_xpos is not None and global_ypos is not None:
        global_xpos = int(global_xpos)
        global_ypos = int(global_ypos)


def on_press(event):
    import sys
    print('press', event.key)
    sys.stdout.flush()


def get_depth_image_mm(cam):
    '''
        get the depth image (unit mm)
    '''
    depth = cam.GetDepthImage().astype(float) * cam.GetDepthUnit_mm()
    return depth


def get_ir_image(cam):
    '''
        get the ir image
    '''
    ir = cam.GetIrImage()
    return ir


def show_image(depth_image):
    import matplotlib.pyplot as plt
    plt.imshow(depth_image)
    plt.show()


def display(mode="depth"):
    global global_xpos, global_ypos
    import matplotlib.pyplot as plt
    # 打开交互模式
    plt.ion()
    fig1 = plt.figure('frame')

    plt.connect('motion_notify_event', mouse_move)

    plt.connect('key_press_event', on_press)
    # fig2 = plt.figure('subImg')
    while True:
        fig1.clf()
        ax1 = fig1.add_subplot(1, 1, 1)
        if mode == "depth":
            res = get_depth_image_mm(cam)
        elif mode == "ir":
            res = get_ir_image(cam)
        res = resize(res)
        # print(f"raw {res.dtype}")

        if global_xpos is not None and global_ypos is not None:
            print(
                f"\rdepth ({global_ypos}, {global_xpos}) = {int(res[global_ypos, global_xpos])} mm",
                end='')
        ax1.imshow(res)
        ax1.title.set_text("depth image (mm)")
        # ax1.plot(p1[:, 0], p1[:, 1], 'g.')
        plt.pause(3e-2)


def load_agent(agent_json_path):
    '''
        load the agent from given config file
    '''
    import shutil
    single_name = os.path.split(agent_json_path)[-1]
    import time
    single_name = str(time.time()) + single_name
    shutil.copy(agent_json_path, single_name)
    import json
    from train import build_net
    net_type = build_net(single_name)
    import torch
    net_device = torch.device("cuda", 0)
    net = net_type(single_name, device=net_device)
    os.remove(single_name)
    return net


def infer_net(net, depth_image):
    '''
        Given a net and a depth image, return the parameter
    '''
    depth_image = resize(depth_image)
    depth_image = np.expand_dims(depth_image, 0)
    param = net.infer(depth_image)
    print(f"get feature {param}")
    show_image(depth_image.squeeze())
    return param


def ir_camera_calibration():
    cam = video_manager.video_manager()

    import matplotlib.pyplot as plt
    import cv2
    from axon_calibrate import get_camera_pts_to_world_coord
    plt.ion()
    fig1 = plt.figure('frame')

    while True:
        # clear but do not close the figure
        fig1.clf()
        ax1 = fig1.add_subplot(1, 1, 1)
        image = get_ir_image(cam).astype(np.uint8)
        print(image.shape)
        ax1.imshow(image)
        # get_camera_pts_to_world_coord([image])

        # draw the image
        ax1.title.set_text("ir image")

        # pause
        plt.pause(3e-2)

    print(f"get ir succ, shape {image.shape}")


if __name__ == "__main__":
    ir_camera_calibration()
    # cam = video_manager.video_manager()
    # unit = cam.GetDepthUnit_mm()
    # print(f"cur unit {unit} mm")
    # depth = get_depth_image_mm(cam)
    # conf_path = "..\config\\train_configs\\conv_conf.json"
    # print(f"depth shape {depth.shape}")
    # # display()
    # net = load_agent(conf_path)
    # infer_net(net, depth)