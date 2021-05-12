## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import PIL
import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
from scipy import signal
import os

global_xpos = 0
global_ypos = 0


def calc_depth_map(scale):
    align_to = rs.stream.color
    align = rs.align(align_to)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    real_depth = np.asanyarray(aligned_depth_frame.get_data()).astype(
        np.float) * scale
    return real_depth

    # scale = rs.get_depth_scale()
    # print(scale)
    # exit(0)
    # print(np_array.shape)
    # exit(0)
    # aligned_depth_frame.get_distance
    # dis = aligned_depth_frame.get_distance(x, y)


def resize(image):
    # height, width
    height, width = image.shape
    mid = width / 2
    assert width % 2 == 0
    # to a square
    image = image[:,
                  int(mid - height / 2):int(mid + height / 2)].astype(np.float)
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


def mouse_move_callback(event, x, y, flags, param):
    global global_xpos, global_ypos
    if event == cv2.EVENT_MOUSEMOVE:
        # print(f"mouse move to {x} {y}")
        global_xpos = x
        global_ypos = y


def save_depth_image(image, save_path):
    Image.fromarray(image).save(save_path)
    print(f"[log] save depth image to {save_path}")
    return


def inference(agent_json_path, depth_image, pkl_path=""):
    
    # 1. create a temp config json, load the agent
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
    depth_image = resize(depth_image)
    depth_image = np.expand_dims(depth_image, 0)
    # 2. inference
    # depth_image = torch.Tensor(depth_image).to(net_device)
    return net.infer(depth_image)


if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    depth_scale = pipeline_profile.get_device().first_depth_sensor(
    ).get_depth_scale()

    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    finished = False
    save_dir = "log/depths"
    cur_save_id = 0
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    try:
        while not finished:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            # depth_image = np.asanyarray(depth_frame.get_data())
            # color_image = np.asanyarray(color_frame.get_data())

            # depth_image = resize(depth_image)
            depth_image = (calc_depth_map(depth_scale) * 400).astype(np.uint8)
            conf_path = "..\config\\train_configs\\conv_conf.json"
            param = inference(conf_path, depth_image)
            print(f"output {param}")
            exit()
            # print(depth_image.dtype)
            # exit()
            # print(depth_image.shape)
            if global_xpos < depth_image.shape[
                    1] and global_ypos < depth_image.shape[0]:
                print(
                    f"depth[{global_xpos}, {global_ypos}] = {depth_image[global_ypos, global_xpos]}"
                )
            else:
                print(f"exceed {global_xpos} {global_ypos}")
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

            cv2.setMouseCallback('RealSense', mouse_move_callback)
            cv2.imshow('RealSense', depth_image)
            key = cv2.waitKey(1)
            if key == 27:
                finished = True
            elif key == ord('s'):
                fullname = os.path.join(save_dir, f"{cur_save_id}.png")
                save_depth_image(depth_image, fullname)
    finally:

        # Stop streaming
        pipeline.stop()