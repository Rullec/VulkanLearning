## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
from scipy import signal

global_xpos = 0
global_ypos = 0


def resize(raw_image):
    # height, width
    height, width = raw_image.shape
    mid = width / 2
    assert width % 2 == 0
    raw_image = raw_image[:,
                          int(mid - height / 2):int(mid + height / 2)].astype(
                              np.float)
    # print(np.max(raw_image))
    # print(np.min(raw_image))
    # exit(0)
    print(f"raw max {np.max(raw_image)} min {np.min(raw_image)}")
    image = Image.fromarray(raw_image)
    # image = signal.resample(raw_image, 512, 512)
    image = image.resize((512, 512))
    image = np.array(image)
    # print(image.shape)
    import matplotlib.pyplot as plt
    plt.imshow(image)
    return image
    # print(f"new max {np.max(image)} min {np.min(image)}")
    # plt.show()


def mouse_move_callback(event, x, y, flags, param):
    global global_xpos, global_ypos
    if event == cv2.EVENT_MOUSEMOVE:
        # print(f"mouse move to {x} {y}")
        global_xpos = x
        global_ypos = y


if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    def calc_distance(x, y):
        align_to = rs.stream.color
        align = rs.align(align_to)
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        # 深度参数，像素坐标系转相机坐标系用到
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
        ).intrinsics
        color_frame = aligned_frames.get_color_frame()

        # 深度图
        d = np.asanyarray(aligned_depth_frame.get_data())
        # 彩色图
        image_np = np.asanyarray(color_frame.get_data())
        # 输入像素的x和y计算真实距离
        dis = aligned_depth_frame.get_distance(x, y)
        print(f"{x} {y} distance = {dis}m")

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # print(
            #     f"depth image max {np.max(depth_image)} min {np.min(depth_image)}"
            # )
            # shape = depth_image.shape
            # depth_image = resize(depth_image)
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', depth_image)
            # cv2.waitKey(1)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            # print(f"depth dim {depth_colormap_dim}")
            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image,
                                                 dsize=(depth_colormap_dim[1],
                                                        depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                # images = np.hstack((resized_color_image, depth_colormap))
                images = depth_colormap
            else:
                # images = np.hstack((color_image, depth_colormap))
                images = depth_colormap

            # Show images
            calc_distance(global_xpos, global_ypos)
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

            cv2.setMouseCallback('RealSense', mouse_move_callback)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)
    finally:

        # Stop streaming
        pipeline.stop()