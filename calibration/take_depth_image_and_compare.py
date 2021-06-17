import numpy as np
from drawer_util import DynaPlotter
from device_util import get_depth_image, create_kinect_device, get_depth_mode_str, get_mtx_and_dist_from_sdk
import process_data_scene
from copy import deepcopy
'''
    current pos and focus pos:
        self pos  self focus []

    calculate the depth image and do comparision
        
'''


def get_calibrated_camera_extrinsics():

    real_cam_pos = np.array([2.1079, 468.7629, 350.01404, 1]) * 1e-3
    real_cam_focus = np.array([3.2936509265447738, 262.06643546591494, 0.0, 1
                               ]) * 1e-3
    real_cam_fov = 61.92
    
    real_cam_focus[3] = 1
    real_cam_focus[3] = 1
    return real_cam_pos, real_cam_focus, real_cam_fov


def cut_depth_image_by_given_window(scene, old_full_image):
    enable_cutted = scene.GetEnableOnlyExportingCuttedWindow()
    cutted_window = scene.GetCuttedWindow()
    resolution = scene.GetResolution()
    # print(f"depth image resolution {old_full_image.shape}")
    # print(f"resolution {resolution}")
    # print(f"cutted_window {cutted_window}")
    # exit()
    if enable_cutted == True:
        
        old_full_image[0:cutted_window[1, 0], :] = 0
        old_full_image[cutted_window[1, 1]:, :] = 0

        old_full_image[:, 0:cutted_window[0, 0]] = 0
        old_full_image[:, cutted_window[0, 1]:] = 0
    return old_full_image


def calculate_raycast_result(cam_pos, cam_focus, cam_fov):

    shape = scene.GetDepthImageShape()
    raycast_depth_image = scene.CalcEmptyDepthImage(cam_pos, cam_focus,
                                                    cam_fov)
    raycast_depth_image = np.ascontiguousarray(
        raycast_depth_image.reshape(shape))

    return raycast_depth_image


import cv2


def undistort_depth_image_by_sdk_intrinsics(scene, device, old_image):
    mtx, dist = get_mtx_and_dist_from_sdk(device)
    new_image = deepcopy(old_image)
    # new_image = cast_int32_to_uint8(new_image)
    new_image = new_image.astype(np.float32) * 1e-3
    # print(new_image.dtype)
    # exit()
    new_image = cv2.undistort(new_image, mtx, dist)

    new_image = cut_depth_image_by_given_window(scene, new_image)
    return new_image


def get_scene():
    config_path = "./config/data_process.json"
    scene = process_data_scene.process_data_scene()
    scene.Init(config_path)
    return scene


if __name__ == "__main__":
    # 0. create depth image
    cam = create_kinect_device(mode=get_depth_mode_str())

    scene = get_scene()
    # 1. get calibrated camera extrinsics
    real_cam_pos, real_cam_focus, real_cam_fov = get_calibrated_camera_extrinsics(
    )

    # 2. calcualte the raycast result
    raycast_depth_image = calculate_raycast_result(real_cam_pos,
                                                   real_cam_focus,
                                                   real_cam_fov)

    # 4. do display
    plot = DynaPlotter(1, 3, iterative_mode=True)

    while plot.is_end == False:
        # 3. get the depth image from camera
        real_depth_image_distort = get_depth_image(cam)
        real_depth_image_undistort = undistort_depth_image_by_sdk_intrinsics(
            scene, cam, real_depth_image_distort)

        plot.add(raycast_depth_image)
        plot.add(real_depth_image_undistort, "undistort")
        plot.add(real_depth_image_distort, "distort")
        plot.show()