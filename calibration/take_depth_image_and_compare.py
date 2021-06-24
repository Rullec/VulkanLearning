from device_util import get_depth_to_color_image
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


def get_calibrated_camera_extrinsics_old():

    real_cam_pos = np.array(
        [0.01760628071078309, 0.4716434966730053, 0.40155680889293943, 1])
    real_cam_focus = np.array(
        [0.01124955884155198, 0.22071030716450926, 0.0, 1])

    real_cam_fov = 61.92
    return real_cam_pos, real_cam_focus, real_cam_fov


def mask_depth_image_by_given_window(scene, old_full_image):
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


def cut_depth_image_by_given_window(scene, old_full_image):
    enable_cutted = scene.GetEnableOnlyExportingCuttedWindow()
    cutted_window = scene.GetCuttedWindow()
    resolution = scene.GetResolution()
    # print(f"depth image resolution {old_full_image.shape}")
    # print(f"resolution {resolution}")
    # print(f"cutted_window {cutted_window}")
    # exit()
    if enable_cutted == True:
        old_full_image = old_full_image[cutted_window[1, 0]:cutted_window[1,
                                                                          1],
                                        cutted_window[0, 0]:cutted_window[0,
                                                                          1]]
    return old_full_image


def calculate_raycast_result(scene, cam_pos, cam_focus, cam_fov):
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

    new_image = mask_depth_image_by_given_window(scene, new_image)
    return new_image


def get_scene():
    config_path = "./config/data_process.json"
    scene = process_data_scene.process_data_scene()
    scene.Init(config_path)
    return scene


def get_calibrated_camera_extrinsics_new():

    # rgb camera ori & pos
    real_cam_pos = np.array(
        [0.049792031956888125, 0.4808656785922257, 0.4155189073952698, 1])
    real_cam_focus = np.array(
        [0.027227736992100748, 0.24315683743796296, 0.0, 1])
    real_cam_fov = 58

    return real_cam_pos, real_cam_focus, real_cam_fov


def old_nfov_unbinned_comp():
    # 0. create depth image
    cam = create_kinect_device(mode=get_depth_mode_str())

    scene = get_scene()
    # 1. get calibrated camera extrinsics
    real_cam_pos, real_cam_focus, real_cam_fov = get_calibrated_camera_extrinsics_old(
    )

    # 2. calcualte the raycast result
    raycast_depth_image = calculate_raycast_result(
        scene, real_cam_pos, real_cam_focus, real_cam_fov) * 1e3

    # 4. do display
    plot = DynaPlotter(1, 4, iterative_mode=True)

    while plot.is_end == False:
        # 3. get the depth image from camera
        real_depth_image_distort = get_depth_image(cam)
        real_depth_image_undistort = undistort_depth_image_by_sdk_intrinsics(
            scene, cam, real_depth_image_distort) * 1e3
        plot.add(raycast_depth_image, f"raycast")
        plot.add(real_depth_image_undistort, "undistort")
        plot.add(real_depth_image_distort, "distort")
        plot.add(raycast_depth_image - real_depth_image_undistort,
                 "raycast - undistort")
        plot.show()


def new_depth_to_color_comp():
    # 1. create depth to color image
    cam = create_kinect_device(mode=get_depth_mode_str())
    scene = get_scene()

    # 2. get extrinsics
    real_cam_pos, real_cam_focus, real_cam_fov = get_calibrated_camera_extrinsics_new(
    )

    raycast_depth_image = calculate_raycast_result(
        scene, real_cam_pos, real_cam_focus, real_cam_fov) * 1e3
    plot = DynaPlotter(1, 3, iterative_mode=True)

    while plot.is_end == False:
        depth_to_color_image = get_depth_to_color_image(cam)
        print(f"depth image shape {depth_to_color_image.shape}")
        plot.add(depth_to_color_image, f"reals depth image")
        plot.add(raycast_depth_image, f"raycasted result")
        plot.add(depth_to_color_image - raycast_depth_image, f"real - raycast")
        plot.show()

    pass


if __name__ == "__main__":
    # old_nfov_unbinned_comp()
    new_depth_to_color_comp()