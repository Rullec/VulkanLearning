import numpy as np
import os
from opencv_calibration import Calibration
from device_util import create_kinect_device, get_mtx_and_dist_from_sdk
from file_util import load_pkl, convert_nparray_to_json
from copy import deepcopy
from drawer_util import DynaPlotter, calculate_subplot_size
import cv2


def undistort(mtx, dist, raw_image):
    image = deepcopy(raw_image)
    return cv2.undistort(image, mtx, dist)


def compare_intrinsics_sdk_and_self(self_mtx, self_dist, sdk_mtx, sdk_dist,
                                    images):
    num_of_images = len(images)
    print(f"num_of_images {num_of_images}")
    rows, cols = calculate_subplot_size(3 * num_of_images)
    plot = DynaPlotter(rows, cols, iterative_mode=False)

    for i in range(num_of_images):
        self_image = undistort(self_mtx, self_dist, images[i])
        sdk_image = undistort(sdk_mtx, sdk_dist, images[i])
        plot.add(images[i], f"raw image {i}")
        plot.add(self_image, f"self image {i}")
        plot.add(sdk_image, f"sdk image {i}")
    plot.show()


def show_undistort_result(mtx, dist, images):
    num_of_images = len(images)
    print(f"num_of_images {num_of_images}")
    rows, cols = calculate_subplot_size(2 * num_of_images)
    plot = DynaPlotter(rows, cols, iterative_mode=False)

    for i in range(num_of_images):
        image = undistort(mtx, dist, images[i])
        plot.add(images[i], f"raw image {i}")
        plot.add(image, f"undisroted image {i}")
    plot.show()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    output_image_dir = "passive_ir_images.log/"

    # 1. loading images from the directory
    def load_ir_images_from_directory():
        files = os.listdir(output_image_dir)
        files = [os.path.join(output_image_dir, i) for i in files]
        img_lst = [load_pkl(filepath) for filepath in files]
        return img_lst

    # 2. do calibration by calling APIs
    all_images = load_ir_images_from_directory()
    calib = Calibration("calib_config.json")
    ret, mtx, dist, rvecs, tvecs = calib.calc_intrinsics_from_images(
        all_images)

    print(f"self mtx json\n{convert_nparray_to_json(mtx)}")
    print(f"self dist json \n{convert_nparray_to_json(dist)}")
    show_undistort_result(mtx, dist, all_images[:5])
    # device = create_kinect_device()
    # sdk_mtx, sdk_dist = get_mtx_and_dist_from_sdk(device)
    # print(f"sdk mtx {sdk_mtx}")
    # print(f"sdk dist {sdk_dist}")
    # compare_intrinsics_sdk_and_self(mtx, dist, sdk_mtx, sdk_dist, all_images[:3])
    # compare the result
    # get_mtx_and_dist_from_sdk()
    # print(f"rvecs {rvecs}")
    # print(f"tvecs {tvecs}")