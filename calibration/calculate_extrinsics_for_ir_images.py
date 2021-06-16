import numpy as np
import os
from opencv_calibration import Calibration
from device_util import create_kinect_device, get_mtx_and_dist_from_sdk
from file_util import load_pkl, convert_nparray_to_json
from copy import deepcopy
from drawer_util import DynaPlotter, calculate_subplot_size
import cv2

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    output_image_dir = "passive_ir_images.log/"
    calib = Calibration("calib_config.json")

    # 1. loading images from the directory
    def load_ir_images_from_directory():
        files = os.listdir(output_image_dir)
        files = [os.path.join(output_image_dir, i) for i in files]
        img_lst = [load_pkl(filepath) for filepath in files]
        return img_lst

    all_images = load_ir_images_from_directory()
    calib = Calibration("calib_config.json")
    ret, mtx, dist, rvecs, tvecs = calib.calc_intrinsics_from_images(
        all_images)

    
    for image in all_images:
        rvecs, tvecs, plotted_image, reproj_error = calib.calc_extrinsics(
            image, mtx, dist)
        print(f"rvecs {rvecs.T / np.pi * 180}")
        print(f"tvecs {tvecs.T}")
        print(f"reproj error {reproj_error}")
        plot = DynaPlotter(1, 1, iterative_mode=False)
        plot.add(plotted_image)
        plot.show()