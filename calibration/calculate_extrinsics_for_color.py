import numpy as np
from drawer_util import DynaPlotter, to_gray
from device_util import *
from opencv_calibration import Calibration
import json

self_mtx = np.array([[624.9886883433672, 0.0, 637.0424747071835],
                     [0.0, 623.8646982605328, 366.4002950671636],
                     [0.0, 0.0, 1.0]])
self_dist = np.array([
    0.10973763151657619, -0.10491086469551321, 0.0006643112931425941,
    -0.00019728122828219883, 0.056559958189034504
])

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    plot = DynaPlotter(1, 2, iterative_mode=True)
    calib = Calibration("calib_config.json")
    calib.set_debug_mode(False)
    device = create_kinect_device()

    # 1. capture image
    # 2. input image into calibration (sdk params and self params)
    # 3. calculate extrinsics
    sdk_mtx, sdk_dist = get_mtx_and_dist_from_sdk(device)
    # self_cam_pos = None
    self_rvecs = None
    while plot.is_end == False:
        color_image = to_gray(get_color_image(device))
        try:
            self_rvecs, self_tvecs, self_image, self_error = calib.calc_extrinsics(
                color_image, self_mtx, self_dist)
            camera_pts_to_world_coords = calib.convert_rtvecs_to_transform(
                self_rvecs, self_tvecs)
            cam_pos, cam_focus = calib.convert_transform_to_campos_and_camfocus(
                camera_pts_to_world_coords)
            print(f"rvecs = {self_rvecs / np.pi * 180}")
            print(
                f"camera_pts_to_world_coords = \n{camera_pts_to_world_coords}")
            print(
                f"self pos {json.dumps(list(cam_pos * 1e-3))} self focus {json.dumps(list(cam_focus * 1e-3))}"
            )
            plot.add(self_image, "image")

            # self_cam_pos, self_cam_focus, camera_pts_to_world_coords = calib.calc_extrinsics_camera_parameter(
            #     color_image, self_mtx, self_dist)

        # sdk_rvecs, sdk_tvecs, sdk_image, sdk_error = calib.calc_extrinsics(
        #     color_image, sdk_mtx, sdk_dist)
        except Exception as e:
            print(f"exception: {e}")
        if self_rvecs is not None:
            # if self_cam_pos is not None:
            # and sdk_rvecs is not None:
            # print(f"self rvecs {np.transpose(self_rvecs) / np.pi * 180}")
            pass
            # print(f"rvecs {self_rvecs / np.pi * 180}, tvecs {self_tvecs}")
            # plot.add(self_image, "self image")

            # print(f"self err {self_error} sdk err {sdk_error}")

        else:
            print(f"calibration failed")
        plot.add(color_image, "color image")

        plot.show()