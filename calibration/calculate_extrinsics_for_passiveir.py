import numpy as np
from opencv_calibration import Calibration
import cv2
from device_util import create_kinect_device, get_ir_image, get_mtx_and_dist_from_sdk
from drawer_util import DynaPlotter, cast_int32_to_uint8, resize

self_mtx = np.array([[502.1827161608854, 0.0, 511.0979713205807],
                     [0.0, 503.82251092282405, 508.8838972133547],
                     [0.0, 0.0, 1.0]])
self_dist = np.array([
    -0.3175459043404582, 0.11647982327752722, -0.0004945177774058406,
    -0.0004635829683942412, -0.021436060392820675
])

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    plot = DynaPlotter(1, 1, iterative_mode=True)
    calib = Calibration("calib_config.json")
    device = create_kinect_device()

    # 1. capture image
    # 2. input image into calibration (sdk params and self params)
    # 3. calculate extrinsics
    sdk_mtx, sdk_dist = get_mtx_and_dist_from_sdk(device)
    while plot.is_end == False:
        ir_image = cast_int32_to_uint8(get_ir_image(device))
        # self_rvecs, self_tvecs, self_image, self_error = calib.calc_extrinsics(
        #     ir_image, self_mtx, self_dist)
        self_cam_pos, self_cam_focus, camera_pts_to_world_coords = calib.calc_extrinsics_camera_parameter(
            ir_image, self_mtx, self_dist)

        # sdk_rvecs, sdk_tvecs, sdk_image, sdk_error = calib.calc_extrinsics(
        #     ir_image, sdk_mtx, sdk_dist)

        if self_cam_pos is not None:
            # and sdk_rvecs is not None:
            # print(
            #     f"self rvecs {np.transpose(self_rvecs) / np.pi * 180} sdk rvecs {np.transpose(sdk_rvecs)  / np.pi * 180}"
            # )
            import json

            print(
                f"self pos {json.dumps(list(self_cam_pos * 1e-3))} self focus {json.dumps(list(self_cam_focus * 1e-3) )}"
            )
            # print(f"self err {self_error} sdk err {sdk_error}")

        else:
            print(f"calibration failed")
        plot.add(ir_image)
        plot.show()