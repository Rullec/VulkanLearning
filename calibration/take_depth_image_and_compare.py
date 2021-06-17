import numpy as np
from drawer_util import DynaPlotter
from device_util import get_depth_image, create_kinect_device, get_depth_mode_str
import process_data_scene
'''
    current pos and focus pos:
        self pos  self focus []

    calculate the depth image and do comparision
        
'''
real_cam_pos = np.array([-23.40066073, 360.10044023, 399.44058796, 1]) * 1e-3
real_cam_focus = np.array([-25.84474082, 197.29701774, 0., 1]) * 1e-3
real_cam_fov = 61.92
real_cam_focus[3] = 1
real_cam_focus[3] = 1

def cast_depth_image(scene):
    global real_cam_pos, real_cam_focus, real_cam_fov
    shape = scene.GetDepthImageShape()
    raycast_depth_image = scene.CalcEmptyDepthImage(real_cam_pos,
                                                    real_cam_focus,
                                                    real_cam_fov)

    raycast_depth_image = np.ascontiguousarray(
        raycast_depth_image.reshape(shape))
    return raycast_depth_image


if __name__ == "__main__":
    # cam = create_kinect_device(mode=get_depth_mode_str())
    plot = DynaPlotter(1, 1, iterative_mode=False)

    config_path = "./config/data_process.json"
    scene = process_data_scene.process_data_scene()
    scene.Init(config_path)
    depth_image =  cast_depth_image(scene)
    plot.add(depth_image)
    plot.show()
    # while plot.is_end is False:
        # depth_image = get_depth_image(cam)
        # plot.add(depth_image)
        # plot.show()