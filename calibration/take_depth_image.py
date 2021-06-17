import numpy as np
from drawer_util import DynaPlotter
from device_util import get_depth_image, create_kinect_device, get_depth_mode_str
'''
    current pos and focus pos:
        self pos [-23.40066073 360.10044023 399.44058796] self focus [-25.84474082 197.29701774   0.]

    calculate the depth image and do comparision
        
'''

if __name__ == "__main__":
    cam = create_kinect_device(mode=get_depth_mode_str())
    plot = DynaPlotter(1, 1, iterative_mode=True)

    while plot.is_end is False:
        depth_image = get_depth_image(cam)
        plot.add(depth_image)
        plot.show()