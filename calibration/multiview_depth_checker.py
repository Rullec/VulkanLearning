from matplotlib.pyplot import plot
import numpy as np
import os
from file_util import load_pkl
from drawer_util import DynaPlotter

if __name__ == "__main__":
    # output_dir = "cutted_dir.log"
    output_dir = "cutted_dir.log"
    files = [os.path.join(output_dir, i) for i in os.listdir(output_dir)]

    # 1. begin to load the pkl files
    depth_image_lst = [load_pkl(pkl) for pkl in files]

    for _idx, depth_image in enumerate(depth_image_lst):
        ploter = DynaPlotter(1, len(depth_image), iterative_mode=False)
        for j in depth_image:
            ploter.add(j)
        ploter.show()