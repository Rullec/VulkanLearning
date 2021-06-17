'''
sample a group of depth images as a data point in network
'''
from drawer_util import DynaPlotter
from device_util import get_depth_image, create_kinect_device, get_depth_mode_str
import numpy as np
from file_util import save_pkl, clear_and_create_dir

need_to_save_ir_image = False


def keyboard_callback(event):
    global need_to_save_ir_image
    if event.key == "w":
        need_to_save_ir_image = True


if __name__ == "__main__":
    # all counters
    num_of_views = 4
    num_of_datapoints = 4
    output_dir = "./current_dir.log/"
    clear_and_create_dir(output_dir)

    # 1. init device
    device = create_kinect_device(mode=get_depth_mode_str())

    plotter = DynaPlotter(1, 1)
    plotter.set_keypress_callback(keyboard_callback)
    plotter.set_supresstitle("multiview depth sampler")

    img_lst = []
    cur_datapoint_idx = 0
    while cur_datapoint_idx < num_of_datapoints:
        # 1. get depth image
        depth_image = get_depth_image(device)

        # 2. show the depth image
        plotter.add(depth_image)
        plotter.show()

        # 3. detect the keyboard event
        if need_to_save_ir_image == True:
            img_lst.append(depth_image)
            print(f"add img list {len(img_lst)}")
            need_to_save_ir_image = False

        if len(img_lst) >= num_of_views:
            save_file = f"{output_dir}/{cur_datapoint_idx}.pkl"
            save_pkl(save_file, img_lst)
            img_lst = []
            cur_datapoint_idx += 1
            print(
                f"current datapoint save to {save_file}, current samples {cur_datapoint_idx}"
            )