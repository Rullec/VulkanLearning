from device_util import *
import time
from drawer_util import DynaPlotter, to_gray
from copy import deepcopy
from file_util import *

try_to_save = False
output_dir = "color_depth_stream"
not_clear_and_create_dir(output_dir)


def is_try_to_save():
    global try_to_save
    return try_to_save


def set_try_to_save(value):
    global try_to_save
    try_to_save = value


def keyboard_callback(event):
    global try_to_save
    if event.key == "x":
        log_print("set to save")
        set_try_to_save(True)


if __name__ == "__main__":
    device = create_kinect_device(get_depth_mode_str())

    plot = DynaPlotter(1, 2, iterative_mode=True)
    plot.set_keypress_callback(keyboard_callback)
    cur_iters = 0

    while plot.is_end is False:
        color_img = get_color_image(device)[:, :, :3]
        depth_img = get_depth_to_color_image(device)
        gray = to_gray(color_img)
        plot.add(color_img, "color img")
        plot.add(depth_img, "depth img")

        color_path = f"color_{cur_iters}.pkl"
        depth_path = f"depth_{cur_iters}.pkl"
        save_pkl(color_path, color_img)
        save_pkl(depth_path, depth_img)
        print(f"save {color_path} {depth_path}")
        # if try_to_save == True:
        #     print(f"try to save {try_to_save}")
        # color_png_name = "color.png"
        # depth_png_name = "depth.png"
        # print(depth_img.shape)
        # print(depth_img.dtype)
        # exit()
        # save_png_image(color_png_name, color_img)
        # save_png_image(depth_png_name, depth_img)
        #     try_to_save = False

        plot.show()
        cur_iters += 1
