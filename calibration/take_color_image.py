from device_util import *
import time
from drawer_util import DynaPlotter, to_gray
from copy import deepcopy
from file_util import *
from opencv_calibration import Calibration

try_to_save = False
output_image_dir = "color_images.log/"
not_clear_and_create_dir(output_image_dir)


def is_try_to_save():
    global try_to_save
    return try_to_save


def set_try_to_save(value):
    global try_to_save
    try_to_save = value


def keyboard_callback(event):
    global try_to_save
    if event.key == "x":
        log_print(f"custom callback, keyboard press {event.key}")
        set_try_to_save(True)


def down_sample(img):
    if len(img.shape) == 3:
        img = img[0::2, 0::2, :]
    elif len(img.shape) == 2:
        img = img[0::2, 0::2]
    return img


if __name__ == "__main__":
    device = create_kinect_device(get_depth_mode_str())
    calib_util = Calibration("calib_config.json")

    plot = DynaPlotter(1, 2, iterative_mode=True)
    plot.set_keypress_callback(keyboard_callback)
    cur_iters = 0
    while plot.is_end is False:
        # st = time.time()
        color_img = get_color_image(device)[:, :, :3]
        # ed = time.time()
        # print(f"get color img {ed - st} s")
        # depth_img = get_depth_to_color_image(device)

        # new_img = deepcopy(color_img)

        gray = to_gray(color_img)
        # ed1 = time.time()
        # print(f"get to_gray img {ed1 - ed} s")

        # for i in range(new_img.shape[2]):
        #     new_img[:, :, i] += depth_img.astype(np.uint8)
        plot.add(color_img, "color img")
        plot.add(gray, "gray img")
        # ed2 = time.time()
        # print(f"add plot {ed2 - ed1} s")

        # 1. check if the image is recognizable
        if is_try_to_save() is True:
            if calib_util.judge_image_recognizable([gray]) == True:
                print(f"current image is recognizable")
                save_pkl(f"color_images.log/{cur_iters}.pkl", gray)
                cur_iters += 1
            else:
                print(f"current image is not recognizable")
            set_try_to_save(False)
        # ed3 = time.time()
        # print(f"judge image recog {ed3 - ed2} s")
        plot.show()
        import time
        time.sleep(0.01)
