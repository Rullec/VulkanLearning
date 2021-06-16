from file_util import load_pkl
from drawer_util import DynaPlotter, cast_int32_to_uint8, calculate_subplot_size
from device_util import *
from file_util import save_pkl, not_clear_and_create_dir
from opencv_calibration import Calibration
import os


def init_device():
    '''
        create the device 
    '''
    cam = create_kinect_device()
    set_device_mode(cam, get_passive_ir_mode_str())
    return cam


try_to_save = False
output_image_dir = "passive_ir_images.log/"
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


def get_beginning_id():
    global output_image_dir
    files = os.listdir(output_image_dir)
    return len(files)


def capture_passive_ir_images():
    # 1. create passive ir device
    cam = init_device()
    calib_util = Calibration("calib_config.json")

    plot = DynaPlotter(1, 1, "passive_ir_view")
    plot.set_keypress_callback(keyboard_callback)
    succ_iters = get_beginning_id()
    while plot.is_end == False:
        # 1. take the ir image
        ir_image = get_ir_image(cam)
        ir_image = cast_int32_to_uint8(ir_image)

        if is_try_to_save() == True:
            if calib_util.judge_image_recognizable([ir_image]) == True:
                save_pkl(f"{output_image_dir}/{succ_iters}.pkl", ir_image)
                succ_iters += 1
            else:
                print(f"the input ir image is not valid, ignore save")
            set_try_to_save(False)

        # 2. display the ir image
        plot.add(ir_image)
        plot.show()


def visualize_passive_ir_images():
    files = os.listdir(output_image_dir)
    num_of_files = len(files)
    print(num_of_files)
    rows, cols = calculate_subplot_size(num_of_files)
    plot = DynaPlotter(rows, cols, "visualize_passive_ir", False)
    for _idx, file in enumerate(files):
        fullname = os.path.join(output_image_dir, file)
        image = load_pkl(fullname)
        plot.add(image, str(_idx))
    plot.show()


if __name__ == "__main__":
    # visualize_passive_ir_images()
    capture_passive_ir_images()