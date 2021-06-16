from drawer_util import DynaPlotter
from device_util import *
from file_util import save_pkl, clear_and_create_dir


def init_device():
    '''
        create the device 
    '''
    cam = create_kinect_device()
    set_device_mode(cam, get_passive_ir_mode_str())
    return cam


try_to_save = False
output_image_dir = "passive_ir_images.log/"
clear_and_create_dir(output_image_dir)

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


def judge_valid(img):
    return True


if __name__ == "__main__":
    # 1. create passive ir device
    cam = init_device()

    plot = DynaPlotter(1, 1, "passive_ir_view")
    plot.set_keypress_callback(keyboard_callback)
    succ_iters = 0
    while plot.is_end == False:
        # 1. take the ir image
        ir_image = get_ir_image(cam)

        if is_try_to_save() == True:
            if judge_valid(ir_image) == True:
                save_pkl(f"{output_image_dir}/{succ_iters}.pkl", ir_image)
                succ_iters += 1
            else:
                print(f"the input ir image is not valid, ignore save")
            set_try_to_save(False)

        
        # 2. display the ir image
        plot.add(ir_image)

        plot.show()