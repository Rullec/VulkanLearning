from axon import get_ir_image, convert_kinect_ir_image
from calib_axon import calibrate_camera_intrinstic
import device_manager
import matplotlib.pyplot as plt
import os

plt.ion()
fig1 = plt.figure('frame')

output_dir = "captured_ir_images/"
cam = device_manager.kinect_manager("passive_ir")
import shutil
if os.path.exists(output_dir) is True:
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

iters = 1
pressed = False


def on_press(event):
    global pressed
    import sys
    if event.key != "escape":
        print('press', event.key)
        sys.stdout.flush()
        pressed = True


plt.connect('key_press_event', on_press)


def judge_succ(image):
    from copy import deepcopy
    new_image = deepcopy(image)
    mtx, dist = calibrate_camera_intrinstic([new_image])
    return (mtx is not None) and (dist is not None)


while True:
    # clear but do not close the figure
    fig1.clf()
    ax1 = fig1.add_subplot(1, 1, 1)
    captured_img = convert_kinect_ir_image(get_ir_image(cam))
    path = os.path.join(output_dir, f"{iters}.png")
    if pressed is True:
        succ = judge_succ(captured_img)
        if succ is True:
            
        else:
            print("[warn] calibrate failed, should not be included")
        pressed = False
    ax1.imshow(captured_img)
    ax1.title.set_text("captured ir image")
    plt.pause(1e-3)
