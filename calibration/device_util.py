'''
Kinect device utils
    1. create device
    2. capture the depth image / ir image from device
'''
import dll_util
import numpy as np
from PIL import Image
import device_manager
from log_util import *


def create_kinect_device(mode="passive_ir"):
    '''
        create kinect device
    '''
    device = device_manager.kinect_manager(mode)
    return device


def get_device_mode(device):
    '''
        get the depth mode for given kinect
    '''
    return device.GetDepthMode()


def set_device_mode(device, mode):
    '''
        set the device to specified mode
    '''
    device.SetDepthMode(mode)


def get_passive_ir_mode_str():
    return "passive_ir"


def get_depth_mode_str():
    return "nfov_unbinned"


def set_device_passive_ir_mode(device):
    '''
        set the device to passive ir mode
    '''

    set_device_mode(device, get_passive_ir_mode_str())


def set_device_nfov_unbinned_mode(device):
    '''
        set the device to nfov(narrow fov) unbinned mode
    '''

    set_device_mode(device, get_depth_mode_str())


def get_depth_image(device):
    '''
        get the depth image (integers)
    '''
    depth = device.GetDepthImage()
    return depth


def get_depth_image_mm(device):
    '''
        get the depth image (unit mm)
    '''
    new_image = get_depth_image().astype(np.float32) * device.GetDepthUnit_mm()
    return new_image


def get_ir_image(device):
    '''
        get the ir image (integers)
    '''
    ir = device.GetIrImage()
    return ir


def get_depth_image(device):
    '''
        get the depth image (integers)
    '''
    get_device_mode(device)
    depth = device.GetDepthImage()
    return depth


def get_depth_image_mm(device):
    '''
        get the depth image (unit mm)
    '''
    new_image = get_depth_image().astype(np.float32) * device.GetDepthUnit_mm()
    return new_image


def get_ir_image(device):
    '''
        get the ir image (integers)
    '''
    ir = device.GetIrImage()
    return ir


if __name__ == "__main__":
    log_print("begin to test device utils")
    cam = create_kinect_device()
    log_print(f"current mode {get_device_mode(cam)}")
    set_device_nfov_unbinned_mode(cam)
    log_print(f"current mode {get_device_mode(cam)}")

    ir_image = get_ir_image(cam)
    log_print(f"ir image shape {ir_image.shape} type {ir_image.dtype}")
