import numpy as np

import matplotlib.pyplot as plt
from calib_axon import get_camera_pts_to_world_coord, get_mtx_and_dist_from_self, get_mtx_and_dist_from_sdk, calibrate_camera_intrinstic, draw_solvepnp
import device_manager
import os
from scipy.spatial.transform import Rotation as R

global_xpos = None
global_ypos = None
cam = None



def mouse_move(event):
    '''
        Mouse move callback function for matplotlib
    '''
    global global_xpos, global_ypos
    global_xpos, global_ypos = event.xdata, event.ydata

    if global_xpos is not None and global_ypos is not None:
        global_xpos = int(global_xpos)
        global_ypos = int(global_ypos)


def on_press(event):
    import sys
    print('press', event.key)
    sys.stdout.flush()





def show_image(depth_image):
    import matplotlib.pyplot as plt
    plt.imshow(depth_image)
    plt.show()


def display(cam, mode="depth", save=False):
    global global_xpos, global_ypos
    import matplotlib.pyplot as plt
    # 打开交互模式
    plt.ion()
    fig1 = plt.figure('frame')

    plt.connect('motion_notify_event', mouse_move)

    plt.connect('key_press_event', on_press)
    # fig2 = plt.figure('subImg')

    iters = 0
    while True:
        fig1.clf()
        ax1 = fig1.add_subplot(1, 1, 1)
        if mode == "depth":
            raw_res = get_depth_image_mm(cam)
        elif mode == "ir":
            raw_res = get_ir_image(cam)

        res = resize(raw_res)
        # print(f"raw {res.dtype}")

        if global_xpos is not None and global_ypos is not None:
            print(
                f"\rdepth ({global_ypos}, {global_xpos}) = {int(res[global_ypos, global_xpos])} mm",
                end='')
        ax1.imshow(res)
        if save == True:
            import pickle
            string = f"{iters}.pkl"
            with open(string, 'wb') as f:
                pickle.dump(raw_res, f)
            iters += 1
            print(f"[log] save to {string}")
        ax1.title.set_text("depth image (mm)")
        # ax1.plot(p1[:, 0], p1[:, 1], 'g.')
        plt.pause(3e-2)


def load_agent(agent_json_path):
    '''
        load the agent from given config file
    '''
    import shutil
    single_name = os.path.split(agent_json_path)[-1]
    import time
    single_name = str(time.time()) + single_name
    shutil.copy(agent_json_path, single_name)
    import json
    from train import build_net
    net_type = build_net(single_name)
    import torch
    net_device = torch.device("cuda", 0)
    net = net_type(single_name, device=net_device)
    os.remove(single_name)
    return net


def infer_net(net, depth_image):
    '''
        Given a net and a depth image, return the parameter
    '''
    depth_image = resize(depth_image)
    depth_image = np.expand_dims(depth_image, 0)
    param = net.infer(depth_image)
    print(f"get feature {param}")
    show_image(depth_image.squeeze())
    return param


def ir_camera_calibration_legacy():
    cam = device_manager.kinect_manager()

    import matplotlib.pyplot as plt
    import cv2
    from calib_axon import get_camera_pts_to_world_coord
    plt.ion()
    fig1 = plt.figure('frame')
    iter = 0
    while True:
        # clear but do not close the figure
        fig1.clf()
        ax1 = fig1.add_subplot(1, 1, 1)
        image = get_ir_image(cam).astype(np.uint8)
        from PIL import Image
        new_pil_image = Image.fromarray(image)
        new_pil_image.save(f"tmp/{iter}.png")
        iter += 1
        # print(image.shape)
        ax1.imshow(image)
        mat_lst = get_camera_pts_to_world_coord([image], False)
        print(f"camera pos {mat_lst[0][:, 3]}")
        # draw the image
        ax1.title.set_text("ir image")

        # pause
        plt.pause(3e-2)

    print(f"get ir succ, shape {image.shape}")


def extract_trans_mat(mat_str: str) -> np.ndarray:
    row_list = []
    for _idx, i in enumerate(mat_str.split("\n")):
        # print(f"{_idx} : {i}")
        res = [float(i) for i in i.replace("[", "").replace("]", "").split()]
        if len(res) != 4:
            continue
        else:
            row_list.append(res)

        # assert len(row_list[-1]) == 4, f"{row_list[-1]}"
    # exit()
    assert len(row_list) == 4
    mat = np.array(row_list)
    return mat


def refine_rotmat(rotmat):
    rotation = R.from_matrix(rotmat)
    rotation = R.from_rotvec(R.as_rotvec(rotation))
    return rotation.as_matrix()


def calc_intersection_ray_plane(ori, dir, plane_point, plane_normal):
    D = -np.dot(plane_point, plane_normal)
    t = -(np.dot(plane_normal, ori) + D) / (np.dot(plane_normal, dir))
    new_pt = dir * t + ori
    return new_pt


def cal_focus(rotmat, cam_pos):
    dir = np.array([0, 0, 1])
    cam_ori = cam_pos[:3]
    cam_dir = np.matmul(rotmat, dir)
    plane_point = np.array([0, 0, 0])
    plane_normal = np.array([0, 0, 1])
    # print(f"cam point {cam_ori}")
    # print(f"cam dir {cam_dir}")
    focus = calc_intersection_ray_plane(cam_ori, cam_dir, plane_point,
                                        plane_normal)
    # p_cam, p_cloth_center, dist = calc_two_line_nearest_points(
    #     cam_ori, cam_dir, cloth_center_ori, cloth_center_dir)
    # print(f"p_cam {p_cam}")
    # print(f"p_cloth_center {p_cloth_center}")
    # print(f"dist {dist}")
    return focus


def rotation_info_output(mat):
    # 2. get rotmat and pos
    cam_rotmat = refine_rotmat(mat[:3, :3])
    cam_pos = mat[:, 3]
    # print(f"rot mat \n{cam_rotmat}")
    focus_point = cal_focus(cam_rotmat, cam_pos)

    # output
    # cam_pos[0:3] *= 1e-3
    print(f"camera pos {cam_pos} mm")
    print(f"focus point {focus_point} mm")
    # print(f"cam rotmat {cam_rotmat}")
    rotvec = R.from_matrix(cam_rotmat).as_rotvec()
    # print(f"rotvec {rotvec}")
    print(f"camera rot theta {(np.pi - np.linalg.norm(rotvec)) / np.pi * 180}")


def convert_kinect_ir_image(image):
    import copy
    new_image = copy.deepcopy(image)
    new_image[new_image >= 60000] = 20000
    max = np.max(new_image)
    new_image[new_image == 20000] = max
    # print(f"max {max}")
    return (image.astype(np.float32) / max * 255).astype(np.uint8)


def ir_camera_calc_extrinsics(mode):
    # print("begin to init")
    cam = device_manager.kinect_manager(mode)
    # print("end to init")

    # exit(0)
    import matplotlib.pyplot as plt
    mtx, dist = get_mtx_and_dist_from_sdk(cam)

    plt.ion()
    fig1 = plt.figure('frame')
    iter = 0
    # minus_id = 0
    # positive_id = 0
    # os.makedirs("positive")
    # os.makedirs("negative")
    # clear = lambda: os.system('cls')
    avg_trans = np.zeros([4, 4])
    avg_counter = 0
    import time
    while True:
        # st = time.time()
        # print("------------------------")
        # clear but do not close the figure
        fig1.clf()
        ax1 = fig1.add_subplot(1, 1, 1)
        # print("begin to get ir")
        image = get_ir_image(cam)
        # print("end to get ir")

        # print("begin to convert ir")
        image = convert_kinect_ir_image(image)
        # image = resize(image).astype(np.uint8)
        print(image.shape)
        # exit()
        # print("end to convert ir")
        iter += 1
        # print(image.shape)
        # print("begin to show ir")
        ax1.imshow(image)
        image = np.ascontiguousarray(image)
        # print("end to show ir")

        # print("begin to get sdk")
        # get camera transform from sdk intrinsics
        sdk_mtx, sdk_dist = get_mtx_and_dist_from_sdk(cam)
        # print("end to get sdk")
        # self_mtx, self_dist = get_mtx_and_dist_from_self(cam)

        # print("begin to calc")
        sdk_camera_pts_to_world_coords = get_camera_pts_to_world_coord(
            sdk_mtx, sdk_dist, image)
        # print("end to calc")
        # self_camera_pts_to_world_coords = get_camera_pts_to_world_coord(
        #     self_mtx, self_dist, image)
        if sdk_camera_pts_to_world_coords is not None:
            # print(f"camera pos {sdk_camera_pts_to_world_coords[:, 3]}")
            # print(f"camera rot \n{sdk_camera_pts_to_world_coords[0:3, 0:3]}")
            print(
                f"camera trans(sdk intri) \n{sdk_camera_pts_to_world_coords}")
            # print(
            #     f"camera trans(self intri) \n{self_camera_pts_to_world_coords}"
            # )
            rotation_info_output(sdk_camera_pts_to_world_coords)
            avg_trans = (avg_trans * avg_counter +
                         sdk_camera_pts_to_world_coords) / (avg_counter + 1)
            avg_counter += 1
            # print(f"avg trans \n{avg_trans}")

        # draw the image
        ax1.title.set_text("ir image")

        # pause
        plt.pause(3e-1)
        # ed = time.time()
        # print(f"frame cost {ed - st} s")

    print(f"get ir succ, shape {image.shape}")


def ir_camera_calc_intrinsics(data_dir):
    png_files = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
    # pkl_files = pkl_files[:2]
    from PIL import Image
    img_lst = []
    for file in png_files:

        img = Image.open(file)
        img = np.array(img)
        img = convert_kinect_ir_image(img)

        img_lst.append(img)
    res = calibrate_camera_intrinstic(img_lst)
    print(res)


def draw_axis():
    cam = device_manager.kinect_manager()

    plt.ion()
    fig1 = plt.figure('frame')
    iter = 0
    import time
    while True:
        st = time.time()
        # print("------------------------")
        # clear but do not close the figure
        fig1.clf()
        ax1 = fig1.add_subplot(1, 1, 1)
        image = get_ir_image(cam)

        image = convert_kinect_ir_image(image)
        iter += 1
        # print(image.shape)
        image = np.ascontiguousarray(image)

        # get camera transform from sdk intrinsics
        sdk_mtx, sdk_dist = get_mtx_and_dist_from_self(cam)
        r, t, image = draw_solvepnp(sdk_mtx, sdk_dist, image)
        print(f"r {r}")
        if image is None:
            continue
        ax1.imshow(image)
        # draw the image
        ax1.title.set_text("ir image")

        # pause
        plt.pause(3e-2)
        ed = time.time()




def ir_display():
    cam = device_manager.kinect_manager()

    plt.ion()
    fig1 = plt.figure('frame')
    iter = 0
    import time
    while True:
        # clear but do not close the figure
        fig1.clf()
        ax1 = fig1.add_subplot(1, 1, 1)
        image = get_ir_image(cam)

        image = convert_kinect_ir_image(image)
        iter += 1
        ax1.imshow(image)
        # draw the image
        ax1.title.set_text("ir image")

        # pause
        plt.pause(3e-3)
if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    depth_mode = "passive_ir"
    ir_camera_calc_extrinsics(depth_mode)
    # draw_axis()
    # ir_camera_calc_intrinsics(data_dir="captured_ir_images")
