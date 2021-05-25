import numpy as np
from calib_axon import get_camera_pts_to_world_coord, get_mtx_and_dist, get_mtx_and_dist_sdk
import video_manager
import os
from scipy.spatial.transform import Rotation as R
global_xpos = None
global_ypos = None
cam = None


def resize(image, size = 128):
    # height, width
    height, width = image.shape
    mid = width / 2
    assert width % 2 == 0
    # to a square
    image = image[:, int(mid - height / 2):int(mid + height / 2)].astype(float)
    # expand this square to
    from PIL import Image
    image = Image.fromarray(image)
    image = image.resize((size, size))
    image = np.array(image)
    return image


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


def get_depth_image_mm(cam):
    '''
        get the depth image (unit mm)
    '''
    # print("begin to get depth image")
    depth = cam.GetDepthImage().astype(float) * cam.GetDepthUnit_mm()
    # print("succ to get depth image")
    return depth


def get_ir_image(cam):
    '''
        get the ir image
    '''
    ir = cam.GetIrImage()
    return ir


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
    cam = video_manager.video_manager()

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

def ir_camera_calibration():
    cam = video_manager.video_manager()
    # sdk_mtx = cam.GetDepthIntrinsicMtx()
    # sdk_dist = cam.GetDepthIntrinsicDistCoef()
    # # print(f"GetDepthIntrinsicMtx \n{sdk_mtx}")
    # # print(f"GetDepthIntrinsicDistCoef {sdk_dist}")
    # self_mtx, self_dist = get_mtx_and_dist()
    # diff_mtx = self_mtx - sdk_mtx
    # diff_dist = self_dist - sdk_dist[0:len(self_dist)]
    # print(f"self mtx \n{self_mtx}")
    # print(f"sdk mtx \n{sdk_mtx}")
    # print(f"diff mtx \n{diff_mtx}")

    # print(f"self dist {self_dist}")
    # print(f"sdk dist {sdk_dist}")
    # print(f"diff dist {diff_dist}")

    # exit(0)
    import matplotlib.pyplot as plt
    import cv2
    # from calib_axon import get_camera_pts_to_world_coord
    mtx, dist = get_mtx_and_dist_sdk()

    plt.ion()
    fig1 = plt.figure('frame')
    iter = 0
    minus_id = 0
    positive_id = 0
    # os.makedirs("positive")
    # os.makedirs("negative")
    clear = lambda: os.system('cls')
    avg_trans = np.zeros([4, 4])
    avg_counter = 0
    camera_matrix, dist_coef = get_mtx_and_dist_sdk()
    while True:
        # clear()
        # print("------------------------")
        # clear but do not close the figure
        fig1.clf()
        ax1 = fig1.add_subplot(1, 2, 1)
        ax2 = fig1.add_subplot(1, 2, 2)
        image = get_ir_image(cam).astype(np.uint8)
        undistorted_image = cv2.undistort(
            image, camera_matrix, dist_coef, None, None
        )
        # from PIL import Image
        # new_pil_image.save(f"tmp/{iter}.png")
        iter += 1
        # print(image.shape)
        ax1.imshow(image)
        ax2.imshow(undistorted_image)
        # mat_lst = get_camera_pts_to_world_coord([image], False)
        image = np.ascontiguousarray(image, dtype=np.uint8)

        # get camera transform from sdk intrinsics
        sdk_mtx, sdk_dist = get_mtx_and_dist_sdk()
        sdk_camera_pts_to_world_coords = get_camera_pts_to_world_coord(
            sdk_mtx, sdk_dist, image)

        if sdk_camera_pts_to_world_coords is not None:
            # print(f"camera pos {sdk_camera_pts_to_world_coords[:, 3]}")
            # print(f"camera rot \n{sdk_camera_pts_to_world_coords[0:3, 0:3]}")
            print(f"camera trans \n{sdk_camera_pts_to_world_coords}")
            rotation_info_output(sdk_camera_pts_to_world_coords)
            avg_trans = (avg_trans * avg_counter +
                         sdk_camera_pts_to_world_coords) / (avg_counter + 1)
            avg_counter += 1
            # print(f"avg trans \n{avg_trans}")

        # # get camera transform from self intrinsics
        # self_mtx, self_dist = get_mtx_and_dist()
        # self_camera_pts_to_world_coords = get_camera_pts_to_world_coord(
        #     self_mtx, self_dist, image)
        # if self_camera_pts_to_world_coords is not None:
        #     print(f"self_camera pos {self_camera_pts_to_world_coords[:, 3]}")
        #     print(f"self_camera rot \n{self_camera_pts_to_world_coords[0:3, 0:3]}")

        # draw the image
        ax1.title.set_text("ir image")

        # pause
        plt.pause(3e-2)

    print(f"get ir succ, shape {image.shape}")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    ir_camera_calibration()
    # import cv2
    # from calib_axon import calc_objp, calc_objective_coordinate_in_screen_coordinate
    # img = cv2.imread("images/1.bmp")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # objp = calc_objp()
    # objpoints, imgpoints = calc_image_points([img], objp)
    # print(f"obj points {objpoints}")
    # print(f"img points {imgpoints}")
    # X_positive, Y_positive = calc_objective_coordinate_in_screen_coordinate(
    #     objpoints[0], imgpoints[0])

    # new_objp = objp[0]
    # new_objp[]

    # chess_board_size = get_chessboard_size()

    # ir_camera_calibration()
    # cam = video_manager.video_manager()
    # display(cam, save=True)
    # unit = cam.GetDepthUnit_mm()
    # print(f"cur unit {unit} mm")
    # depth = get_depth_image_mm(cam)
    # import matplotlib.pyplot as plt
    # plt.imshow(depth)
    # plt.plot()
    # conf_path = "..\config\\train_configs\\conv_conf.json"
    # print(f"depth shape {depth.shape}")
    # # display()
    # net = load_agent(conf_path)
    # infer_net(net, depth)