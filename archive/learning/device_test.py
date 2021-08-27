from numpy import disp
import device_manager
from axon import mouse_move, on_press, get_depth_image_mm, resize, global_xpos, global_ypos, get_ir_image


def display_ir(cam, save=False):
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
        raw_res = get_ir_image(cam) * 100

        # res = resize(raw_res)
        # print(f"raw {res.dtype}")

        # if global_xpos is not None and global_ypos is not None:
        #     print(
        #         f"\rdepth ({global_ypos}, {global_xpos}) = {int(res[global_ypos, global_xpos])} mm",
        #         end='')
        ax1.imshow(raw_res)
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


def display_depth(cam, save=False):
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
        raw_res = get_depth_image_mm(cam)

        import cv2
        mat, dist = cam.GetDepthIntrinsicMtx_sdk( ), cam.GetDepthIntrinsicDistCoef_sdk()
        raw_res = cv2.undistort(raw_res, mat, dist)
        # res = resize(raw_res)
        # print(f"raw {res.dtype}")

        # if global_xpos is not None and global_ypos is not None:
        #     print(
        #         f"\rdepth ({global_ypos}, {global_xpos}) = {int(res[global_ypos, global_xpos])} mm",
        #         end='')
        ax1.imshow(raw_res)
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


if __name__ == "__main__":
    cam = device_manager.kinect_manager("passive_ir")
    mat_passive_ir, dist_coef_passive_ir = cam.GetDepthIntrinsicMtx_sdk(
    ), cam.GetDepthIntrinsicDistCoef_sdk()
    # print(cam.GetDepthMode())
    cam.SetDepthMode("nfov_unbinned")
    mat_nfov_unbinned, dist_coef_nfov_unbinned = cam.GetDepthIntrinsicMtx_sdk(
    ), cam.GetDepthIntrinsicDistCoef_sdk()
    print(f"ir {dist_coef_passive_ir}\n {mat_passive_ir}")
    print(f"nfov unbinned {dist_coef_nfov_unbinned}\n {mat_nfov_unbinned}")
    # exit()
    # exit()
    # display_ir(cam)
    display_depth(cam)
