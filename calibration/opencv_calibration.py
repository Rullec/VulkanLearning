# 3. given an image, calculate the extrinsics

# 4. config, load the intrinsics (self-calibration), load the chessboard format

# 6. output the intrinsics

# 7. compare two intrinsics, do undistort and display
import cv2
import json
import numpy as np
from copy import deepcopy
from drawer_util import DynaPlotter, calculate_subplot_size, resize


# 1. given an image, judge whether its format meets with the calibration..
def calibration_format_checker(images):
    assert type(images) == list
    for single_image in images:
        assert type(single_image) == np.ndarray
        # assert single_image.dtype == np.uint8, f"{single_image.dtype}"


def draw_image_points(img_lst, imgpoints_lst, chessboard_size_tuple):
    assert len(img_lst) == len(imgpoints_lst)
    size = len(img_lst)
    rows, cols = calculate_subplot_size(size)
    plot = DynaPlotter(rows, cols, "draw_chessboard")
    for _idx in range(size):
        cur_img = cv2.drawChessboardCorners(img_lst[_idx],
                                            chessboard_size_tuple,
                                            imgpoints_lst[_idx], True)
        plot.add(cur_img)
    plot.show()
    while plot.is_end == False:
        pass


class Calibration:
    CHESSBOARD_ROWS_KEY = "chessboard_rows"
    CHESSBOARD_COLS_KEY = "chessboard_cols"
    CHESSBOARD_SIZE_KEY = "chessboard_size"

    # SELF_MTX_KEY = "self_mtx",
    # SELF_DIST_KEY = "self_dist"

    def __init__(self, config_path):
        self.__load_param(config_path)

    def __load_param(self, config_path):
        with open(config_path, 'r') as f:
            cont = json.load(f)
        self.chessboard_rows = cont[Calibration.CHESSBOARD_ROWS_KEY]
        self.chessboard_cols = cont[Calibration.CHESSBOARD_COLS_KEY]
        self.chessboard_size = cont[Calibration.CHESSBOARD_SIZE_KEY]
        # self.self_mtx = cont[Calibration.SELF_MTX_KEY]
        # self.self_dist = cont[Calibration.SELF_DIST_KEY]
        self.enable_imgpoints_visualization = False

    def set_debug_mode(self, value):
        assert type(value) == bool
        self.enable_imgpoints_visualization = value

    def get_chessboard_info(self):
        '''
            return the current chessboard configuration
        '''
        return [
            self.chessboard_rows, self.chessboard_cols, self.chessboard_size
        ]

    def calc_objpoints(self):
        '''
            calculate the object coordinate points sequence (np format)
        '''
        objp = np.zeros((1, self.chessboard_rows * self.chessboard_cols, 3),
                        float)
        objp[0, :, :2] = np.mgrid[0:self.chessboard_rows,
                                  0:self.chessboard_cols].T.reshape(-1, 2)
        objp *= self.chessboard_size
        objp = objp.astype(np.float32)
        return objp

    def calc_corner(self, cur_img):
        scale = 1
        if cur_img.shape[0] == 1024 and cur_img.shape[1] == 1024:
            small_img = deepcopy(cur_img)
            small_img = resize(small_img, 512)
            scale = 2
        else:
            small_img = cur_img

        ret, corners = cv2.findChessboardCorners(
            small_img, (self.chessboard_rows, self.chessboard_cols),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            for i in range(len(corners)):
                corners[i] *= scale
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        100, 0.0001)
            corners = cv2.cornerSubPix(cur_img, corners, (11, 11), (-1, -1),
                                       criteria)
        return corners

    def calc_imgpoints(self, images_lst_, objpoints_pattern):
        '''
            Given a bunch of images and obj points.
            calculate the img point (chessboard corners) respectively

            return: list of [objpoints] and [imgpoints]
        '''
        # 1. check input legality
        calibration_format_checker(images_lst_)

        # 2. calculate the corners
        imgpoints_lst = []
        images_lst = deepcopy(images_lst_)
        validate_images_lst = []
        for cur_img in images_lst:
            imgpoint = self.calc_corner(cur_img)
            if imgpoint is not None:
                validate_images_lst.append(cur_img)
                imgpoints_lst.append(imgpoint)

        # 3. visualize if needed
        if self.enable_imgpoints_visualization == True:
            draw_image_points(validate_images_lst, imgpoints_lst,
                              (self.chessboard_rows, self.chessboard_cols))

        # 4. return
        objpoints_lst = [objpoints_pattern for _ in imgpoints_lst]
        return objpoints_lst, imgpoints_lst

    # 2. given an image, judge whether the img points can be recognized
    def judge_image_recognizable(self, images):
        objp_pattern = self.calc_objpoints()
        objpt_lst, imgpt_lst = self.calc_imgpoints(images, objp_pattern)
        return len(images) == len(objpt_lst)

    # 5. given a batch of image, calculate the intrinsics (calibration)
    def calc_intrinsics_from_images(self, images):
        calibration_format_checker(images)
        objpts_pattern = self.calc_objpoints()
        objpoints_lst, imgpoints_lst = self.calc_imgpoints(
            images, objpts_pattern)

        for _idx in range(len(objpoints_lst)):
            objpoints_lst[_idx] = np.squeeze(objpoints_lst[_idx])

        # print(f"objpoints type {type(objpoints_lst)}")
        # print(f"objpoints len {len(objpoints_lst)}")
        # print(f"objpoints[0] type {type(objpoints_lst[0])}")
        # print(f"objpoints[0] type {objpoints_lst[0]}")
        # print(f"objpoints[0] shape {objpoints_lst[0]}")
        # print(f"imgpoints type {type(imgpoints_lst)}")
        # print(f"imgpoints len {len(imgpoints_lst)}")
        import time
        st = time.time()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints_lst, imgpoints_lst, images[0].shape[::-1], None,
            np.zeros(8))
        ed = time.time()
        print(f"cost {ed - st} s")
        return ret, mtx, dist, rvecs, tvecs

    def calc_reprojection_error(self, objpoints, imgpoints, rvecs, tvecs, mtx,
                                dist):
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs, tvecs, mtx,
                                              dist)
            error = cv2.norm(imgpoints[i], imgpoints2,
                             cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        return total_error

    def calc_extrinsics(self, image, mtx, dist):
        '''
            image: an image
            mtx: camera matrix
            dist: dist coef

            return: transformation matrix, an axis-drawn image
        '''
        objp = self.calc_objpoints()
        objpoints, imgpoints = self.calc_imgpoints([image], objp)
        if len(objpoints) == 0 and len(imgpoints) == 0:
            return None, None, None, None

        assert len(objpoints) == 1
        ret, rvecs, tvecs = cv2.solvePnP(objpoints[0], imgpoints[0], mtx, dist)

        def draw_axis(img, corners, imgpts):
            corner = tuple(corners[0].ravel())
            corner = (int(corner[0]), int(corner[1]))
            new_imgpts_lst = []
            for _idx in range(len(imgpts)):
                a, b = int(imgpts[_idx][0][0]), int(imgpts[_idx][0][1])
                new_imgpts_lst.append((a, b))
            img = cv2.line(img, corner, new_imgpts_lst[0], (255, 0, 0), 5)
            img = cv2.line(img, corner, new_imgpts_lst[1], (0, 255, 0), 5)
            img = cv2.line(img, corner, new_imgpts_lst[2], (0, 0, 255), 5)
            return img

        axis = np.array([[33, 0, 0], [0, 33, 0],
                         [0, 0, -33]]).astype(np.float32).reshape(-1, 3)
        proj_imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        new_image = deepcopy(image)
        new_image = np.ascontiguousarray(new_image)
        new_image = draw_axis(new_image, imgpoints[0], proj_imgpts)

        reproj_error = self.calc_reprojection_error(objpoints, imgpoints,
                                                    rvecs, tvecs, mtx, dist)
        return rvecs, tvecs, new_image, reproj_error
