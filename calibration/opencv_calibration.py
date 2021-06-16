# 3. given an image, calculate the extrinsics

# 4. config, load the intrinsics (self-calibration), load the chessboard format

# 5. given a batch of image, calculate the intrinsics (calibration)

# 6. output the intrinsics

# 7. compare two intrinsics, do undistort and display
import cv2
import json
import numpy as np
from copy import deepcopy
from drawer_util import DynaPlotter, calculate_subplot_size


# 1. given an image, judge whether its format meets with the calibration..
def calibration_format_checker(images):
    assert type(images) == list
    for single_image in images:
        assert type(single_image) == np.ndarray
        assert single_image.dtype == np.uint8


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
    CHESSBOARD_ROWS_KEY = "CHESSBOARD_ROWS"
    CHESSBOARD_COLS_KEY = "CHESSBOARD_COLS"
    CHESSBOARD_SIZE_KEY = "CHESSBOARD_SIZE"

    def __init__(self, config_path):
        self.__load_param(config_path)

    def __load_param(self, config_path):
        with open(config_path, 'r') as f:
            cont = json.load(f)
        self.chessboard_rows = cont[Calibration.CHESSBOARD_ROWS_KEY]
        self.chessboard_cols = cont[Calibration.CHESSBOARD_COLS_KEY]
        self.chessboard_sze = cont[Calibration.CHESSBOARD_SIZE_KEY]
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
        return objp

    def calc_corner(self, cur_img):
        ret, corners = cv2.findChessboardCorners(
            cur_img, (self.chessboard_rows, self.chessboard_cols),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,
                        0.001)
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