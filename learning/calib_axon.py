#!/usr/bin/env python

import cv2
import numpy as np
import glob
from numpy.lib.function_base import disp
from scipy.spatial.transform import Rotation as R

# from axon import calc_objective_coordinate_in_screen_coordinate

CHECKERBOARD = (6, 9)
square_size = 50.0  # mm
display = False

def calc_objective_coordinate_in_screen_coordinate(objpoints, imgpoints):
    # print(f"obj shape {objpoints.shape}")
    # print(f"img shape {imgpoints.shape}")
    assert type(objpoints) is np.ndarray
    assert type(imgpoints) is np.ndarray
    assert len(objpoints.shape) is 3
    assert len(imgpoints.shape) is 3
    new_objpoints = np.squeeze(objpoints)
    new_imgpoints = np.squeeze(imgpoints)
    num = new_objpoints.shape[0]
    # 1. confirm the origin point's index, obj position, resolution position
    origin_id = 0
    origin_objpt = new_objpoints[origin_id, :]
    origin_imgpt = new_imgpoints[origin_id, :]
    # print(
    #     f"image points[0] range {np.max( new_imgpoints[:, 0]) -np.min( new_imgpoints[:, 0])  }"
    # )
    # print(
    #     f"image points[1] range {np.max( new_imgpoints[:, 1]) -np.min( new_imgpoints[:, 1])  }"
    # )

    # print(f"origin img point {new_imgpoints[0, :]}")
    # print(f"obj pt {new_objpoints[:10, :]}")
    # print(f"img pt {new_imgpoints[:10, :]}")
    # 2. find a point which has the same obj poistion X with origin, but different Y position
    for i in range(num):
        cur_pt = new_objpoints[i, :]
        if cur_pt[0] != origin_objpt[0] and cur_pt[1] == origin_objpt[1]:
            X_positive_dir = (new_imgpoints[i, :] -
                              origin_imgpt) / (cur_pt[0] - origin_objpt[0])
            # print(f"raw X_positive_dir {X_positive_dir}, origin imgpt {origin_imgpt}, new imgpt {new_imgpoints[i, :]}")
            if np.abs(X_positive_dir[0]) > np.abs(X_positive_dir[1]):
                X_positive_dir[1] = 0
            else:
                X_positive_dir[0] = 0
            X_positive_dir = X_positive_dir / np.linalg.norm(X_positive_dir)
            # print(f"X_positive_dir in screen coords: {X_positive_dir}")
            break

    # 3. find a point which has the same obj poistion Y with origin, but different X position
    for i in range(num):
        cur_pt = new_objpoints[i, :]
        if cur_pt[0] == origin_objpt[0] and cur_pt[1] != origin_objpt[1]:
            # print(f"new_imgpoint {new_imgpoints[i, :]}, origin img point {origin_imgpt}")
            # print(f"new_objpoint {cur_pt}, origin obj point {origin_objpt}")
            Y_positive_dir = (new_imgpoints[i, :] -
                              origin_imgpt) / (cur_pt[1] - origin_objpt[1])
            # print(f"raw Y positive dir {Y_positive_dir}")
            if np.abs(Y_positive_dir[0]) > np.abs(Y_positive_dir[1]):
                Y_positive_dir[1] = 0
            else:
                Y_positive_dir[0] = 0
            Y_positive_dir = Y_positive_dir / np.linalg.norm(Y_positive_dir)
            # print(f"Y_positive_dir in screen coords: {Y_positive_dir}")
            break
    
    assert np.dot(X_positive_dir, Y_positive_dir) == 0, f"({X_positive_dir}, {Y_positive_dir})"
    return X_positive_dir, Y_positive_dir
    exit(0)


def axis_angle_to_rotmat(aa):
    return R.from_rotvec(aa).as_matrix()


def combine_transform_mat(rot_mat, translate_vec):
    assert rot_mat.shape == (3, 3), rot_mat.shape
    mat = np.identity(4)
    mat[0:3, 0:3] = rot_mat
    mat[0:3, 3] = translate_vec[:3]
    return mat


def calc_image_points(images, objp):
    assert type(images) is list
    assert type(objp) is np.ndarray
    assert len(objp.shape) == 3
    global display
    if len(images) == 0:
        return None, None

    # Defining the dimensions of checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = []
    for _idx, gray_ in enumerate(images):
        gray = gray_.copy()
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       criteria)
            # print(type(corners))
            # print(corners.shape)
            # exit()
            # swap = corners[:, :, 0]
            # corners[:, :, 0] = corners[:, :, 1]
            # corners[:, :, 1] = swap
            # set the origin to black
            # print(imgpoints[0].shape)
            # exit()
            img = cv2.drawChessboardCorners(gray, CHECKERBOARD, corners, ret)
            # corners = corners[:, :, ::-1]

            # draw corners to black
            imgpoints.append(corners)
        else:
            # print(f"[warn] calibrate failed for img {_idx}, continue")
            continue
        if display == True:
            cv2.imshow('img', img)
            cv2.waitKey(0)

    if display == True:
        cv2.destroyAllWindows()
    return objpoints, imgpoints


def get_chessboard_size():
    global CHECKERBOARD
    return CHECKERBOARD


def calc_objp():
    global CHECKERBOARD
    global square_size

    # Creating vector to store vectors of 3D points for each checkerboard image

    # Creating vector to store vectors of 2D points for each checkerboard image

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), float)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                              0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= 30
    return objp


def calibrate_camera(images):

    # Extracting path of individual image stored in a given directory
    objp = calc_objp()
    # h, w = img.shape[:2]
    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    none_result = (None, None, None, None)
    objpoints, imgpoints = calc_image_points(images, objp)
    # if objpoints is None or imgpoints is None:
    if len(objpoints) == 0 or len(imgpoints) == 0:
        print("[warn] failed to find the chessboard vertices")
        return none_result

    print(f"num of images {len(objpoints)}")
    # exit(0)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       images[0].shape[::-1],
                                                       None, np.zeros(8))
    if ret is False:
        return none_result
    else:
        return mtx, dist, rvecs, tvecs


def convert_rtvecs_to_transform(rvecs, tvecs):
    rvecs = np.squeeze(rvecs)
    tvecs = np.squeeze(tvecs)
    transformat_mat_lst = []
    if len(rvecs.shape) == 2:
        for i in range(rvecs.shape[0]):
            transformat_mat_lst.append(
                combine_transform_mat(axis_angle_to_rotmat(rvecs[i, :]),
                                      tvecs[i, :]))
    else:
        transformat_mat_lst.append(
            combine_transform_mat(axis_angle_to_rotmat(rvecs), tvecs))
    return transformat_mat_lst


def calculate_transform_matrix_for_obj_points_to_camera_coords(images):
    if len(images) == 0:
        print("[debug] empty image list")
        return None
    '''
        return a transformation matrix which can convert points in objects coords to camera coords
    '''

    mtx, dist, rvecs, tvecs = calibrate_camera(images)

    if mtx is None:
        print("[error] failed to calibrate the camera")
        return None

    transformat_mat_lst = convert_rtvecs_to_transform(rvecs, tvecs)

    return transformat_mat_lst


def get_world_pts_to_obj_coord(X_positive_in_screen_coord,
                               Y_positive_in_screen_coord):
    '''
    Get a transformation matrix which can convert points in world coordinate to object coordinates
    '''
    global CHECKERBOARD
    global square_size
    height_of_table = 63  # mm
    trans_mat = np.identity(4)
    # in screen coordinate, X plus is down, Y plus is left, origin is right-up corner
    if (X_positive_in_screen_coord[1]
            == 1.0) and (Y_positive_in_screen_coord[0] == -1.0):
        x_size = CHECKERBOARD[0]
        y_size = (CHECKERBOARD[1] - 1) / 2
        # y_size = y_size + 1
        # print((x_size, y_size))
        trans_mat[0, 3] = square_size * x_size + height_of_table
        trans_mat[1, 3] = square_size * y_size
        trans_mat[2, 3] = 0
        trans_mat[0:3, 0:3] = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    # in screen coordinate, X plus is up, Y plus is right, origin is left-bottom corner
    elif (X_positive_in_screen_coord[1]
          == -1.0) and (Y_positive_in_screen_coord[0] == 1.0):

        x_size = -1
        y_size = (CHECKERBOARD[1] - 1) / 2
        # y_size = y_size - 1
        trans_mat[0, 3] = square_size * x_size - height_of_table
        trans_mat[1, 3] = square_size * y_size
        trans_mat[2, 3] = 0
        trans_mat[0:3, 0:3] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    else:
        print(
            f"[error] X plus {X_positive_in_screen_coord} Y plus {Y_positive_in_screen_coord} unsupported"
        )
    # print(f"world to obj coord \n{trans_mat}")
    # x_size = CHECKERBOARD[0]
    # y_size = (CHECKERBOARD[1] - 1) / 2
    # trans_mat[0, 3] = square_size * x_size
    # trans_mat[1, 3] = square_size * y_size
    # trans_mat[2, 3] = 0
    # trans_mat[0:3, 0:3] = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    return trans_mat


def get_obj_pts_to_world_coord():
    return np.linalg.inv(get_world_pts_to_obj_coord())
    # print(np.matmul (new_res, old_res))


def draw_solvepnp(mtx, dist, image):
    def draw(img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        corner = (int(corner[0]), int(corner[1]))
        new_imgpts_lst = []
        for _idx in range(len(imgpts)):
            a, b = int(imgpts[_idx][0][0]), int(imgpts[_idx][0][1])
            new_imgpts_lst.append((a, b))
            # print(imgpts[_idx])
            # print(type(imgpts[_idx]))

        # print(f"corner {corner}")
        # print(f"imgpts 0 {new_imgpts_lst[0]}")
        # print(f"imgpts 1 {new_imgpts_lst[1]}")
        # print(f"imgpts 2 {new_imgpts_lst[2]}")
        # print(f"example {(1, 2)}")
        # exit()
        # exit()
        img = cv2.line(img, corner, new_imgpts_lst[0], (255, 0, 0), 5)
        img = cv2.line(img, corner, new_imgpts_lst[1], (0, 255, 0), 5)
        img = cv2.line(img, corner, new_imgpts_lst[2], (0, 0, 255), 5)
        return img

    objp = calc_objp()

    axis = float([[33, 0, 0], [0, 33, 0], [0, 0, -33]]).reshape(-1, 3)
    objpoints, imagepoints = calc_image_points([image], objp)
    if len(objpoints) == 0 or len(imagepoints) == 0:
        return None, None, None
    else:
        ret, rvecs, tvecs = cv2.solvePnP(objpoints[0], imagepoints[0], mtx,
                                         dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        image = draw(image, imagepoints[0], imgpts)
        return rvecs, tvecs, image


def get_camera_pts_to_world_coord(mtx, dist, image):
    assert type(image) == np.ndarray
    # print("begin to calibrate extrinstics")
    rvecs, tvecs, X_plus_vector_in_screen_coords, Y_plus_vector_in_screen_coords = calibrate_camera_extrinstic(
        mtx, dist, [image])
    # print("end to calibrate extrinstics")
    # print(f"rvecs {rvecs}")
    # print(
    #     f"X plus {X_plus_vector_in_screen_coords}, Y plus {Y_plus_vector_in_screen_coords}"
    # )
    if rvecs is not None and tvecs is not None:
        obj_pts_to_camera_coords = convert_rtvecs_to_transform(rvecs, tvecs)
        # print(f"obj_pts_to_camera_coords\n {obj_pts_to_camera_coords[0]}")
        world_pts_to_camera_coords = np.matmul(
            obj_pts_to_camera_coords[0],
            get_world_pts_to_obj_coord(X_plus_vector_in_screen_coords,
                                       Y_plus_vector_in_screen_coords))
        camera_pts_to_world_coords = np.linalg.inv(world_pts_to_camera_coords)
        return camera_pts_to_world_coords
        # print(f"camera pos {camera_pts_to_world_coords[:, 3]}")
        # print(f"camera rot \n{camera_pts_to_world_coords[0:3, 0:3]}")

        # if camera_pts_to_world_coords[0, 3] < 0:
        #     Image.fromarray(image).save(f"negative/{minus_id}.bmp")
        #     minus_id += 1
        # else:
        #     Image.fromarray(image).save(f"positive/{positive_id}.bmp")
        #     positive_id += 1
    else:
        print("[warn] calibrate failed")
        return None


def legacy_cal_camera_pos():
    images = glob.glob('./images/*.bmp')
    gray_images = []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(type(gray))
        # print(gray.shape)
        # print(gray.dtype)
        # exit(0)
        gray_images.append(gray)

    camera_pts_to_world_coord = get_camera_pts_to_world_coord(gray_images)
    for idx, i in enumerate(camera_pts_to_world_coord):
        print(f"camera_pts_to_world_coord {idx}\n {i}")


def calibrate_camera_intrinstic(images):

    mtx, dist, rvecs, tvecs = calibrate_camera(images)
    # for i in range(len(images)):
    #     print(f"{i} mtx \n {mtx}, dist {dist[i]}")
    return mtx, dist


def calibrate_camera_extrinstic(mtx, dist, image):
    # print("begin to calc objp")
    objp = calc_objp()
    # print("end to calc objp")
    # print("begin to calc img pts")
    objpoints, imagepoints = calc_image_points(image, objp)
    # print("end to calc img pts")
    if len(objpoints) == 0 or len(imagepoints) == 0:
        return None, None, None, None
    else:
        # print("begin to calc pnp")
        ret, rvecs, tvecs = cv2.solvePnP(objpoints[0], imagepoints[0], mtx,
                                         dist)
        # print("end to calc pnp")
        X_positive, Y_positive = calc_objective_coordinate_in_screen_coordinate(
            objpoints[0], imagepoints[0])
        if ret is False:
            print("[warn] solvePnP failed")
        return rvecs, tvecs, X_positive, Y_positive


def load_images(pat):
    images = glob.glob(pat)
    gray_images = []
    # assert len(images) >= 2
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(type(gray))
        # print(gray.shape)
        # print(gray.dtype)
        # exit(0)
        gray_images.append(gray)
    return gray_images


# def get_mtx_and_dist_sdk():
#     assert False, "axon result, deprecated"
#     mtx = np.array([[504.65344238, 0., 314.07043457],
#                     [0., 504.35171509, 233.0559845], [0., 0., 1.]])
#     dist = np.array([
#         -0.03364468, -0.03456402, -0.0007391, 0.00034046, -0.09825993, 0., 0.,
#         0.
#     ])
#     return mtx, dist


def get_mtx_and_dist_from_sdk(cam):
    return cam.GetDepthIntrinsicMtx_sdk(), cam.GetDepthIntrinsicDistCoef_sdk()


def get_mtx_and_dist_from_self(cam):
    return cam.GetDepthIntrinsicMtx_self(), cam.GetDepthIntrinsicDistCoef_self(
    )
    # mtx = np.array([[504.65344238, 0., 314.07043457],
    #                 [0., 504.35171509, 233.0559845], [0., 0., 1.]])
    # dist = np.array([
    #     -0.03364468, -0.03456402, -0.0007391, 0.00034046, -0.09825993, 0., 0.,
    #     0.
    # ])
    # return mtx, dist


def get_mtx_and_dist():
    # assert False, "please donot call this API anymore... use get_mtx_and_dist_sdk instead"
    mtx = np.array([[500.8670683, 0., 302.22452252],
                    [0., 504.03608172, 226.05892077], [
                        0.,
                        0.,
                        1.,
                    ]])

    dist = np.array(
        [[-0.02475915, -0.0327474, -0.0040662, -0.00849125, 0.14034689]])
    return mtx, dist


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    # legacy_cal_camera_pos()
    pat = './images/*.bmp'
    # images = load_images(pat)

    mtx, dist = get_mtx_and_dist_sdk()
    # print(f"mtx = \n{mtx}")
    # print(f"dist = {dist}")

    images = load_images(pat)
    rvecs, tvecs = calibrate_camera_extrinstic(mtx, dist, [images[0]])

    obj_pts_to_camera_coords = convert_rtvecs_to_transform(rvecs, tvecs)
    # print(obj_pts_to_camera_coords)
    # exit()
    if obj_pts_to_camera_coords is not None:
        camera_pts_to_obj_coords = [
            np.linalg.inv(i) for i in obj_pts_to_camera_coords
        ]
        obj_pts_to_world_coord = get_obj_pts_to_world_coord()
        camera_pt_to_world_coord_lst = [
            np.matmul(i, obj_pts_to_world_coord)
            for i in camera_pts_to_obj_coords
        ]
        print(f"camera_pt_to_world_coord = \n{camera_pt_to_world_coord_lst}")

    # print(f"0 tvecs {tvecs} rvecs {rvecs}")
