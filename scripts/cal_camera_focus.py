from scipy.spatial.transform import Rotation as R
import numpy as np

# from calib_axon import calculate_transform_matrix_for_obj_points_to_camera_coords

trans_mat_str = '''
[[  0.9999532   -0.00249152  -0.00934514   1.87508856]
 [ -0.00479696  -0.96678903  -0.25553055 428.42519843]
 [ -0.0083981    0.25556343  -0.96675575 559.07583299]
 [  0.           0.           0.           1.        ]]
'''


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


# def calc_two_line_nearest_points(line0_ori, line0_dir, line1_ori, line1_dir):
#     '''
#         calculate the nearest point pair between two given lines,
#         return: point on 0
#     '''
#     assert len(line0_ori) == 3
#     assert len(line1_ori) == 3
#     assert len(line0_dir) == 3
#     assert len(line1_dir) == 3
#     A = np.zeros([3, 3])
#     b = np.zeros(3)
#     d3 = np.cross(line0_dir, line1_dir)
#     A[:, 0] = line0_dir
#     A[:, 1] = -line1_dir
#     A[:, 2] = d3
#     b = line1_ori - line0_ori
#     det = np.linalg.det(A)
#     if np.abs(det) < 1e-10:
#         # parallel or conincidence,
#         normed_line0_dir = line0_dir / np.linalg.norm(line0_dir)
#         p0_prime = np.dot((line1_ori - line0_ori),
#                           normed_line0_dir) * normed_line0_dir + line0_ori
#         # print("[error] failed to find the nearest points in coincidence case")
#         dist = np.linalg.norm(p0_prime - line1_ori)
#         return p0_prime, line1_ori, dist
#     else:
#         t = np.linalg.solve(A, b)
#         t0, t1, t2 = t[0], t[1], t[2]
#         p0 = line0_ori + t0 * line0_dir
#         p1 = line1_ori + t1 * line1_dir
#         dist = np.linalg.norm(d3 * t2)
#         return p0, p1, dist


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


if __name__ == "__main__":
    # 1. string to mat
    np.set_printoptions(suppress=True)
    mat = extract_trans_mat(trans_mat_str)
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
    # print(np.matmul(mat, np.transpose(mat)))
    # print(f"rotmat \n{cam_rotmat}")
    # print(f"camera pos {cam_pos}")
    # print(
    #     calc_two_line_nearest_points(np.array([0, 0, 0]), np.array([1, 1, 0]),
    #                                  np.array([0, 0, 0]), np.array([1, 1, 0])))
