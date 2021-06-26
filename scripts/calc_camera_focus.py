
import numpy as np

# from calib_axon import calculate_transform_matrix_for_obj_points_to_camera_coords

trans_mat_str = '''
[[  0.9999532   -0.00249152  -0.00934514   1.87508856]
 [ -0.00479696  -0.96678903  -0.25553055 428.42519843]
 [ -0.0083981    0.25556343  -0.96675575 559.07583299]
 [  0.           0.           0.           1.        ]]
'''



if __name__ == "__main__":
    # 1. string to mat
    np.set_printoptions(suppress=True)
    
    # print(np.matmul(mat, np.transpose(mat)))
    # print(f"rotmat \n{cam_rotmat}")
    # print(f"camera pos {cam_pos}")
    # print(
    #     calc_two_line_nearest_points(np.array([0, 0, 0]), np.array([1, 1, 0]),
    #                                  np.array([0, 0, 0]), np.array([1, 1, 0])))
