import numpy as np
# 1. three unit axes of the object frame, expressed in world frame
obj_x_axis = [0, -1, 0, 0]
obj_y_axis = [-1, 0, 0, 0]
obj_z_axis = [0, 0, -1, 0]

# 2. the origin of object frame, expressed in world frame
obj_origin = [200, 300, 0, 1]
chessboard_bottom_to_ground = 137
obj_origin[1] += chessboard_bottom_to_ground


# 1. first, get obj_pts to world coords transform matrix
def calc_obj_pts_to_world_coords():
    global obj_x_axis, obj_y_axis, obj_z_axis, obj_origin
    obj_pts_to_world_coords = np.matrix(
        [obj_x_axis, obj_y_axis, obj_z_axis, obj_origin]).transpose()
    return obj_pts_to_world_coords


#  calc_obj_pts_to_world_coords()
def calc_world_pts_to_obj_coords():
    return np.linalg.inv(calc_obj_pts_to_world_coords())


world_pts_to_obj_coords = calc_world_pts_to_obj_coords()
import json

world_pts_to_obj_coords_str = json.dumps(world_pts_to_obj_coords.tolist())
print(f"world pts to obj coords\n{world_pts_to_obj_coords}")
print(f"world pts to obj coords json\n{world_pts_to_obj_coords_str}")