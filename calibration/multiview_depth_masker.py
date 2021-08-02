import numpy as np

# 1. load the depth images
# 2. set a threshold, all values bigger than that, is set to zero
# 3. dobule check the result. if the resulting images cannot meet our need, manually set the result
from matplotlib.pyplot import plot
import numpy as np
from copy import deepcopy
import os
from file_util import load_pkl, save_pkl, not_clear_and_create_dir, get_basename
from drawer_util import DynaPlotter

if __name__ == "__main__":
    origin_dir = "fixed_cutted_dir.log"
    output_dir = "no_background_dir.log"
    assert os.path.exists(origin_dir) == True
    not_clear_and_create_dir(output_dir)

    files = [os.path.join(origin_dir, i) for i in os.listdir(origin_dir)]
    threshold = 670
    # 1. begin to load the pkl files
    for _idx in range(len(files)):
        depth_image = load_pkl(files[_idx])

        basename = get_basename(files[_idx])
        new_img_lst = []
        for i in range(len(depth_image)):
            old_image = depth_image[i]
            plot = DynaPlotter(1, 3, iterative_mode=False)
            plot.set_supresstitle(f"{basename}-view{i}")
            plot.add(old_image, f"raw")
            new_image = deepcopy(old_image)
            new_image[np.where(new_image > threshold)] = 0
            plot.add(new_image, f"masked thre {threshold}")
            diff = new_image - old_image
            new_img_lst.append(new_image)
            plot.add(diff, f"diff")
            plot.show()
        name = f"{output_dir}//{basename}-masked.pkl"
        save_pkl(name, new_img_lst)