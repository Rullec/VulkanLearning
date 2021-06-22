from file_util import load_pkl, save_pkl, not_clear_and_create_dir, get_basename
from drawer_util import DynaPlotter, calculate_subplot_size
import os
from copy import deepcopy
import numpy as np


def run_median_filter(raw_image, size=5):
    from scipy.ndimage import median_filter
    zero_mask = raw_image == 0

    new_image = deepcopy(raw_image)
    # exit()
    filtered_image = median_filter(raw_image, size)

    new_image = new_image.astype(np.float32)
    filtered_image = filtered_image.astype(np.float32)

    # new_image[np.where(zero_mask == True)] = filtered_image[np.where(
    #     zero_mask == True)]
    new_image[zero_mask] = filtered_image[zero_mask]
    return new_image


def run_median_filter_till_nohole(raw_image, debug_draw=False):
    img_lst = []
    cur_image = raw_image
    img_lst.append(cur_image)
    max_iters = 100
    while np.min(cur_image) == 0 and len(img_lst) < max_iters:
        cur_image = run_median_filter(cur_image, size=10)
        img_lst.append(cur_image)
        # print(f"iters {len(img_lst)}")
    if debug_draw == True:
        rows, cols = calculate_subplot_size(len(img_lst) + 1)
        diff = cur_image - raw_image

        plot = DynaPlotter(rows, cols, iterative_mode=False)
        for _idx, img in enumerate(img_lst):
            plot.add(img, f"{_idx}")
        plot.add(diff, "diff")

        plot.show()
    return cur_image


if __name__ == "__main__":
    origin_dir = "cutted_dir.log"
    export_dir = "fixed_cutted_dir.log"
    not_clear_and_create_dir(export_dir)

    assert os.path.exists(origin_dir) == True
    filenames = os.listdir(origin_dir)
    images = [load_pkl(os.path.join(origin_dir, i)) for i in filenames]
    iters = 0
    for _group_idx, cur_pkl_lst in enumerate(images):
        new_img_lst = []
        rows, cols = calculate_subplot_size(3 * len(cur_pkl_lst))
        plotter = DynaPlotter(
            rows,
            cols,
            window_title="run median filter",
            iterative_mode=False,
        )
        for _idx, cur_image in enumerate(cur_pkl_lst):
            new_image = run_median_filter_till_nohole(cur_image, False)
            new_img_lst.append(new_image)
            plotter.add(cur_image, f"{_group_idx}-{_idx} raw")
            plotter.add(new_image, f"{_group_idx}-{_idx} fixed")
            plotter.add(new_image - cur_image, f"{_group_idx}-{_idx} diff")
        basename = get_basename(filenames[_group_idx])
        plotter.set_supresstitle(basename)
        plotter.show()
        output_name = f"{export_dir}/{basename}-fixed.pkl"
        save_pkl(output_name, new_img_lst)
        print(f"save pkl into {output_name}")