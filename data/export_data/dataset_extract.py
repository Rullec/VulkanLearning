import os
from tqdm import tqdm


def extract_mesh_dataset(origin_dir, target_dir, datapoint_judge_func,
                         sort_func, noised_sample_per_prop, sample_gap_prop):
    # 1. get raw data

    # origin_data_lst = [
    #     i for i in os.listdir(origin_dir) if i.find(".json") != -1
    # ]
    origin_data_lst = []
    for i in os.listdir(origin_dir):
        if True == datapoint_judge_func(os.path.join(origin_dir, i)):
            origin_data_lst.append(i)

    origin_data_lst.sort(key=sort_func)

    if os.path.exists(target_dir) == True:
        raise ValueError(f"{target_dir} exists")

    os.makedirs(target_dir)

    ori_tar_params = []
    import shutil
    copy_func = None
    for _idx, filename in enumerate(tqdm(origin_data_lst)):
        prop_id = int(_idx / noised_sample_per_prop)
        if prop_id % sample_gap_prop == 0:
            # copy current tree (file or dir)
            origin_stuff = os.path.join(origin_dir, filename)
            target_stuff = os.path.join(target_dir, filename)
            ori_tar_params.append((origin_stuff, target_stuff))
            if os.path.isfile(origin_stuff) == True:
                assert (copy_func is None) or (copy_func == shutil.copyfile)
                copy_func = shutil.copyfile
            else:
                assert (copy_func is None) or (copy_func == shutil.copytree)
                # shutil.copytree(origin_stuff, target_stuff)
                copy_func = shutil.copytree

    print(f"begin to do {len(ori_tar_params)} copies...")
    from multiprocessing import Pool
    pool = Pool(12)
    for i in pool.starmap(copy_func, ori_tar_params):
        pass


if __name__ == "__main__":
    # mesh (json) selecting method
    # mesh_origin_dir = "isotropic_200prop_16samples_amp_0.05"
    # mesh_target_dir = "isotropic_50prop_16samples_amp_0.05"

    # def mesh_sort_func(e):
    #     return int(e[:e.find(".")])

    # def mesh_datapoint_judge_func(path):
    #     return path.find(".json") != -1

    # extract_mesh_dataset(mesh_origin_dir,
    #                      mesh_target_dir,
    #                      datapoint_judge_func=mesh_datapoint_judge_func,
    #                      sort_func=mesh_sort_func)
    noised_sample_per_prop = 16
    sample_gap_prop = 8

    depth_origin_dir = f"uniform_sample10_noised{noised_sample_per_prop}_4camnoised_2rot_4view"
    depth_target_dir = f"uniform_sample10_noised{int(noised_sample_per_prop / sample_gap_prop)}_4camnoised_2rot_4view"

    def depth_sort_func(e):
        st_int = e.find("mesh") + len("mesh")
        val = int(e[st_int:])
        return val

    def depth_datapoint_judge_func(path):
        return path.find("mesh") != -1 and os.path.isdir(path)

    extract_mesh_dataset(depth_origin_dir,
                         depth_target_dir,
                         datapoint_judge_func=depth_datapoint_judge_func,
                         sort_func=depth_sort_func,
                         noised_sample_per_prop=noised_sample_per_prop,
                         sample_gap_prop=sample_gap_prop)
