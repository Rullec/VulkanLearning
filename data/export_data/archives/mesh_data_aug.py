import os
import numpy as np
import json

from multiprocessing import Pool

origin_dir = "500_to_5000_mesh"
target_dir = "500_to_5000_mesh_aug"
# noised_samples = 5


def handle(origin_name, iters = 5):
    global target_dir
    base_name = origin_name.split(".")[0]
    full_origin_name = os.path.join(origin_dir, origin_name)
    for _ in range(iters):
        # 1. load
        offset = (np.random.rand(3) - 0.5) / 20
        # print(f"offset {offset*100} cm")

        with open(full_origin_name) as f:
            cont = json.load(f)
        # 2. add offset
        for i in range(0, len(cont["input"]), 3):
            cont["input"][i] += offset[0]
            cont["input"][i + 1] += offset[1]
            cont["input"][i + 2] += offset[2]

        # 3. save to a new place
        new_name = f"{base_name}_{_}.json"
        new_name = os.path.join(target_dir, new_name)

        with open(new_name, 'w') as f:
            json.dump(cont, f)


# 1. load all origin names

if __name__ == "__main__":
    if os.path.exists(target_dir) == False:
        os.makedirs(target_dir)
    origin_files = [i for i in os.listdir(origin_dir) if i.find("json") != -1]
    from tqdm import tqdm

    with Pool(6) as p:
        list(tqdm(p.imap(handle, origin_files), total=len(origin_files)))
    # for ori in tqdm(origin_files):
    #     if ori.find("json") != -1:
    #         # print(f"ori {ori}")
    #         handle(ori, noised_samples)
# 2. do aug for each origin mesh