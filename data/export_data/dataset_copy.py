import os

origin_dir = "500_5000_feature2_nonoise_500sample_real_copy"
files = os.listdir(origin_dir)
id_lst = []
for file in files:
    if file.find(".json") != -1:
        cur_id = int(file.split('.')[0])

        id_lst.append(cur_id)

import numpy as np

max_id = np.max(id_lst)
for _idx, file in enumerate(files):
    cur_id = max_id + _idx + 1
    new_file = f"{cur_id}.json"
    import shutil
    shutil.copy(os.path.join(origin_dir, file),
                os.path.join(origin_dir, new_file))
