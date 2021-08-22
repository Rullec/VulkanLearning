import os

data = "fake_linear_data"
if os.path.exists(data) is False:
    os.makedirs(data)

st = 500
ed = 2000
steps = 20
noised = 20
dims = 30000


def gen_feature(dims, num):
    import numpy as np
    return np.random.rand(dims) * 100 + num


from tqdm import tqdm

data_id = 0
for cur_feature_value in tqdm(range(st, ed, int((ed - st) / steps))):
    for _ in range(noised):
        cont = {}
        cont["input"] = list(gen_feature(dims, cur_feature_value))
        cont["output"] = [500, 500, cur_feature_value]
        # print(cont["output"])
        import json
        with open(os.path.join(data, f"{data_id}.json"), 'w') as f:
            json.dump(cont, f, indent=True)
        data_id += 1
