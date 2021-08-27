import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os


def get_subdirs(path):
    return [
        os.path.join(path, i) for i in os.listdir(path)
        if os.path.isdir(os.path.join(path, i)) == True
    ]


def show_images(image_batch):
    print(image_batch.shape)
    show_size = 4
    # exit()
    rows = max(int(np.sqrt(show_size)), 1)
    cols = max(int(show_size / rows), 1)

    for i in range(show_size):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(np.array(torch.squeeze(image_batch[0][i]).cpu()))
    plt.show()


def torch_dataloader_iteration_example(dali_iter):
    iters = 0
    total_st = time.time()
    st = time.time()
    # while True:
    for _idx, data in enumerate(dali_iter):
        # input, output = data
        # print(f"{input.shape}")
        # print(f"{output.shape}")
        ed = time.time()
        print(f"cost0 {ed - st}")
        st = time.time()
        iters += 1
    total_ed = time.time()
    print(f"avg cost0 {(total_ed - total_st) / iters}")
    print(f"total cost0 {total_ed - total_st}")

    for _idx, data in enumerate(dali_iter):
        # input, output = data
        # print(f"{input.shape}")
        # print(f"{output.shape}")
        ed = time.time()
        print(f"cost1 {ed - st}")
        st = time.time()
        iters += 1
    total_ed = time.time()
    print(f"avg cost1 {(total_ed - total_st) / iters}")
    print(f"total cost1 {total_ed - total_st}")


def split_dataset(data_root_dir, split_perc):
    assert split_perc > 0 and split_perc < 1
    print(f"data root dir {data_root_dir}")
    all_lst = []
    for subdir in get_subdirs(data_root_dir):
        all_lst.append(subdir)
    perm = np.random.permutation(len(all_lst))
    train_id = perm[:int(split_perc * len(all_lst))]
    test_id = perm[int(split_perc * len(all_lst)):]
    train_dirs = []
    test_dirs = []
    for i in range(len(perm)):
        assert (i in train_id) != (i in test_id)
        if i in train_id:
            train_dirs.append(all_lst[i])
        else:
            test_dirs.append(all_lst[i])
    assert len(train_dirs) != 0, f"train dir is empty"
    assert len(test_dirs) != 0, f"test dir is empty"
    # print(f"train dir {train_dirs}")
    # print(f"test dir {test_dirs}")
    return train_dirs, test_dirs
