import numpy as np
import torch

def apply_mesh_data_noise(batch):
    inputs, outputs = batch
    size = inputs.shape[1]
    noise = (np.random.rand(3).astype(np.float32) - 0.5) / 10  # +-5cm
    noise_all = np.repeat([np.tile(noise, size // 3)], inputs.shape[0], axis = 0)
    inputs += noise_all
    # for _idx in range(inputs.shape[0]):
    #     # print(f"noise = {noise}")
    #     inputs[_idx] += noise_all
    return (inputs, outputs)
