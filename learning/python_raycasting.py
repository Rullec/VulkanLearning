import process_data_scene
import numpy as np

config_path = "./config/data_process.json"
scene = process_data_scene.process_data_scene()
scene.Init(config_path)
shape = scene.GetDepthImageShape()
pos = np.array([0.00187508856, 0.42842519843, 0.55907583299, 1])
center = np.array([-0.00352920924, 0.28065162501, 0, 1])
fov = 52.5
img = scene.CalcEmptyDepthImage(pos, center, fov)
print(f"shape {shape}")
print(f"img shape {img.shape}")
img = img.reshape(shape)
print(f"img shape {img.shape}")

import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()