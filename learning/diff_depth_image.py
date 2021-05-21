import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


def load_capture_depth_image():
    depth_dir = "depths.log/"
    files = [os.path.join(depth_dir, i) for i in os.listdir(depth_dir)]
    load_depth_image = None
    for pkl in files:
        with open(pkl, "rb") as f:
            cont = pickle.load(f)
            if load_depth_image is None:
                load_depth_image = cont
            else:
                load_depth_image += cont

    load_depth_image /= len(files)
    assert load_depth_image.shape == (480, 640)
    load_depth_image = load_depth_image[:, 80:560]
    from PIL import Image
    load_depth_image = np.array(Image.fromarray(load_depth_image).resize((512, 512)))
    return load_depth_image


def load_cast_depth_image():
    path = r"D:\SimpleClothSimulator\data\export_data\test_geodata_gen\0.png"
    from PIL import Image
    image = np.array(Image.open(path), dtype=np.float32)
    image = np.mean(image, axis=2)
    image /= 200 # convert to m
    image *= 1000 # convert to mm

    # read the value, divide it by 200
    # print(image.shape)
    return image


capture_depth_image = load_capture_depth_image()
cast_depth_image = load_cast_depth_image()

print(f"capture_depth_image shape {capture_depth_image.shape}")
print(f"cast_depth_image shape {cast_depth_image.shape}")
plt.subplot(1, 3, 1)
plt.imshow(capture_depth_image)
plt.title("capture_depth_image")
plt.subplot(1, 3, 2)
plt.imshow(cast_depth_image)
plt.title("cast_depth_image")
plt.subplot(1, 3, 3)
plt.imshow(cast_depth_image - capture_depth_image)
plt.title("diff")
plt.show()
