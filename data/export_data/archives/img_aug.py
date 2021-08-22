from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

origin_dir = "test_geodata_gen"
target_dir = "test_geodata_gen_aug"


# 1. test global depth noise
def add_global_offset(offset_range, samples, raw_image_path, target_dir):
    # 1. load the png file. extrace per pixels to numpy
    raw_img = np.array(Image.open(raw_image_path), dtype=np.float32)[:, :, :3]
    raw_img = np.mean(raw_img, axis=2)

    # 2. convert them to float depth (meter) by divide 255.99
    import copy
    raw_img /= 255.99
    # for loop
    basename = os.path.split(raw_image_path)[-1][:-4]

    for i in range(samples):
        new_name = os.path.join(target_dir, f"{basename}_{i}.png")
        # 3. calc a random offset and apply
        cur_off = np.random.uniform(offset_range[0], offset_range[1])
        # cur_off = 0
        new_img = copy.deepcopy(raw_img)
        new_img += cur_off

        # 4. convert back to the png, save
        new_img = Image.fromarray((new_img * 255.99).astype(np.int8), 'L')
        new_img.save(new_name)
        print(f"save to {new_name}")
        # print(cur_off)

        # print(new_name)


def apply_lackness_noise(raw_img):
    import cv2 as cv

    def find_edge(raw_png):
        edge = cv.Canny(raw_png, 50, 100)

        # edge =  cv.Laplacian(raw_png, cv.CV_16S, ksize = 3)
        return edge

    raw_png = cv.imread("screen.png")
    # raw_png = cv.imread("test_geodata_gen_aug/0_0.png")
    raw_png = raw_png[:, :raw_png.shape[0]]
    raw_png = cv.cvtColor(raw_png, cv.COLOR_BGR2GRAY)
    import matplotlib.pyplot as plt
    # edge = find_edge(raw_png)
    ax2 = plt.subplot(1, 3, 1)
    ax2.imshow(raw_png)
    ax2.title.set_text("raw image")

    kernel = np.ones((10, 10), np.uint8)
    image_dilated = cv.dilate(raw_png, kernel=kernel)
    ax3 = plt.subplot(1, 3, 2)
    ax3.imshow(image_dilated)
    ax3.title.set_text("dilated")

    image_erosed = cv.erode(image_dilated, kernel=kernel)

    ax4 = plt.subplot(1, 3, 3)
    ax4.imshow(image_erosed)
    ax4.title.set_text("dilated & erosed")
    plt.show()

    return


if __name__ == "__main__":
    offset_range = [-0.03, 0.03]  # unit: m, global depth offset
    samples = 5
    files = [
        os.path.join(origin_dir, i) for i in os.listdir(origin_dir)
        if i.find("png") != -1
    ]
    if os.path.exists(target_dir) == False:
        os.makedirs(target_dir)

    # add_global_offset(offset_range, samples, files[0], target_dir)
    apply_lackness_noise(files)
