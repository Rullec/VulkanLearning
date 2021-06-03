import os
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
# new_dir = "./test_geodata_gen/"
new_dir = "./500_to_5000_mesh_gen_small/"
# dest_dir = "./test_small_data/"

png_files = [i for i in os.listdir(new_dir) if i.find("png") != -1]


# for png in tqdm(png_files):
#     png = os.path.join(dir, png)
#     old_image = Image.open(png)
#     new_image = old_image
#     # old_image.show()
#     new_image = new_image.resize((512, 512))
#     # new_image.show()
#     new_image.save(png)
#     # exit()
def handle(file):
    try:
        image = Image.open(file)
        # print(image.format)
        # print(image.mode)
        image = image.convert('L')
        height, width = image.size
        # image = image.crop((height / 4, width / 4, 3 * height / 4, 3 * width / 4))
        # print(image.size)
        if image.size[1] > 200:
            image = image.resize(
                (int(image.size[0] / 2), int(image.size[1] / 2)))
        image.save(file)
    except Exception as e:
        pass


# with Pool(8) as p:
if __name__ == "__main__":
    all_files = [os.path.join(new_dir, png) for png in png_files]
    with Pool(6) as p:
        r = list(tqdm(p.imap(handle, all_files), total=len(all_files)))
