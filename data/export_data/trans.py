import os
from PIL import Image
from tqdm import tqdm

new_dir = "./reduced_noised_gen360/"
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

for png in tqdm(png_files):
    file = os.path.join(new_dir, png)
    image = Image.open(file)
    # print(image.format)
    # print(image.mode)
    image = image.convert('L')
    image = image.resize((128, 128))
    image.save(file)
    # print(f"save {file} done")
    # print(image.format)
    # print(image.mode)