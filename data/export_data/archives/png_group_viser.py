import os
import matplotlib.pyplot as plt
from PIL import Image

mesh_png_dir = "500_to_5000_mesh_gen\mesh1"
png_group_lst = []
for subdir in os.listdir(mesh_png_dir):
    subdir_full = os.path.join(mesh_png_dir, subdir)
    if os.path.isdir(subdir_full):
        png_files_dir = os.path.join(subdir_full, "cam0")
        png_group_lst.append([
            os.path.join(png_files_dir, i) for i in os.listdir(png_files_dir)
        ])

num_groups = len(png_group_lst)
num_views_per_group = len(png_group_lst[0])

for g_idx in range(num_groups):
    for v_idx in range(num_views_per_group):
        ax = plt.subplot(num_groups, num_views_per_group,
                         g_idx * num_views_per_group + v_idx + 1)
        png_filepath = png_group_lst[g_idx][v_idx]
        ax.imshow(Image.open(png_filepath))

plt.show()