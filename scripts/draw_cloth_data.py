import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

fig = plt.figure()
ax = Axes3D(fig)
data = r"D:\SimpleClothSimulator\data\export_data\500_to_5000_mesh\100.json"
with open(data) as f:
    cont = json.load(f)
    print(cont.keys())
    pos = cont["input"]

x_lst, y_lst, z_lst = [], [], []
for i in range(int(len(pos) / 3)):
    x_lst.append(pos[3 * i + 0])
    y_lst.append(pos[3 * i + 1])
    z_lst.append(pos[3 * i + 2])

ax.plot3D(x_lst, y_lst, z_lst)
plt.show()