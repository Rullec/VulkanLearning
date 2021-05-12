import numpy as np

dist = 0.6
y = 0.45
num_of_views = 360
gap = 2 * np.pi / num_of_views

cont = []


def reduce(num):
    return float("{:.5f}".format(num))


for i in range(num_of_views):
    cur_theta = gap * i
    x = reduce(np.cos(cur_theta) * dist)
    z = reduce(-np.sin(cur_theta) * dist)
    # res = np.array()
    cont.append([x, y, z])
print(cont)