import numpy as np

dist = 0.3
num_of_views = 48
gap = 2 * np.pi / num_of_views

cont = []


def reduce(num):
    return float("{:.2f}".format(num))


for i in range(num_of_views):
    cur_theta = gap * i
    x = reduce(np.cos(cur_theta) * dist)
    y = 0.12
    z = reduce(-np.sin(cur_theta) * dist)
    # res = np.array()
    cont.append([x, y, z])
print(cont)