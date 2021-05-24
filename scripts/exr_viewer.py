import OpenEXR, Imath
import numpy as np

file = r"D:\SimpleClothSimulator\data\export_data\test_geodata_gen\0.exr"

pt = Imath.PixelType(Imath.PixelType.FLOAT)
golden = OpenEXR.InputFile(file)
dw = golden.header()['dataWindow']
size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
redstr = golden.channel('Z', pt)

red = np.fromstring(redstr, dtype=np.float32)
red.shape = (size[1], size[0])  # Numpy arrays are (row, col)
# print(f"size {size}")
# print(f"red shape {red.shape}")
print(f"the first value = {red[0, 0]}")

import matplotlib.pyplot as plt
plt.imshow(red)
plt.show()