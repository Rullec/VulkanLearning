import numpy as np

center = (0.5, 0.5)
radius = 1.0 / 6
samples = 20
theta_unit = np.pi * 2 / samples

pts = []
for i in range(samples):
    print(
        f"[{np.cos(theta_unit * i) * radius + center[0] : 5.3f}, {np.sin(theta_unit * i) * radius  + center[1] : 5.3f}],"
    )
