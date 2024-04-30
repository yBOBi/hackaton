import numpy as np
import random as r
p0 = [[]]*20
x_coords = np.random.uniform(low=0, high=1080, size=(20, 1))
y_coords = np.random.uniform(low=0, high=720, size=(20, 1))
p0 = np.concatenate((x_coords, y_coords), axis=1).reshape(-1, 1, 2).astype(np.float32)
print(p0)