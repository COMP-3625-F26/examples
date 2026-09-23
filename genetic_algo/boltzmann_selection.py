import numpy as np

fitness = np.array([1.0, 2.0, 3.0, 5.0])
T = 0.1

p = np.exp(fitness / T) / np.exp(fitness / T).sum()

print(p)