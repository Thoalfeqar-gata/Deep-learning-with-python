import numpy as np

#broadcasting example
x = np.random.random((3, 4, 3))
y = np.random.random((4, 3))
z = np.maximum(x, y)
print(f'x: {x}\n {x.shape}', f'y: {y}\n {y.shape}', f'z: {z}\n {z.shape}', sep = '\n\n')


