import matplotlib.pyplot as plt
import numpy as np
from scipy.datasets import face

def block_mean(image, x_size=20, y_size=20):
    result = np.zeros_like(image)
    for y in range(0, result.shape[0], y_size):
        for x in range(0, result.shape[1], x_size):
            sart = image[y:y+y_size, x:x+x_size]
            result[y:y+y_size, x:x+x_size] = np.min(sart)
    return result

image = face(gray=True)
plt.imshow(block_mean(image), cmap='gray')

plt.show()