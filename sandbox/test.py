from scipy.datasets import face
import matplotlib.pyplot as plt
import numpy as np


def convolve(image, kernel):
    h, w = kernel.shape[0] // 2, kernel.shape[1] // 2
    output = np.zeros_like(image).astype("f4")
    
    for y in range(1, image.shape[0] - h):
        for x in range(1, image.shape[1] - w):
            sub = image[y - h:y + h + 1, x - w:x + w + 1]
            result = (sub * kernel).sum()
            result /= np.abs(kernel).sum()
            output[y, x] = result
    
    return output


# Different kernel examples (only the last one is used)
kernel = np.zeros((3, 3))
kernel[1, 1] = 1

kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

kernel = np.array([[1, 1, 1],
                   [1, 0, -1],
                   [-1, -1, -1]]) * 5

# Load image and apply convolution
image = face(gray=True)
convolved = convolve(image, kernel)

# Display result
plt.imshow(convolved, cmap="gray")
plt.show()