import matplotlib.pyplot as plt
import numpy as np
from skimage import draw


def hist(gray):
    h = np.zeros(256, dtype="int")
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            val = gray[y, x]
            h[val] += 1
    return h

image = np.zeros((1000,1000), dtype="uint8")
image = np.random.randint(10, 75, image.shape)
ys, xs = draw.disk((500, 500), 220)
image[ys, xs] = np.random.randint(110, 160, len(ys))
ys, xs = draw.disk((800, 800), 200)
image[ys, xs] = np.random.randint(80, 100, len(xs))

threshold = 77
binary = image > threshold


plt.subplot(131)
plt.imshow(image, cmap="gray")
plt.subplot(132)
plt.plot(hist(image))
plt.subplot(133)
plt.imshow(binary)
plt.show()