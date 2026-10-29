import numpy as np
import matplotlib.pyplot as plt

def translate(image, v):
    result = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            new_x = x + v[1]
            new_y = y + v[0]
            if(new_x < image.shape[1] and new_x >=0 and new_y < image.shape[0] and new_y >= 0):
                result[new_y, new_x] = image[y, x]
    return result

def dilation(image, struct):
    result = np.zeros_like(image)
    for y in range(image.shape[0]-1):
        for x in range(image.shape[1]-1):
            s = np.logical_and(image[y, x], struct)
            result[y-1:y+2, x-1:x+2] = np.logical_or(result[y-1:y+2, x-1:x+2], s)

arr = np.zeros((30, 30))
arr[10: -10, 10: -10] = 1
arr[10, 15] = 0
arr[15, 9] = 1
arr[14, 8] = 1

v = np.ones((5,5))
arr = translate(arr, (5,5))

plt.imshow(dilation(arr, np.ones((3,3))))
plt.show()