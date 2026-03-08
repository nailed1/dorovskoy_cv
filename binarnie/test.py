import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.setrecursionlimit(30000)

def count_objects(binary: np.ndarray) -> float:
    unique = np.unique(binary)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError("Input array must be binary")

    external = 0
    internal = 0
    for y in range(binary.shape[0] - 1):
        for x in range(binary.shape[1] - 1):
            q = binary[y : y + 2, x : x + 2]
            q_sum = int(q.sum())
            match q_sum:
                case 1:
                    external += 1
                case 3:
                    internal += 1
                case 2:
                    if (
                        np.array_equal(q, np.array([[1, 0], [0, 1]]))
                        or np.array_equal(q, np.array([[0, 1], [1, 0]]))
                    ):
                        external += 2
    return (external - internal)/4

path = Path("/Users/nailed1/Documents/GitHub/dorovskoy_cv/npy-images")

def neighbours(y, x):
    return((y, x-1), (y, x+1), (y-1, x), (y+1, x))

def fill(labeled, lable, y, x):
    labeled[y, x] = lable
    for ny, nx in neighbours(y, x):
        if labeled[ny, nx] == -1:
            fill(labeled, lable, ny, nx)

def recursive_lable(binary):
    labeled = binary*-1
    lable = 0
    for y in range(labeled.shape[0]):
        for x in range(labeled.shape[1]):
            if labeled[y, x] == -1:
                lable += 1
                fill(labeled, lable, y, x)
    return labeled

image = np.load(path / "ex4.npy")


# count= 0
# for i in range(image.ndim):
#     count += count_objects(image[:, :, i])
# print(count)
labeled = recursive_lable(image)

plt.imshow(labeled)
plt.show()