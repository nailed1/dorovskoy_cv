import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label

image = np.load("/Users/nailed1/Documents/GitHub/dorovskoy_cv/stars/stars.npy")

def area(labeled, label):
    return (labeled == label).sum()

def neighbours4(y, x):
    return [(y, x+1), (y+1, x), (y-1, x), (y, x-1)]

def neighboursX(y, x):
    return [(y-1, x+1), (y+1, x+1), (y+1, x-1), (y-1, x-1)]

def neighbours8(y, x):
    return neighbours4(y, x) + neighboursX(y, x)

def centroid(labeled, label=1):
    ys, xs = np.where(labeled == label)
    return np.mean(ys), np.mean(xs)

def is_star(neighboursX):
    return all(n == 1 for n in neighboursX)

def is_cross(neighbours4):
    return all(n == 1 for n in neighbours4)

labeled, num_features = label(image, return_num=True)

star_count = 0
cross_count = 0

for label_id in range(1, num_features + 1):
    ys, xs = np.where(labeled == label_id)
    cy, cx = int(np.round(np.mean(ys))), int(np.round(np.mean(xs)))
    
    height, width = labeled.shape
    n4_values = []
    for dy, dx in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
        ny, nx = cy + dy, cx + dx
        if 0 <= ny < height and 0 <= nx < width:
            n4_values.append(labeled[ny, nx])
        else:
            n4_values.append(0)
    
    nX_values = []
    for dy, dx in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
        ny, nx = cy + dy, cx + dx
        if 0 <= ny < height and 0 <= nx < width:
            nX_values.append(labeled[ny, nx])
        else:
            nX_values.append(0)
    
    if all(val == label_id for val in nX_values):
        star_count += 1
    elif all(val == label_id for val in n4_values):
        cross_count += 1

print(f"Stars: {star_count}, Crosses: {cross_count}")

plt.imshow(image)
plt.show()