import numpy as np
import matplotlib.pyplot as plt

image = np.load("/Users/nailed1/Documents/GitHub/dorovskoy_cv/stars/stars.npy")

def area(labeled, label):
    return (labeled == label).sum()

def neighbours4(y ,x):
    return (y, x+1), (y+1, x), (y-1, x), (y, x-1)

def neighboursX(y, x):
    return (y-1, x+1), (y+1, x+1), (y+1, x-1), (y-1, x-1)

def neighbours8(y, x):
    return neighbours4(y, x) + neighboursX(y, x)

def centroid(labeled, label=1):
    ys, xs = np.where(labeled == label)
    return np.mean(ys), np.mean(xs)

def is_star(neighboursX):
    return all(n == 1 for n in neighboursX)

def is_cross(neighbours4):
    return all(n == 1 for n in neighbours4)

# Label connected components
from scipy.ndimage import label

labeled, num_features = label(image)

star_count = 0
cross_count = 0

for label_id in range(1, num_features + 1):
    ys, xs = np.where(labeled == label_id)
    cy, cx = int(np.round(np.mean(ys))), int(np.round(np.mean(xs)))
    
    # Get neighbors at centroid
    n4_coords = neighbours4(cy, cx)
    nX_coords = neighboursX(cy, cx)
    
    neighbours4_vals = [labeled[y, x] == label_id if 0 <= y < labeled.shape[0] and 0 <= x < labeled.shape[1] else False 
                        for y, x in n4_coords]
    neighboursX_vals = [labeled[y, x] == label_id if 0 <= y < labeled.shape[0] and 0 <= x < labeled.shape[1] else False 
                        for y, x in nX_coords]
    
    if is_star(neighboursX_vals):
        star_count += 1
    elif is_cross(neighbours4_vals):
        cross_count += 1

print(f"Stars: {star_count}, Crosses: {cross_count}")

plt.imshow(image)
plt.show()