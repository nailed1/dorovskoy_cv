import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as measure

image = np.load("/Users/nailed1/Documents/GitHub/dorovskoy_cv/trajectory/out/h_96.npy")

labeled = measure.label(image, background=0)
max_label = labeled.max()

def area(labeled, label):
    return (labeled == label).sum()

areas = []
for i in range(1, max_label + 1):
    S = area(labeled, i)
    areas.append(S)

print(len(areas))

def centroid(labeled, label):
    coords = np.argwhere(labeled == label)
    if coords.size > 0:
        return coords.mean(axis=0)
    else:
        return None

centroids = []
for i in range(1, max_label + 1):
    center = centroid(labeled, i)
    if center is not None:
        centroids.append(center)

print(centroids)
def distance_to_last(i):
    target_point = centroids[i]

    dy = last[0] - target_point[0]
    dx = last[1] - target_point[1]
    dx_squared = dx ** 2
    dy_squared = dy ** 2

    sum_of_squares = dy_squared + dx_squared

    distance = sum_of_squares ** 0.5

    return distance

visited = [0]
while len(visited) < len(centroids):
    last = centroids[visited[-1]]

    possible_indices = []
    for i in range(len(centroids)):
        if i not in visited:
            possible_indices.append(i)

    best_index_so_far = possible_indices[0]
    smallest_distance_so_far = distance_to_last(best_index_so_far)

    for i in possible_indices[1:]:
        current_distance = distance_to_last(i)
        if current_distance < smallest_distance_so_far:
            smallest_distance_so_far = current_distance
            best_index_so_far = i

    next_i = best_index_so_far

    visited.append(next_i)


trajectory = np.array([centroids[i] for i in visited])

plt.imshow(image)
plt.plot(trajectory[:, 1], trajectory[:, 0], 'r-o', linewidth=2, markersize=4)
plt.show()