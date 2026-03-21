import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as measure

def area(labeled, label):
    return (labeled == label).sum()

image = np.load("coins.npy")

labeled = measure.label(image, background=0)
max_label = labeled.max()

areas = []
for i in range(1, max_label + 1):
    S = area(labeled, i)
    areas.append(S)

areas_unique = np.unique(areas)
areas_sorted = np.sort(areas_unique)

values = np.array([1, 2, 5, 10], dtype=int)

area_to_value = dict(zip(areas_sorted, values))

nominals = [area_to_value[S] for S in areas]

nominals = np.array(nominals)
total_sum = int(nominals.sum())
print("Сумма монет:", total_sum)

plt.imshow(image, cmap='flag')
plt.show()
