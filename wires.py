import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import binary_opening

image = np.load("/Users/nailed1/Documents/GitHub/dorovskoy_cv/wires/wires1.npy")
struct = np.ones((3, 1))
process = binary_opening(image, struct)

labeled_image = label(image)
labeled_process = label(process)
print(f"Original {np.max(labeled_image)}")
print(f"Processed {np.max(labeled_process)}")

num_wires = np.max(labeled_image)
for i in range(1, num_wires + 1):
    mask = (labeled_image == i)
    overlapping_labels = np.unique(labeled_process[mask])
    overlapping_labels = overlapping_labels[overlapping_labels != 0]
    num_segments = len(overlapping_labels)
    if(num_segments==1):
        print("провод не порван")
    else: 
        print(f"Провод {i}: частей после обработки = {num_segments}")


plt.subplot(121)
plt.imshow(labeled_image)
plt.subplot(122)
plt.imshow(process)
plt.show()