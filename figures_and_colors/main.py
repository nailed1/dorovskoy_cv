import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label,regionprops
from skimage.io import imread
from skimage.color import rgb2hsv

image = imread("balls_and_rects.png")
hsv = rgb2hsv(image)
h = hsv[:,:,0]

colors = []
for color in np.unique(h):
    binary = h == color
    labeled = label(binary)
    colors.extend([color] * np.max(labeled))

rect = []
circ = []

for color in np.unique(h):
    if color == 0: continue
    binary = (h == color)
    labeled_color = label(binary)
    
    for region in regionprops(labeled_color):
        if region.extent > 0.95:
            rect.append(color)
        else:
            circ.append(color)

print(f"Кол-во: {len(colors)-1}")

def count_fig(colors):
    groups = [[colors[0]]]
    delta = 0.05
    for i in range(1,len(colors)):
        if abs(colors[i-1] - colors[i]) < delta:
            groups[-1].append(colors[i])
        else:
            groups.append([colors[i]])
            
    for grp in groups:
        print(np.mean(grp),len(grp))

print("Прямоугольники")
count_fig(rect)
print("Круги")
count_fig(circ)

plt.subplot(121)
plt.imshow(h,cmap="gray")
plt.subplot(122)
plt.plot(colors,"o-")
plt.show()