import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops
from skimage.morphology import erosion

image = imread("balls/balls.png")
hsv = rgb2hsv(image)
hue = hsv[:, :, 0]

result = {}
total_balls = 0

for color in np.unique(hue):
    if color == 0.0:
        continue
    
    binary = hue == color
    labeled = label(binary)
    count = np.max(labeled)
    
    if count == 0:
        continue
    
    merged = False
    for key in list(result.keys()):
        delta = abs(color - key)
        if delta < 0.1:
            result[key] += count
            merged = True
            break
    
    
    if not merged:
        result[color] = count
    
    total_balls += count

for val, count in result.items():
    print(f"Цвет (hue {val:.2f}): {count} шт.")
print(f"Всего шариков: {total_balls}")