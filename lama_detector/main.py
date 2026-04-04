import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.filters import sobel
from skimage.morphology import binary_closing
import sys

sys.setrecursionlimit(1500000)

image = imread("./lama_on_moon.png")[32:-50, 40:-20, :-1]

print(image.shape)

gray = image.mean(2)
contours = sobel(gray)
binary = contours > 15
binary = binary_closing(binary, footprint=np.ones((3,3)))
labeled = label(binary)

regions = regionprops(labeled)
regions = sorted(regions, key=lambda region: region.perimeter)
region = regions[-1]

print(region.perimeter)

# for region in regionprops(labeled):
#     if region.area < 1000:
#         rr, cc = region.coords[:, 0], region.coords[:, 1]
#         binary[rr, cc] = 0

def neighbours4(y, x):
    return (y, x+1), (y+1, x), (y-1, x), (y, x-1)


def fill(binary, y, x):
    if binary[y,x]==0:
        binary[y,x]=1
    for yn, xn in neighbours4(y, x):
        if((0<=yn<binary.shape[0]) and (0<=xn<binary.shape[1])):
            if binary[yn, xn] == 0:
                fill(binary, yn, xn)

mask = labeled == region.label
fill(mask,0,0)
mask = ~mask

gray_masked = gray * mask

plt.imshow(gray_masked, cmap='gray')
plt.show()