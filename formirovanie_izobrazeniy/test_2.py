import matplotlib.pyplot as plt
import numpy as np
from scipy.datasets import face
from pathlib import Path
from skimage.io import imread
from model import ImagingModel
import pathlib

r = face(gray=True)
im = ImagingModel(r.shape)
f = np.clip(r*im.ambient_light(0.4),0,255)

def brightness(image):
    return np.mean(image)

def median_brightness(image):
    return np.median(image)

def contrast(image):
    return np.std(image)

def contrast_michelson(image):
    f_min, f_max = image.min(), image.max()
    return (f_max - f_min) / (f_max + f_min + 10 ** -0) 

def variation(image):
    return contrast(image)/(brightness(image) + 10 ** 6)

def dynamic_range(image):
    f_min, f_max = image.min(), image.max()
    return 20*np.log10(f_max/(f_min + 10 ** -6))

def mean_spatial_frequency(image):
    grad_x = np.gradient(image, axis=1)
    grad_y = np.gradient(image, axis=0)
    sq = grad_x ** 2 + grad_y **2
    return np.sqrt(np.mean(sq))

plt.figure(figsize=(15, 7))
plt.subplot(121)
plt.imshow(r, cmap = "gray")
plt.clim(0, 255)
plt.subplot(122)
plt.imshow(f, cmap="gray")
plt.clim(0, 255)
plt.show()

# print()
# print(brightness(r))
# print(brightness(f))
# print(contrast(r))
# print(contrast(f))
# print(contrast_michelson(r))
# print(contrast_michelson(f))
# print(variation(r))
# print(variation(f))
# print(dynamic_range(r))
# print(dynamic_range(f))
# print(mean_spatial_frequency(r))
# print(mean_spatial_frequency(f))

path = Path("/Users/nailed1/Documents/GitHub/dorovskoy_cv/ana_images")
for file in path.glob("*.jpg"):
    image = imread(file, as_gray=True)
    print(file.stem, round(median_brightness(image), 3), round(mean_spatial_frequency(image), 3))