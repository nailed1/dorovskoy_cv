import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1



def extractor(region):
    cy ,cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    perimeter = region.perimeter / region.image.size
    holes = count_holes(region)
    vertical_lines = (np.sum(region.image, 0) == region.image.shape[1]).sum()
    horizontal_lines = (np.sum(region.image, 1) == region.image.shape[0]).sum()
    eccentricity = region.eccentricity
    aspect = region.image.shape[0] / region.image.shape[1]
    area = region.area
    compactness = 4 * np.pi * area / (region.perimeter ** 2 + 1e-6)
    solidity = area / (region.area_convex + 1e-6)
    extent = area / (region.area_bbox + 1e-6)
    return np.array([
        perimeter,
        cy,
        cx,
        holes,
        vertical_lines,
        horizontal_lines,
        eccentricity,
        aspect,
        compactness,
        solidity,
        extent
    ])

def classificator(region, templates):
    features = extractor(region)
    result = ""
    min_d = 10**16
    for symbol, t in templates.items():
        d = ((t - features) ** 2).sum() ** 0.5
        if d < min_d:
            result = symbol
            min_d = d
    return result

template = imread("vector_recognition/alphabet-small.png")[:, :, :-1]
template = template.sum(2)
binary = template != 765

labeled = label(binary)
props = regionprops(labeled)

# print(type(props[0]))
# print(props[0].area, props[0].centroid, props[0].label)

templates = {}

for region, symbol in zip(props, ["8", "0", "A", "B", "1", "W", "X", "*", "/", "-"]):
    templates[symbol] = extractor(region)


image = imread("vector_recognition/alphabet.png")[:, :, :-1]
binary_alphabet = image.mean(2) > 0
labeled_alphabet = label(binary_alphabet)
print(np.max(labeled_alphabet))
alphabet_props = regionprops(labeled_alphabet)

result = {}
image_path = save_path / "out"
image_path.mkdir(exist_ok=True)

plt.ion()
plt.figure(figsize=(5, 7))

for region in alphabet_props:
    symbol = classificator(region, templates)
    if symbol not in result:
        result[symbol]=0
    result[symbol] += 1
    plt.cla()
    plt.title(f"Class -  '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")
print(result)
plt.imshow(binary_alphabet)
plt.show()