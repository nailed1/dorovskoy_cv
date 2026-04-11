import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

def classificator(region):
    holes = count_holes(region)
    if holes == 2: #8, B
        vertical_lines = (np.sum(region.image, 0) == region.image.shape[0]).sum()
        
        if vertical_lines > 4:
            return "B"
        else:
            return "8"
    elif holes == 1: #A, 0
        labeled = label(np.logical_not(region.image))
        bays = 0
        for r in regionprops(labeled):
            if r.area > 3:
                bays +=1
        if region.eccentricity > 0.58 and bays == 5:
            return "0"
        else:
            return "A"
    else: #1, W, X, *, -, /
        if region.image.sum() / region.image.size == 1.0:
            return "-"
        shape = region.image.shape
        aspect = np.min(shape) / np.max(shape)
        if aspect >= 0.91:
            return "*"
        vertical_lines = (np.sum(region.image, 0) == region.image.shape[0]).sum()
        horizontal_lines = (np.sum(region.image, 1) == region.image.shape[1]).sum()
        if vertical_lines > 1 and horizontal_lines > 1:
            return "1"
        labeled = label(np.logical_not(region.image))
        bays = 0
        for r in regionprops(labeled):
            if r.area > 3:
                bays +=1
        if bays == 2:
            return "/"
        elif bays == 4:
            return "X"
        elif bays == 5:
            return "W"
    if vertical_lines > 1:
        return "A"
    else:
        return "?"


def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1

save_path = Path(__file__).parent

image = imread("vector_recognition/alphabet.png")[:, :, :-1]
binary_alphabet = image.mean(2) > 0
labeled_alphabet = label(binary_alphabet)
print(np.max(labeled_alphabet))
alphabet_props = regionprops(labeled_alphabet)

result = {}
image_path = save_path / "out_tree"
image_path.mkdir(exist_ok=True)

plt.ion()
plt.figure(figsize=(5, 7))

for region in alphabet_props:
    symbol = classificator(region)
    if symbol not in result:
        result[symbol]=0
    result[symbol] += 1
    plt.cla()
    plt.title(f"Class -  '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")
print(result)
print(f"{1.0 - result.get('?', 0) / len(alphabet_props)}")