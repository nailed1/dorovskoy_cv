import numpy as np
from skimage.measure import label
from skimage.morphology import opening

from pathlib import Path

BASE = Path(__file__).resolve().parent

image=np.load(BASE / "stars.npy")

christ = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])

crest = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1]
])

pluses=label(opening(image,christ))
crosses=label(opening(image,crest))

res=pluses.max()+crosses.max()

print(res)
