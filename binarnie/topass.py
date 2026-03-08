import matplotlib.pyplot as plt
import numpy as np

def neighbours2(y ,x):
    return (y-1,x), (y, x-1)

def find(label, links):
    j = label
    while links[j] != 0:
        j = links[j]
    return j

def union(label1, label2, links):
    j = find(label1, links)
    k = find(label2, links)
    if j != k:
        links[k]=j

# def union(label1, label2, links):
#     if label1 == 0 or label2 == 0:
#         return
#     j = find(label1, links)
#     k = find(label2, links)
#     if j != k:
#         root = min(j, k)
#         other = max(j, k)
#         links[other] = root


def label(binary):
    labeled = binary * -1
    label = 0
    links = np.zeros(labeled.size // 2 + 1, dtype=("int32"))
    for y in range(1, binary.shape[0]-1):
        for x in range(1, binary.shape[1]-1):
            if labeled[y, x] == -1:
                n = neighbours2(y,x)
                n1 = labeled[n[0]]
                n2 = labeled[n[1]]
                if n1==0 and n2==0:
                    label +=1
                    lb = label
                elif n1==0 or n2==0:
                    lb = max([n1, n2])
                else:
                    lb = min([n1,n2])

                labeled[y, x] = lb
                union(*sorted([n1, lb]), links)
                union(*sorted([n2, lb]), links)

    for y in range(1, binary.shape[0]-1):
        for x in range(1, binary.shape[1]-1):
            if labeled[y,x] > 0:
                new_label = find(labeled[y,x], links)
                labeled[y, x] = new_label
    
    remap = {}
    new_label = 0
    for y in range(1, binary.shape[0]-1):
        for x in range(1, binary.shape[1]-1):
            if labeled[y, x] > 0:
                if labeled[y, x] not in remap:
                    new_label += 1
                    remap[labeled[y, x]] = new_label
                labeled[y, x] = remap[labeled[y, x]]
    
    print(links[:10])
    return labeled

image = np.load("/Users/nailed1/Documents/GitHub/dorovskoy_cv/binarnie/mark.npy")
plt.imshow(label(image))
plt.show()
