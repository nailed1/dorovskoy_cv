import matplotlib.pyplot as plt
import numpy as np

def area(labeled, label):
    return (labeled == label).sum()
labeled = np.zeros((16,16), dtype="int")
labeled[4:, :4] = 1

labeled[3:10, 8:]=2
labeled[[3,4,3],[8,8,9]]=0
labeled[[8,9,9],[8,8,9]]=0
labeled[[3,4,3],[-2,-1,-1]]=0
labeled[[9,8,9],[-2,-1,-1]]=0

labeled[12:-1, 6:9]=3

for i in range(1, np.max(labeled+1)):
    print(f"Area = {area(labeled, 1)}")



plt.inshow(labeled, cmap='flag')
plt.show()