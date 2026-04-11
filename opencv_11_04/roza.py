import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("opencv_11_04/rose.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower = np.array([0, 200, 100])
upper = np.array([2, 255, 255]) 

mask = cv2.inRange(hsv, lower, upper)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20,20)))
result = cv2.bitwise_and(image, image, mask=mask)

plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()