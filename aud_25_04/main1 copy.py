import cv2
import numpy as np

image = cv2.imread("aud_25_04/contours/gears.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(hierarchy)
cv2.drawContours(image, contours, -1, (0, 255, 0), 6)

cv2.imshow("Gears", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
