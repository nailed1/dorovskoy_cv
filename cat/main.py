import cv2
import matplotlib.pyplot as plt
import numpy as np

cat = cv2.imread("cat/cat.png")

cat1 = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)
cat2 = cv2.imread("cat/cat2.png", cv2.IMREAD_GRAYSCALE)

diff = cv2.absdiff(cat1, cat2)
_, mask = cv2.threshold(diff, 25, 255, cv2. THRESH_BINARY)

mask = cv2.dilate(mask, None, iterations=2)

contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(len(contours))

for c in contours:
    x, y, w, h =  cv2.boundingRect(c)
    cv2.rectangle(cat, (x,y), (x+w, y+h),
                  (0, 0, 255), 2)

cv2.putText(cat, f"Diffs = {len(contours)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

cv2.namedWindow("Original", cv2.WINDOW_GUI_NORMAL)
cv2.imshow("Original", cat)
cv2.waitKey(0)