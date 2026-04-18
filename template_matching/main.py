import cv2
import time
import numpy as np

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cam.set(cv2.CAP_PROP_EXPOSURE, 0)

# prev_time = time.perf_counter()
background = None
roi = None
while cam.isOpened():
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('b'):
        x, y, w, h = cv2.selectROI("ROI", gray)
        roi = gray[y:y+h, x:x+w]
        cv2.imshow("Template",  roi)
        cv2.destroyWindow("ROI")
    if roi is not None:
        result = cv2.matchTemplate(gray, roi, cv2.TM_CCORR_NORMED)
        # cv2.imshow("CORR", result)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        tl = max_loc
        br = (tl[0] + roi.shape[1], tl[1] + roi.shape[0])
        cv2.rectangle(frame, tl, br, (0, 0, 255), 2)
    cv2.imshow("Camera", frame)
    # curr_time = time.perf_counter()
    # print(f"FPS = {1/(curr_time - prev_time)}:.1f")
    # prev_time = curr_time
cam.release()
cv2.destroyAllWindows()