import cv2
import matplotlib.pyplot as plt
import numpy as np

tv = cv2.imread("kak_nibud/news.jpg")

chebu = cv2.imread("kak_nibud/cheburashka.jpg", cv2.IMREAD_COLOR_RGB)

rows, cols, _ = chebu.shape
pts1 = np.array([[0,0],[cols, 0],[cols,rows],[0,rows]], dtype="f4")
pts2 = np.array([[18,25], [432, 53], [435, 270], [39,294]], dtype="f4")

m = cv2.getPerspectiveTransform(pts1, pts2)
transformed = cv2.warpPerspective(chebu, m, (tv.shape[1], tv.shape[0]))

cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (cols, rows))
    
    transformed = cv2.warpPerspective(frame_resized, m, (tv.shape[1], tv.shape[0]))

    gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    mask_inv = cv2.bitwise_not(mask)

    bg_masked = cv2.bitwise_and(tv, tv, mask=mask_inv)
    
    fg = cv2.bitwise_and(transformed, transformed, mask=mask)
    
    result = cv2.add(bg_masked, fg)
    
    
    cv2.imshow("Camera in TV", result)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


