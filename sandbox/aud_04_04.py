import cv2
import time

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

cam = cv2.VideoCapture(0)

prev_time = time.perf_counter()

while cam.isOpened():
    ret, frame = cam.read()
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    curr_time = time.perf_counter()
    print(f"FPS = {1/(curr_time - prev_time)}:.1f")
    prev_time = curr_time
cam.release()
cv2.destroyAllWindows()