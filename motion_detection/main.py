import cv2
import time

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cam.set(cv2.CAP_PROP_EXPOSURE, 0)

# prev_time = time.perf_counter()
background = None
while cam.isOpened():
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('b'):
        background = gray.copy()
    if background is not None:
        delta = cv2.absdiff(background, gray)
        mask = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            area = cv2.contourArea(c)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow("Background", mask)
        cv2.imshow("Camera", frame)
    # curr_time = time.perf_counter()
    # print(f"FPS = {1/(curr_time - prev_time)}:.1f")
    # prev_time = curr_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
cam.release()
cv2.destroyAllWindows()