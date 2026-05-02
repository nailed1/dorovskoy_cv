import cv2
import numpy as np
import time
from math import dist
from pathlib import Path
import json
import random

save_path = Path(__file__).parent
config_path = save_path/"config.json"

cv2.namedWindow("Image", cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_EXPANDED)

# cv2.namedWindow("Camera",cv2.WINDOW_KEEPRATIO)
clicked = False
position = None

def on_click(event, x, y, flags, params):
    global position, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        position = [x, y]
        clicked = True 

cv2.setMouseCallback("Image",on_click)
cam=cv2.VideoCapture(0)


task = ["green", "red", "blue"]
random.shuffle(task)
user_input = list()
print(f"задание: {task}")

calibration = {"green": None, "red": None, "blue": None}
if config_path.exists():
    with config_path.open("r") as f:
        calibration = json.load(f)



upper=None
lower=None

if config_path.exists():
    with config_path.open("r") as f:
        js = json.load(f)
        if "lower" in js:
            lower = np.array(js["lower"], dtype="u1")
            upper = np.array(js["upper"], dtype="u1")

positions = list()
prev_time = time.time()
curr_time = time.time()
d = 6.36 #cm


while cam.isOpened():
    ret, frame=cam.read()
    blurred = cv2.GaussianBlur(frame, (11,11),0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('g'):
        current_calibration_color = "green"
        print(f"Калибруем {current_calibration_color}. Кликните на зеленый шарик")
    elif key == ord('r'):
        current_calibration_color = "red"
        print(f"Калибруем {current_calibration_color}. Кликните на красный шарик")
    elif key == ord('b'):
        current_calibration_color = "blue"
        print(f"Калибруем {current_calibration_color}. Кликните на синий шарик")
    if clicked:
        clicked = False
        color = hsv[position[1], position[0]].astype(np.float32)  
        lower = np.clip(color * 0.9, 0, 255).astype("u1")
        upper = np.clip(color * 1.1, 0, 255).astype("u1")
        upper[1] = 255
        upper[2] = 255
    if lower is not None:
        inr = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(inr, cv2.MORPH_CLOSE, np.ones((5, 5), dtype="u1"))
        cv2.imshow("Mask", mask)
        contours, _ = cv2. findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours)) > 0:
            contour = max(contours, key = cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:
                x = int(x)
                y = int(y)
                radius = int(radius)
                cv2.circle(frame, (x, y), radius, (0, 255, 255), 4)
                h_value = hsv[y, x][0]
                
                ball_color = "Unknown"
                if (h_value < 10) or (h_value > 160):
                    ball_color = "red"
                elif 35 < h_value < 85:
                    ball_color = "green"
                elif 100 < h_value < 145:
                    ball_color = "blue"
                
                cv2.putText(frame, f"Color: {ball_color}", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                positions.append((x,y))
                if len(positions) > 20:
                    positions.pop(0)

                for i, position in enumerate(positions[:-1]):
                    cv2.circle(frame, position, i*2 ,(0, 0, 155 + 100 / len(positions) * i), -1)
                curr_time = time.time()
                delta = curr_time - prev_time
                if len(positions) >= 2:
                    curr_pos = positions[-1]
                    prev_pos = positions[-2]
                    dst = dist(curr_pos, prev_pos)
                    pxl_per_cm = d / (2*radius)
                    pxl_per_m = pxl_per_cm/100
                    speed = (dst/delta)*pxl_per_m
                    cv2.putText(frame, f"Seed = {speed:.5f}m/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0))
                    prev_pos = curr_time
    cv2.imshow("Image",frame)
cam.release()
cv2.destroyAllWindows()

with config_path.open("w") as f:
    json.dump(
        {"lower": None if lower is None else lower.tolist(), "upper":None if upper is None else upper.tolist()},
        f
    )