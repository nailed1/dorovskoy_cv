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

clicked = False
position = None
click_processed = False

def on_click(event, x, y, flags, params):
    global position, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Клик: ({x}, {y})")
        position = [x, y]
        clicked = True 

cv2.setMouseCallback("Image", on_click)
cam = cv2.VideoCapture(0)

task = ["green", "red", "blue"]
random.shuffle(task)
user_input = list()
print(f"Задание: {task}")

calib_idx = 0
calibrated_bounds = {}
mode = "calib"
recognition_done = False
print(f"Кликните на шарик цвета '{task[calib_idx]}' для калибровки...")

if config_path.exists():
    with config_path.open("r") as f:
        js = json.load(f)
        for color, (l, u) in js.items():
            calibrated_bounds[color] = (np.array(l, dtype="u1"), np.array(u, dtype="u1"))
    print("Загружена предыдущая калибровка")

positions = list()
prev_time = time.time()
d = 6.36

while cam.isOpened():
    ret, frame = cam.read()
    if not ret: 
        break
        
    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        mode = "calib"
        calib_idx = 0
        calibrated_bounds = {}
        recognition_done = False
        user_input = []
        print("Калибровка сброшена")
        print(f"Кликните на шарик цвета '{task[calib_idx]}' для калибровки...")

    if mode == "calib" and clicked:
        clicked = False
        color_name = task[calib_idx]
        
        if position[1] < hsv.shape[0] and position[0] < hsv.shape[1]:
            h, s, v = hsv[position[1], position[0]].astype(np.float32)
            
            lower = np.clip(np.array([h*0.8, s*0.8, v*0.8]), 0, 255).astype("u1")
            upper = np.clip(np.array([h*1.2, s*1.2, v*1.2]), 0, 255).astype("u1")
            
            lower[1] = max(50, lower[1])
            upper[1] = 255
            lower[2] = max(50, lower[2])
            upper[2] = 255
            
            calibrated_bounds[color_name] = (lower, upper)
            print(f"Калибровка '{color_name}' завершена. Диапазон H: [{lower[0]}, {upper[0]}]")
            
            calib_idx += 1
            if calib_idx >= len(task):
                mode = "recognize"
                with config_path.open("w") as f:
                    json.dump({k: (v[0].tolist(), v[1].tolist()) for k, v in calibrated_bounds.items()}, f)
                print("Калибровка завершена. Покажите все три шарика в ряд слева направо.")
            else:
                print(f"Кликните на шарик цвета '{task[calib_idx]}'")
        else:
            print("Клик за пределами изображения, попробуйте снова")
    
    mask = np.zeros(hsv.shape[:2], dtype="u1")
    if calibrated_bounds:
        for l, u in calibrated_bounds.values():
            color_mask = cv2.inRange(hsv, l, u)
            mask = cv2.bitwise_or(mask, color_mask)
    
    if np.any(mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype="u1"))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype="u1"))
    
    cv2.imshow("Mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if mode == "recognize" and not recognition_done and len(contours) > 0:
        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius > 10:
                    x_int, y_int = int(x), int(y)
                    if y_int < hsv.shape[0] and x_int < hsv.shape[1]:
                        h_val = hsv[y_int, x_int][0]
                        color = "Unknown"
                        
                        if calibrated_bounds:
                            for color_name, (lower, upper) in calibrated_bounds.items():
                                if lower[0] <= h_val <= upper[0]:
                                    color = color_name
                                    break
                        else:
                            if (h_val < 10) or (h_val > 160): 
                                color = "red"
                            elif 35 < h_val < 85: 
                                color = "green"
                            elif 100 < h_val < 145: 
                                color = "blue"
                        
                        detected.append((x, y, radius, color))
                        cv2.circle(frame, (x_int, y_int), int(radius), (0, 255, 0), 3)
                        cv2.putText(frame, color, (x_int-20, y_int-20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if len(detected) >= len(task):
            detected.sort(key=lambda p: p[0])
            temp_input = [p[3] for p in detected[:len(task)]]
            
            cv2.putText(frame, f"Detected: {temp_input}", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to confirm or R to reset", (10, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if key == ord(' '):
                user_input = temp_input
                recognition_done = True
                print(f"Распознан порядок: {user_input}")
                print(f"Задание: {task}")
                if user_input == task:
                    print("Порядок правильный")
                else:
                    print("Порядок неправильный")
            elif key == ord('r'):
                print("Повторное распознавание")

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        if area > 500:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:
                x = int(x)
                y = int(y)
                radius = int(radius)
                cv2.circle(frame, (x, y), radius, (0, 255, 255), 4)
                
                if y < hsv.shape[0] and x < hsv.shape[1]:
                    h_value = hsv[y, x][0]
                    
                    ball_color = "Unknown"
                    if calibrated_bounds:
                        for color_name, (lower, upper) in calibrated_bounds.items():
                            if lower[0] <= h_value <= upper[0]:
                                ball_color = color_name
                                break
                    else:
                        if (h_value < 10) or (h_value > 160): 
                            ball_color = "red"
                        elif 35 < h_value < 85: 
                            ball_color = "green"
                        elif 100 < h_value < 145: 
                            ball_color = "blue"
                    
                    cv2.putText(frame, f"Color: {ball_color}", (10, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    
                    positions.append((x, y))
                    if len(positions) > 20:
                        positions.pop(0)

                    for i, pos in enumerate(positions[:-1]):
                        cv2.circle(frame, pos, i*2, (0, 0, 155 + 100 // len(positions) * i), -1)
                    
                    curr_time = time.time()
                    delta = curr_time - prev_time
                    if len(positions) >= 2 and delta > 0:
                        curr_pos = positions[-1]
                        prev_pos = positions[-2]
                        dst = dist(curr_pos, prev_pos)
                        pxl_per_cm = d / (2 * radius) if radius > 0 else 0
                        pxl_per_m = pxl_per_cm / 100
                        speed = (dst / delta) * pxl_per_m if pxl_per_m > 0 else 0
                        cv2.putText(frame, f"Speed = {speed:.3f} m/s", (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                    prev_time = curr_time

    if mode == "calib":
        status_text = f"Calibration: click on {task[calib_idx]} ball"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif mode == "recognize" and not recognition_done:
        cv2.putText(frame, "Show all 3 balls left to right", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif recognition_done:
        result_text = "Correct" if user_input == task else "Wrong"
        cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if user_input == task else (0, 0, 255), 2)

    cv2.imshow("Image", frame)

cam.release()
cv2.destroyAllWindows()