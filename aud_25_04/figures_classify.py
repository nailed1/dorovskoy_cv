import cv2
import numpy as np

image = cv2.imread("aud_25_04/contours/cubes_1.png")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)

position = [0, 0]
clicked = False
def on_click(event, x , y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at {x}, {y}")
        global position
        global clicked
        position = [x, y]
        clicked = True

def classify(contour):
    verts = -1
    solidity = -1
    approx = []
    figure = "None"
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return verts, approx, solidity, figure

    eps = 0.1 * perimeter
    approx = cv2.approxPolyDP(contour, eps, True)
    verts = len(approx)

    _, radius = cv2.minEnclosingCircle(contour)
    area = cv2.contourArea(contour)
    circle_area = np.pi * radius ** 2
    solidity = area/circle_area

    if solidity > 0.8:
        figure = "sphere"
    elif verts == 3:
        figure = "triangle"
    elif verts == 4:
        figure = "cube  "

    return verts, approx, solidity, figure
cv2.setMouseCallback("Image", on_click)
mask = np.zeros(image.shape[:-1], dtype="u1")
while True:
    display_image = image.copy()
    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        mask[:] = 0
    if clicked:
        clicked = False
        color = hsv[position[1], position[0]]
        print(color)
        lower = np.clip(color * 0.9, 0, 255).astype("u1")
        upper = np.clip(color * 1.1, 0, 255).astype("u1")
        inr = cv2.inRange(hsv, lower, upper)
        inr = cv2.morphologyEx(inr, cv2.MORPH_CLOSE, np.ones((5, 5), dtype="u1"))
        mask = cv2.bitwise_or(mask, inr)
        cv2.imshow("Mask", mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contours = list(filter(
            lambda contour: cv2.contourArea(contour) > 2000,
            contours
        ))
        cv2.drawContours(display_image, contours, -1, (0, 255, 0), 4)
        for i, contour in enumerate(contours):
            (verts, approx, solidity, figure) = classify(contour)
            top_idx = np.argmin(contour[:, 0, 1])
            top_point = tuple(contour[top_idx, 0])
            thickness = 2
            font_scale = 1.5

            text = f"{figure}({verts}, {solidity:.1f})"
            font = cv2.FONT_HERSHEY_COMPLEX
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = max(10, top_point[0]-text_w//2)
            text_y = max(text_h + 10, top_point[1] - 15)

            cv2.rectangle(display_image, (text_x, text_y - text_h - 4), (text_x + text_w, text_y + 4), (0, 0, 0), -1)
            cv2.putText(display_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            if len(approx) > 0:
                for p in approx:
                    cv2.circle(display_image, p[0], 10, (255, 0, 0), -1)
    cv2.imshow("Image", display_image)
cv2.destroyAllWindows