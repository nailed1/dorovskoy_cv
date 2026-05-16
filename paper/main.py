import cv2
import numpy as np
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
socket.connect("tcp://84.237.21.36:6002")
cv2.namedWindow("Stream", cv2.WINDOW_GUI_EXPANDED)
count = 0

TEXT = "Dorovskoy" 


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_paper(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:5]:
        if cv2.contourArea(cnt) < 5000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def render_text_on_paper(frame, corners):
    corners = order_points(corners.astype("float32"))
    tl, tr, br, bl = corners

    width  = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if width < 10 or height < 10:
        return frame

    sheet = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(width, height) / 150
    thickness = max(2, int(font_scale * 2))
    (tw, th), _ = cv2.getTextSize(TEXT, font, font_scale, thickness)
    x = (width - tw) // 2
    y = (height + th) // 2
    cv2.putText(sheet, TEXT, (x, y), font, font_scale, (0, 0, 180), thickness, cv2.LINE_AA)

    src_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    M = cv2.getPerspectiveTransform(src_pts, corners)
    warped = cv2.warpPerspective(sheet, M, (frame.shape[1], frame.shape[0]))

    text_mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
    frame[text_mask] = warped[text_mask]

    for pt in corners.astype(int):
        cv2.circle(frame, tuple(pt), 6, (0, 255, 0), -1)

    return frame


while True:
    msg = socket.recv()
    print(len(msg))
    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    count += 1
    frame = cv2.imdecode(np.frombuffer(msg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        continue

    corners = find_paper(frame)
    if corners is not None:
        frame = render_text_on_paper(frame, corners)

    cv2.putText(frame, f"Count: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Stream", frame)
