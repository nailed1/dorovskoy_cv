from __future__ import annotations
#claude --resume b3ac4d5c-76df-4ee0-ac40-1db6041bdfe7
import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

GRAVITY = 0.55
BALL_RADIUS = 25
RESTITUTION = 0.35
ROLLING_FRICTION = 0.96
AIR_FRICTION = 0.9995


@dataclass
class Ball:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    active: bool = False

    def reset(self, sx: float, sy: float) -> None:
        self.x = sx
        self.y = sy
        self.vx = 0.0
        self.vy = 0.0
        self.active = False


def detect_platforms(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=55, minLineLength=60, maxLineGap=20,
    )
    if lines is None:
        return []
    return [tuple(ln[0]) for ln in lines]


def _resolve_collision(
    bx: float, by: float,
    vx: float, vy: float,
    x1: int, y1: int, x2: int, y2: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Return (new_vx, new_vy, push_x, push_y) or None."""
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1:
        return None

    tx, ty = dx / length, dy / length      # unit tangent
    nx, ny = -ty, tx                        # unit normal (left-perp)

    # Closest point on segment
    t = max(0.0, min(1.0, ((bx - x1) * tx + (by - y1) * ty) / length))
    cx, cy = x1 + t * dx, y1 + t * dy

    dist_x, dist_y = bx - cx, by - cy
    dist = math.hypot(dist_x, dist_y)
    if dist >= BALL_RADIUS or dist < 1e-6:
        return None

    # Normal must point toward ball center
    if dist_x * nx + dist_y * ny < 0:
        nx, ny = -nx, -ny

    overlap = BALL_RADIUS - dist

    v_normal = vx * nx + vy * ny
    if v_normal >= 0:
        return None  # already separating

    # Decompose velocity
    v_tan_x = vx - v_normal * nx
    v_tan_y = vy - v_normal * ny

    # Reflect normal, friction tangential
    new_vx = v_tan_x * ROLLING_FRICTION + (-RESTITUTION * v_normal) * nx
    new_vy = v_tan_y * ROLLING_FRICTION + (-RESTITUTION * v_normal) * ny

    return new_vx, new_vy, nx * overlap, ny * overlap


def update_ball(
    ball: Ball,
    platforms: List[Tuple[int, int, int, int]],
    width: int,
    height: int,
) -> None:
    if not ball.active:
        return

    ball.vy += GRAVITY
    ball.vx *= AIR_FRICTION

    nx = ball.x + ball.vx
    ny = ball.y + ball.vy

    for seg in platforms:
        result = _resolve_collision(nx, ny, ball.vx, ball.vy, *seg)
        if result is not None:
            ball.vx, ball.vy, px, py = result
            nx += px
            ny += py

    ball.x = nx
    ball.y = ny

    # Reset when ball leaves the visible area
    if ball.y > height + BALL_RADIUS + 40 or ball.x < -300 or ball.x > width + 300:
        ball.reset(width // 2, BALL_RADIUS + 10)


def _draw_ball(img: np.ndarray, bx: int, by: int) -> None:
    # Drop shadow
    cv2.circle(img, (bx + 4, by + 4), BALL_RADIUS, (10, 10, 10), -1)
    # Body gradient approximation: dark edge
    cv2.circle(img, (bx, by), BALL_RADIUS, (20, 80, 220), -1)
    # Inner lighter circle
    inner = int(BALL_RADIUS * 0.7)
    cv2.circle(img, (bx, by), inner, (40, 130, 255), -1)
    # Specular highlight
    hl_r = max(3, BALL_RADIUS // 4)
    hl_x = bx - BALL_RADIUS // 3
    hl_y = by - BALL_RADIUS // 3
    cv2.circle(img, (hl_x, hl_y), hl_r, (200, 220, 255), -1)


def render(
    background: np.ndarray,
    ball: Ball,
    platforms: List[Tuple[int, int, int, int]],
    show_platforms: bool,
) -> np.ndarray:
    out = background.copy()

    if show_platforms:
        for x1, y1, x2, y2 in platforms:
            cv2.line(out, (x1, y1), (x2, y2), (50, 220, 50), 3, cv2.LINE_AA)

    _draw_ball(out, int(ball.x), int(ball.y))

    if not ball.active:
        h, w = out.shape[:2]
        msg = "SPACE — запустить / сбросить"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(
            out, msg,
            ((w - tw) // 2, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA,
        )

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Falling ball projector demo")
    p.add_argument("--image", type=str, default=None, help="Static image instead of camera")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--fullscreen", action="store_true", help="Fullscreen window (for projector)")
    p.add_argument("--no-camera", action="store_true", help="Black background (don't show camera feed)")
    p.add_argument("--no-platforms", action="store_true", help="Hide detected platform lines")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- image mode ---
    if args.image is not None:
        static_frame = cv2.imread(args.image)
        if static_frame is None:
            print(f"Error: cannot load image {args.image}")
            return
        h, w = static_frame.shape[:2]
        start_x, start_y = w // 2, BALL_RADIUS + 10
        ball = Ball(x=start_x, y=start_y)
        platforms = detect_platforms(static_frame)

        win = "Falling Ball"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        show_platforms = not args.no_platforms
        background = np.full_like(static_frame, 255) if args.no_camera else static_frame

        while True:
            update_ball(ball, platforms, w, h)
            output = render(background, ball, platforms, show_platforms)
            cv2.imshow(win, output)
            key = cv2.waitKey(16) & 0xFF  # ~60 fps

            if key == 27:
                break
            elif key == ord(" "):
                ball.reset(start_x, start_y)
                ball.active = True
            elif key == ord("p"):
                show_platforms = not show_platforms

        cv2.destroyAllWindows()
        return

    # --- camera mode ---
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera {args.camera}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: cannot read from camera")
        cap.release()
        return

    h, w = frame.shape[:2]
    start_x, start_y = w // 2, BALL_RADIUS + 10
    ball = Ball(x=start_x, y=start_y)

    win = "Falling Ball"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    show_platforms = not args.no_platforms

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        platforms = detect_platforms(frame)
        update_ball(ball, platforms, w, h)

        background = np.full_like(frame, 255) if args.no_camera else frame
        output = render(background, ball, platforms, show_platforms)

        cv2.imshow(win, output)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == ord(" "):
            ball.reset(start_x, start_y)
            ball.active = True
        elif key == ord("p"):
            show_platforms = not show_platforms
        elif key == ord("c"):
            args.no_camera = not args.no_camera

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
