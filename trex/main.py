from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import cv2
import mss
import numpy as np
import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True


@dataclass
class Box:
    left: int
    top: int
    width: int
    height: int

    def to_mss(self) -> dict:
        return {"left": self.left, "top": self.top,
                "width": self.width, "height": self.height}

    def expand_right(self, extra: int) -> "Box":
        return Box(self.left, self.top, self.width + extra, self.height)


@dataclass
class Calibration:
    ground: Box
    monitor: dict


def grab_full_screen(sct: mss.mss) -> tuple[np.ndarray, dict]:
    mon = sct.monitors[1]
    raw = np.array(sct.grab(mon))
    img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    return img, mon


def _select_box(display: np.ndarray, scale: float, title: str,
                allow_skip: bool = False) -> Box | None:
    state = {"start": None, "end": None, "dragging": False, "done": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["end"] = (x, y)
            state["dragging"] = True
            state["done"] = False
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state["end"] = (x, y)
            state["dragging"] = False
            state["done"] = True

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)

    while True:
        view = display.copy()
        if state["start"] is not None and state["end"] is not None:
            x0, y0 = state["start"]
            x1, y1 = state["end"]
            color = (0, 200, 0) if state["done"] else (0, 165, 255)
            cv2.rectangle(view, (x0, y0), (x1, y1), color, 2)

        cv2.imshow(title, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(title)
            sys.exit(1)
        if allow_skip and key in (ord("s"), ord("S")):
            cv2.destroyWindow(title)
            return None
        if key in (ord("r"), ord("R")):
            state.update({"start": None, "end": None,
                          "dragging": False, "done": False})
        if key in (13, 10) and state["done"]:
            break

    cv2.destroyWindow(title)
    x0, y0 = state["start"]
    x1, y1 = state["end"]
    lx, rx = sorted([x0, x1])
    ty, by = sorted([y0, y1])
    return Box(
        left=int(lx / scale),
        top=int(ty / scale),
        width=int((rx - lx) / scale),
        height=int((by - ty) / scale),
    )


def calibrate() -> Calibration:
    with mss.mss() as sct:
        img, mon = grab_full_screen(sct)

    h, w = img.shape[:2]
    scale = min(1.0, 1500.0 / w, 820.0 / h)
    display = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img.copy()

    g = _select_box(display, scale, title="ground", allow_skip=False)
    if g is None or g.width < 8 or g.height < 4:
        sys.exit(1)

    def shift(b: Box) -> Box:
        return Box(b.left + mon["left"], b.top + mon["top"], b.width, b.height)

    return Calibration(ground=shift(g), monitor=mon)


def obstacle_strength(bgra: np.ndarray) -> float:
    gray = cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)
    return float(gray.std())


def run(calib: Calibration, debug: bool = False,
        ground_thresh: float = 3.0) -> None:
    time.sleep(4)

    JUMP_COOLDOWN = 0.05

    last_jump = 0.0
    start_t = time.time()

    with mss.mss() as sct:
        try:
            while True:
                now = time.time()
                elapsed = now - start_t

                extra_px = int(min(160, elapsed * 2.4))
                ground = calib.ground.expand_right(extra_px)

                ground_img = np.array(sct.grab(ground.to_mss()))
                ground_s = obstacle_strength(ground_img)
                ground_hit = ground_s > ground_thresh

                if ground_hit and (now - last_jump) > JUMP_COOLDOWN:
                    pyautogui.keyDown("space")
                    time.sleep(0.10)
                    pyautogui.keyUp("space")
                    last_jump = now

                if debug:
                    panel_box = {"left": ground.left - 40,
                                 "top": ground.top - 60,
                                 "width": ground.width + 80,
                                 "height": ground.height + 90}
                    panel = np.array(sct.grab(panel_box))
                    panel = cv2.cvtColor(panel, cv2.COLOR_BGRA2BGR)

                    x = ground.left - panel_box["left"]
                    y = ground.top - panel_box["top"]
                    color = (0, 0, 255) if ground_hit else (0, 200, 0)
                    cv2.rectangle(panel, (x, y),
                                  (x + ground.width, y + ground.height), color, 2)
                    cv2.putText(panel, f"std:{ground_s:.1f}",
                                (x, max(12, y - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                    cv2.imshow("debug", panel)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                else:
                    time.sleep(0.003)
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true")
    p.add_argument("--ground-thresh", type=float, default=9.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    calib = calibrate()
    run(calib, debug=args.debug, ground_thresh=args.ground_thresh)


if __name__ == "__main__":
    main()
