"""
test_occlusion.py  -  Occlusion test viewer (no camera)
=========================================================
Controls:
  LEFT / RIGHT  ->  yaw  (turn left/right)
  UP   / DOWN   ->  pitch
  A / D         ->  roll
  W / S         ->  closer / further
  R             ->  reset
  Q / ESC       ->  quit

Sliders:
  Fade Start  - where occlusion begins (default 75 = 0.75)
  Fade Width  - softness of the edge   (default 35 = 0.35)
  Yaw Scale   - aggressiveness         (default 120 = 1.20)
"""

import cv2
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.renderer import ARRenderer

WIN  = "Occlusion Tester"
W, H = 960, 600
BG   = (30, 30, 35)

K = np.array([
    [float(W), 0.0,      W / 2.0],
    [0.0,      float(W), H / 2.0],
    [0.0,      0.0,          1.0],
], dtype=np.float32)

STEP_ROT = np.deg2rad(3.0)
STEP_Z   = 20.0

yaw   = 0.0
pitch = 0.0
roll  = 0.0
tz    = 600.0


def make_rvec(yaw, pitch, roll):
    """Build a (3,1) Rodrigues vector from Euler angles."""
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [ 0,           1, 0           ],
                   [-np.sin(yaw), 0, np.cos(yaw)]], np.float32)
    Rx = np.array([[1, 0,              0             ],
                   [0, np.cos(pitch), -np.sin(pitch) ],
                   [0, np.sin(pitch),  np.cos(pitch) ]], np.float32)
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll),  np.cos(roll), 0],
                   [0,             0,            1]], np.float32)
    rvec, _ = cv2.Rodrigues((Ry @ Rx @ Rz).astype(np.float32))
    return rvec   # shape (3,1)  ← correct input for renderer


def get_sliders():
    try:
        fs = cv2.getTrackbarPos("Fade Start", WIN) / 100.0
        fw = cv2.getTrackbarPos("Fade Width", WIN) / 100.0
        ys = cv2.getTrackbarPos("Yaw Scale",  WIN) / 100.0
    except Exception:
        fs, fw, ys = 0.75, 0.35, 1.20
    return max(fs, 0.01), max(fw, 0.01), max(ys, 0.01)


def draw_info(frame, yaw, pitch, roll, tz, fs, fw, ys):
    lines = [
        f"Yaw   : {np.rad2deg(yaw):+.1f} deg   J=left   L=right",
        f"Pitch : {np.rad2deg(pitch):+.1f} deg   I=up     K=down",
        f"Roll  : {np.rad2deg(roll):+.1f} deg   U=ccw    O=cw",
        f"Dist  : {tz:.0f} mm   W=closer  S=farther",
        "",
        f"Fade Start : {fs:.2f}",
        f"Fade Width : {fw:.2f}",
        f"Yaw Scale  : {ys:.2f}",
        "",
        "[R] reset    [Q] quit",
    ]
    for i, l in enumerate(lines):
        cv2.putText(frame, l, (12, 22 + i * 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (170, 210, 170), 1, cv2.LINE_AA)

    bw = 300; bx = W//2 - bw//2; by = H - 28
    cv2.rectangle(frame, (bx, by), (bx+bw, by+8), (45, 45, 55), -1)
    mid    = bx + bw//2
    yaw_px = int(np.clip(yaw / np.pi, -1, 1) * bw // 2)
    col    = (0, 200, 140) if abs(yaw) < 0.25 else (30, 110, 255)
    if yaw_px != 0:
        x0, x1 = (mid + yaw_px, mid) if yaw_px < 0 else (mid, mid + yaw_px)
        cv2.rectangle(frame, (x0, by), (x1, by+8), col, -1)
    cv2.line(frame, (mid, by-2), (mid, by+10), (160, 160, 160), 1)
    cv2.putText(frame, "YAW", (bx - 38, by + 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 130), 1)

def main():
    global yaw, pitch, roll, tz

    renderer = ARRenderer("assets/Glasses.obj")
    style    = {"frame": (200, 200, 200), "lens": (55, 55, 55)}

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)
    cv2.createTrackbar("Fade Start", WIN,  75, 100, lambda x: None)
    cv2.createTrackbar("Fade Width", WIN,  35, 100, lambda x: None)
    cv2.createTrackbar("Yaw Scale",  WIN, 120, 200, lambda x: None)

    while True:
        rvec = make_rvec(yaw, pitch, roll)
        tvec = np.array([[0.0], [0.0], [tz]], np.float32)
        fs, fw, ys = get_sliders()

        frame = np.full((H, W, 3), BG, dtype=np.uint8)
        # crosshair
        cv2.line(frame, (W//2-20, H//2), (W//2+20, H//2), (55, 55, 70), 1)
        cv2.line(frame, (W//2, H//2-20), (W//2, H//2+20), (55, 55, 70), 1)

        frame = renderer.render(
            frame, rvec, tvec, K,
            style          = style,
            offset_x       = 0.0,
            offset_y       = 0.0,
            offset_z       = 0.0,
            mesh_scale     = 3060.0,
            occ_fade_start = fs,
            occ_fade_width = fw,
            occ_yaw_scale  = ys,
        )

        draw_info(frame, yaw, pitch, roll, tz, fs, fw, ys)
        cv2.imshow(WIN, frame)

        k = cv2.waitKey(16) & 0xFF
        if   k in (ord("j"), ord("J")): yaw   -= STEP_ROT
        elif k in (ord("l"), ord("L")): yaw   += STEP_ROT
        elif k in (ord("i"), ord("I")): pitch -= STEP_ROT
        elif k in (ord("k"), ord("K")): pitch += STEP_ROT
        elif k in (ord("u"), ord("U")): roll  -= STEP_ROT
        elif k in (ord("o"), ord("O")): roll  += STEP_ROT
        elif k in (ord("w"), ord("W")): tz = max(100.0,  tz - STEP_Z)
        elif k in (ord("s"), ord("S")): tz = min(2000.0, tz + STEP_Z)
        elif k in (ord("r"), ord("R")): yaw = pitch = roll = 0.0; tz = 600.0
        elif k in (27, ord("q"), ord("Q")): break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()