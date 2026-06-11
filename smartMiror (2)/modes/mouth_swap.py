"""
modes/mouth_swap.py — Mouth image overlay with color matching
Keys (active only in this mode): W/S/A/D = move  |  -/= = size  |  R = reset
"""

import cv2
import numpy as np

MOUTH_IDS = [61,146,91,181,84,17,314,405,321,375,291,308,
             324,318,402,317,14,87,178,88,95,78]

PAD_X       = 0.30
PAD_Y       = 0.40
MAX_OPACITY = 0.82
MOVE_STEP   = 3
SCALE_STEP  = 0.05


def _match_color_lab(src, dst, amask):
    sl = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    dl = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB).astype(np.float32)
    vis = amask > 0.1
    if not vis.any(): return src
    for c in range(3):
        sv = sl[:,:,c][vis]; dv = dl[:,:,c][vis]
        sm, ss = sv.mean(), sv.std() + 1e-6
        dm, ds = dv.mean(), dv.std() + 1e-6
        sl[:,:,c] = (sl[:,:,c] - sm) * (ds / ss) + dm
    return cv2.cvtColor(np.clip(sl, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _ellipse_alpha(h, w, feather):
    m = np.zeros((h, w), np.float32)
    cx, cy = w // 2, h // 2
    cv2.ellipse(m, (cx, cy), (max(cx-4, 1), max(cy-4, 1)), 0, 0, 360, 1.0, -1)
    if feather > 0:
        m = cv2.GaussianBlur(m, (feather*2+1, feather*2+1), feather*0.5)
    return m


def _alpha_blend(bg, ovr, a2d, x, y):
    H, W = bg.shape[:2]; oh, ow = ovr.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x+ow, W), min(y+oh, H)
    if x2 <= x1 or y2 <= y1: return bg
    sx1, sy1 = x1-x, y1-y; sx2, sy2 = sx1+(x2-x1), sy1+(y2-y1)
    roi = bg[y1:y2, x1:x2].astype(np.float32)
    ov  = ovr[sy1:sy2, sx1:sx2].astype(np.float32)
    a   = a2d[sy1:sy2, sx1:sx2, np.newaxis]
    out = bg.copy()
    out[y1:y2, x1:x2] = np.clip(ov*a + roi*(1-a), 0, 255).astype(np.uint8)
    return out


class MouthSwapMode:

    def __init__(self, mouth_img_path: str):
        self.src = None
        self.offset_x = 0; self.offset_y = 0; self.scale_d = 0.0
        try:
            img = cv2.imread(mouth_img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                self.src = img
                print(f"[Mouth] loaded {mouth_img_path}")
            else:
                print(f"[Mouth] image not found: {mouth_img_path}")
        except Exception as e:
            print(f"[Mouth] {e}")

    def handle_key(self, key: int):
        """Call from main loop when mode == Mouth Swap."""
        if   key in (ord('w'), ord('W'), 82): self.offset_y -= MOVE_STEP
        elif key in (ord('s'), ord('S'), 84): self.offset_y += MOVE_STEP
        elif key in (ord('a'), ord('A'), 81): self.offset_x -= MOVE_STEP
        elif key in (ord('d'), ord('D'), 83): self.offset_x += MOVE_STEP
        elif key == ord('-'): self.scale_d = round(max(-0.7, self.scale_d - SCALE_STEP), 2)
        elif key == ord('='): self.scale_d = round(min(2.0,  self.scale_d + SCALE_STEP), 2)
        elif key in (ord('r'), ord('R')): self.offset_x = self.offset_y = 0; self.scale_d = 0.0

    def draw_controls(self, frame):
        sc  = int((1 + self.scale_d) * 100)
        txt = (f"W/S:up-dn  A/D:left-right  -/=:size  R:reset"
               f"   offset({self.offset_x:+d},{self.offset_y:+d}) scale:{sc}%")
        cv2.putText(frame, txt, (10, frame.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100,108,128), 1, cv2.LINE_AA)

    def process(self, frame, lms) -> np.ndarray:
        if self.src is None or lms is None:
            return frame

        H, W = frame.shape[:2]
        pts  = np.array([(int(lms[i].x*W), int(lms[i].y*H)) for i in MOUTH_IDS])
        bx, by, bw, bh = cv2.boundingRect(pts)
        px, py = int(bw * PAD_X), int(bh * PAD_Y)
        rx, ry = max(bx-px, 0), max(by-py, 0)
        rw, rh = min(bw+2*px, W-rx), min(bh+2*py, H-ry)
        if rw < 8 or rh < 8:
            return frame

        sc    = max(0.3, 1.0 + self.scale_d)
        rws   = max(int(rw*sc), 8); rhs = max(int(rh*sc), 8)
        rxs   = rx - (rws-rw)//2 + self.offset_x
        rys   = ry - (rhs-rh)//2 + self.offset_y

        sh, sw = self.src.shape[:2]; asp = sw / sh
        if rws/rhs > asp: fh=rhs; fw=int(fh*asp)
        else:             fw=rws; fh=int(fw/asp)
        fw, fh = max(fw, 4), max(fh, 4)

        res    = cv2.resize(self.src, (fw, fh), interpolation=cv2.INTER_AREA)
        bgr    = res[:, :, :3]
        pa     = res[:, :, 3].astype(np.float32) / 255.0
        fth    = max(int(min(fw, fh)*0.18), 3)
        if fth % 2 == 0: fth += 1
        ca     = _ellipse_alpha(fh, fw, fth) * pa * MAX_OPACITY
        ox2    = rxs + (rws-fw)//2
        oy2    = rys + (rhs-fh)//2

        roi = frame[max(oy2,0):max(oy2,0)+fh, max(ox2,0):max(ox2,0)+fw]
        if roi.shape[:2] == (fh, fw):
            bgr = _match_color_lab(bgr, roi, ca)

        return _alpha_blend(frame, bgr, ca, ox2, oy2)