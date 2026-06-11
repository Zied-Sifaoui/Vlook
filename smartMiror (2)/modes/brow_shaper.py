"""
modes/brow_shaper.py — Rounded eyebrow overlay
"""

import cv2
import numpy as np

LB_UP = [70, 63, 105, 66, 107]
LB_LO = [46, 53,  52, 65,  55]
RB_UP = [300, 293, 334, 296, 336]
RB_LO = [276, 283, 282, 295, 285]


def _smooth_curve(pts, n=60):
    t  = np.linspace(0, 1, len(pts))
    to = np.linspace(0, 1, n)
    return np.column_stack([np.interp(to, t, pts[:, 0]),
                            np.interp(to, t, pts[:, 1])])


def _sample_brow_color(frame, lms, W, H):
    samples = []
    for idx in [234, 454, 10]:
        lm = lms[idx]
        cx, cy = int(lm.x * W), int(lm.y * H)
        p = frame[max(0, cy-12):cy+12, max(0, cx-12):cx+12]
        if p.size > 0:
            samples.append(np.mean(p, axis=(0, 1)))
    if not samples:
        return np.array([25, 35, 50], np.float32)
    skin = np.mean(samples, axis=0)
    gray = np.dot(skin, [0.114, 0.587, 0.299])
    return np.clip(skin * 0.25 + np.array([gray, gray, gray]) * 0.10,
                   8, 110).astype(np.float32)


def _draw_one_brow(frame, up_idx, lo_idx, lms, W, H, color):
    up = np.array([(lms[i].x*W, lms[i].y*H) for i in up_idx], np.float32)
    lo = np.array([(lms[i].x*W, lms[i].y*H) for i in lo_idx], np.float32)
    up = up[np.argsort(up[:, 0])]
    lo = lo[np.argsort(lo[:, 0])]

    n  = 60
    uc = _smooth_curve(up, n)
    lc = _smooth_curve(lo, n)

    t    = np.linspace(0, 1, n)
    lift = (np.exp(-((t - 0.42)**2) / (2 * 0.18**2))
            * np.mean(np.abs(uc[:, 1] - lc[:, 1])) * 0.55)
    au   = uc.copy(); au[:, 1] -= lift

    tap = 0.6 + 0.4 * np.sin(np.linspace(0, np.pi, n)) ** 0.4
    mid = (au[:, 1] + lc[:, 1]) / 2
    half= (lc[:, 1] - au[:, 1]) / 2
    tu  = np.column_stack([uc[:, 0], mid - half * tap])
    tl  = np.column_stack([lc[:, 0], mid + half * tap])
    cont= np.vstack([tu, tl[::-1]]).astype(np.int32)

    h2, w2 = frame.shape[:2]
    mask = np.zeros((h2, w2), np.float32)
    cv2.fillPoly(mask, [cont], 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=6, sigmaY=5)

    core = np.zeros((h2, w2), np.float32)
    sh   = (cont - cont.mean(0)) * 0.68 + cont.mean(0)
    cv2.fillPoly(core, [sh.astype(np.int32)], 1.0)
    core = cv2.GaussianBlur(core, (0, 0), sigmaX=3, sigmaY=2)

    comb = np.clip(mask * 0.50 + core * 0.72, 0, 1)
    a3   = np.stack([comb * 0.70] * 3, axis=-1)
    cl   = np.full_like(frame, color.astype(np.uint8))
    frame[:] = np.clip(frame * (1 - a3) + cl * a3, 0, 255).astype(np.uint8)


class BrowShaperMode:

    def process(self, frame, lms, W, H) -> np.ndarray:
        if lms is None:
            return frame
        color = _sample_brow_color(frame, lms, W, H)
        out   = frame.copy()
        _draw_one_brow(out, LB_UP, LB_LO, lms, W, H, color)
        _draw_one_brow(out, RB_UP, RB_LO, lms, W, H, color)
        return out