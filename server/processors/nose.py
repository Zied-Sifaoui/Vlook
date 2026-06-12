import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from .base import BaseProcessor

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')
_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'images')

SIZE_MULTIPLIER = 0.30
X_OFFSET = 0
Y_OFFSET = -22
ANCHOR_RATIO = 0.50
HIDE_REAL_NOSE = True

SKIN_IDX = [116, 123, 147, 213, 345, 352, 376, 433, 50, 280, 187, 411, 152, 10]
NOSE_IDX = [1, 2, 3, 4, 5, 6, 19, 94, 98, 240, 290, 440, 456, 168, 197, 195]


def _remove_bg(bgr):
    h, w = bgr.shape[:2]
    mx, my = max(6, int(w * .10)), max(6, int(h * .10))
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    gm = np.zeros((h, w), np.uint8)
    bd, fd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gm, rect, bd, fd, 10, cv2.GC_INIT_WITH_RECT)
    except Exception:
        gm[:] = cv2.GC_PR_FGD
    fg = np.where((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    kill_wg = ((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 40)).astype(np.uint8) * 255
    kill_bk = (hsv[:, :, 2] < 15).astype(np.uint8) * 255
    fg = cv2.bitwise_and(fg, cv2.bitwise_not(kill_wg))
    fg = cv2.bitwise_and(fg, cv2.bitwise_not(kill_bk))

    corners = [bgr[0, 0], bgr[0, w - 1], bgr[h - 1, 0], bgr[h - 1, w - 1],
               bgr[0, w // 2], bgr[h - 1, w // 2], bgr[h // 2, 0], bgr[h // 2, w - 1]]
    for c in corners:
        dist = np.abs(bgr.astype(np.int32) - c.astype(np.int32)).max(axis=2)
        fg = cv2.bitwise_and(fg, cv2.bitwise_not((dist < 30).astype(np.uint8) * 255))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = np.where(labels == big, 255, 0).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    fg = cv2.GaussianBlur(fg, (3, 3), 0)
    return fg


def _load_nose(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    bgr = img[:, :, :3]
    if img.shape[2] == 4:
        alpha = np.minimum(img[:, :, 3], _remove_bg(bgr))
    else:
        alpha = _remove_bg(bgr)
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.5)
    bgr = cv2.addWeighted(bgr, 1.3, blur, -0.3, 0)
    return bgr, alpha


def _sample_skin(frame, lms, fw, fh, skin_buf):
    cols = []
    for i in SKIN_IDX:
        if i >= len(lms):
            continue
        x, y = int(lms[i].x * fw), int(lms[i].y * fh)
        if 5 <= x < fw - 5 and 5 <= y < fh - 5:
            patch = frame[y - 4:y + 4, x - 4:x + 4]
            if patch.size > 0:
                cols.append(patch.mean(axis=(0, 1)))
    if not cols:
        return None
    skin_buf.append(np.array(cols, dtype=np.float32).mean(axis=0))
    return np.array(list(skin_buf), dtype=np.float32).mean(axis=0)


def _apply_skin_color(nose_bgr, nose_mask, skin_bgr):
    if skin_bgr is None:
        return nose_bgr
    fg = nose_mask > 80
    if not fg.any():
        return nose_bgr
    nose_lab = cv2.cvtColor(nose_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    skin_lab = cv2.cvtColor(
        skin_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).astype(np.float32).reshape(3)
    nose_mean_lab = np.array([nose_lab[:, :, c][fg].mean() for c in range(3)], dtype=np.float32)
    shift = skin_lab - nose_mean_lab
    shift[0] *= 0.55
    shift[1] *= 0.90
    shift[2] *= 0.90
    result_lab = nose_lab.copy()
    result_lab[:, :, 0] = np.clip(nose_lab[:, :, 0] + shift[0], 0, 255)
    result_lab[:, :, 1] = np.clip(nose_lab[:, :, 1] + shift[1], 0, 255)
    result_lab[:, :, 2] = np.clip(nose_lab[:, :, 2] + shift[2], 0, 255)
    matched = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(matched, 0.75, nose_bgr, 0.25, 0)


def _build_hide_mask(lms, fw, fh):
    pts = []
    for i in NOSE_IDX:
        if i < len(lms):
            pts.append([int(lms[i].x * fw), int(lms[i].y * fh)])
    if not pts:
        return None
    pts = np.array(pts, np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((fh, fw), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def _hide_nose(frame, mask):
    _, hard = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    inp = cv2.inpaint(frame, hard, 5, cv2.INPAINT_TELEA)
    a = mask.astype(float)[:, :, None] / 255.0
    return (inp * a + frame * (1 - a)).astype(np.uint8)


def _get_roll(lms, fw, fh):
    lx, ly = lms[33].x * fw, lms[33].y * fh
    rx, ry = lms[263].x * fw, lms[263].y * fh
    return float(np.degrees(np.arctan2(ry - ly, rx - lx)))


def _rotate_rgba(img, angle):
    if abs(angle) < 0.4:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _make_edge_feather_mask(mask, feather_px=14):
    k_size = feather_px * 2 + 1
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    eroded = cv2.erode(mask, k_erode, iterations=1)
    feathered = cv2.GaussianBlur(eroded, (k_size * 2 + 1, k_size * 2 + 1), 0)
    return np.minimum(feathered, mask)


def _transfer_lighting(nose_bgr, nose_mask, roi_bgr):
    fg = nose_mask > 40
    if not fg.any() or roi_bgr.shape != nose_bgr.shape:
        return nose_bgr

    nose_f = nose_bgr.astype(np.float32)
    roi_f = roi_bgr.astype(np.float32)

    blur_k = 61
    nose_lf = cv2.GaussianBlur(
        cv2.cvtColor(nose_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32),
        (blur_k, blur_k), 0) + 1.0
    roi_lf = cv2.GaussianBlur(
        cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32),
        (blur_k, blur_k), 0) + 1.0

    ratio_2d = np.clip(roi_lf / nose_lf, 0.55, 1.80)
    ratio_3d = np.stack([ratio_2d, ratio_2d, ratio_2d], axis=2)

    result = np.clip(nose_f * ratio_3d, 0, 255)

    fg3 = np.stack([fg, fg, fg], axis=2)
    result[~fg3] = nose_f[~fg3]

    return cv2.addWeighted(result.astype(np.uint8), 0.60, nose_bgr, 0.40, 0)


def _realistic_overlay(frame, nb, nm, cx, cy, feather_px=12):
    fh, fw = frame.shape[:2]
    oh, ow = nb.shape[:2]
    x1, y1 = cx - ow // 2, cy - oh // 2
    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x1 + ow, fw), min(y1 + oh, fh)
    iw, ih = x2c - x1c, y2c - y1c
    if iw <= 0 or ih <= 0:
        return frame

    ox, oy = x1c - x1, y1c - y1
    nb_crop = nb[oy:oy + ih, ox:ox + iw]
    nm_crop = nm[oy:oy + ih, ox:ox + iw]
    roi = frame[y1c:y2c, x1c:x2c]

    nb_lit = _transfer_lighting(nb_crop, nm_crop, roi)

    feathered_mask = _make_edge_feather_mask(nm_crop, feather_px=feather_px)

    a3 = feathered_mask.astype(np.float32)[:, :, np.newaxis] / 255.0
    out = np.clip(
        nb_lit.astype(np.float32) * a3 + roi.astype(np.float32) * (1.0 - a3),
        0, 255
    ).astype(np.uint8)

    frame[y1c:y2c, x1c:x2c] = out
    return frame


class NoseProcessor(BaseProcessor):
    name = "nose"
    description = "Virtual nose overlay and reshaping"

    def __init__(self):
        nose_path = os.path.join(_ASSETS, "nose.png")
        self.orig_bgr, self.orig_mask = _load_nose(nose_path)
        self._skin_buf = deque(maxlen=30)
        self._hx = deque(maxlen=5)
        self._hy = deque(maxlen=5)
        self._hsz = deque(maxlen=5)
        self._hr = deque(maxlen=5)

        model_path = os.path.join(_MODELS, "face_landmarker.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def _sm(self, buf, v):
        buf.append(v)
        return sum(buf) / len(buf)

    def process(self, frame: np.ndarray) -> np.ndarray:
        try:
            if self.orig_bgr is None:
                return frame

            fh, fw = frame.shape[:2]

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = self.landmarker.detect(mp_img)

            if not results.face_landmarks:
                return frame

            lms = results.face_landmarks[0]

            face_w = abs(lms[454].x - lms[234].x) * fw
            tx = int(lms[4].x * fw) + X_OFFSET
            ty = int(lms[4].y * fh) + Y_OFFSET

            sx = int(self._sm(self._hx, tx))
            sy = int(self._sm(self._hy, ty))
            snw = int(self._sm(self._hsz, max(20, int(face_w * SIZE_MULTIPLIER))))
            snh = int(snw * self.orig_bgr.shape[0] / self.orig_bgr.shape[1])
            sr = self._sm(self._hr, _get_roll(lms, fw, fh))

            frame_copy = frame.copy()

            if HIDE_REAL_NOSE:
                hm = _build_hide_mask(lms, fw, fh)
                if hm is not None:
                    frame_copy = _hide_nose(frame_copy, hm)

            skin = _sample_skin(frame_copy, lms, fw, fh, self._skin_buf)
            matched = _apply_skin_color(self.orig_bgr, self.orig_mask, skin)

            patch = cv2.merge([matched, self.orig_mask])
            patch = _rotate_rgba(patch, sr)
            patch = cv2.resize(patch, (max(1, snw), max(1, snh)),
                               interpolation=cv2.INTER_LANCZOS4)

            nb, nm = patch[:, :, :3], patch[:, :, 3]
            cx = sx
            cy = sy - int((ANCHOR_RATIO - 0.5) * snh)

            frame_copy = _realistic_overlay(frame_copy, nb, nm, cx, cy, feather_px=12)

            return frame_copy
        except Exception as e:
            return frame

    def release(self):
        try:
            self.landmarker.close()
        except Exception:
            pass
