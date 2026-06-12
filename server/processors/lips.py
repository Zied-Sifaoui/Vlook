import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .base import BaseProcessor

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')


def _enlarge_upper_lip(frame, landmarks, strength=0.4):
    h, w = frame.shape[:2]

    upper_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]

    lip_points = []
    for idx in upper_idx:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        lip_points.append([x, y])

    lip_points = np.array(lip_points, np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [lip_points], 255)

    x, y, ww, hh = cv2.boundingRect(lip_points)

    pad = int(max(ww, hh) * 0.4)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + ww + pad)
    y2 = min(h, y + hh + pad)

    roi = frame[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]

    rh, rw = roi.shape[:2]

    map_x, map_y = np.meshgrid(np.arange(rw), np.arange(rh))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    cx = rw // 2
    cy = rh // 2

    dx = map_x - cx
    dy = map_y - cy

    dist = np.sqrt(dx ** 2 + dy ** 2)
    radius = max(ww, hh)

    mask_float = roi_mask.astype(np.float32) / 255.0

    scale = 1 - strength * mask_float * (1 - dist / radius)
    scale = np.clip(scale, 0.65, 1.0)

    map_x = cx + dx * scale
    map_y = cy + dy * scale

    warped = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    blur_mask = cv2.GaussianBlur(mask_float, (31, 31), 0)
    blur_mask = blur_mask[:, :, np.newaxis]

    blended = warped * blur_mask + roi * (1 - blur_mask)

    frame[y1:y2, x1:x2] = blended.astype(np.uint8)

    return frame


class LipsProcessor(BaseProcessor):
    name = "lips"
    description = "Lip color and shape enhancement"

    def __init__(self):
        model_path = os.path.join(_MODELS, "face_landmarker.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray) -> np.ndarray:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self.landmarker.detect(mp_img)
            if results.face_landmarks:
                frame = _enlarge_upper_lip(frame.copy(), results.face_landmarks[0], strength=0.4)
            return frame
        except Exception as e:
            return frame

    def release(self):
        try:
            self.landmarker.close()
        except Exception:
            pass
