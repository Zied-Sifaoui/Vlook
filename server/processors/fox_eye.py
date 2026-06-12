import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .base import BaseProcessor

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')


def _fox_eye_warp(frame, landmarks, strength=0.5):
    h, w = frame.shape[:2]
    corners = [33, 263]

    for idx in corners:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)

        radius = 40

        x1 = max(x - radius, 0)
        x2 = min(x + radius, w)
        y1 = max(y - radius, 0)
        y2 = min(y + radius, h)

        roi = frame[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]

        yv, xv = np.indices((rh, rw), dtype=np.float32)

        lift = strength * (1 - (yv / rh))

        map_x = xv - lift * 10
        map_y = yv - lift * 15

        map_x = np.clip(map_x, 0, rw - 1).astype(np.float32)
        map_y = np.clip(map_y, 0, rh - 1).astype(np.float32)

        warped = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)

        frame[y1:y2, x1:x2] = warped

    return frame


class FoxEyeProcessor(BaseProcessor):
    name = "fox_eye"
    description = "Cat-eye lifting effect"

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
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)
            if result.face_landmarks:
                frame = _fox_eye_warp(frame.copy(), result.face_landmarks[0], strength=0.5)
            return frame
        except Exception as e:
            return frame

    def release(self):
        try:
            self.landmarker.close()
        except Exception:
            pass
