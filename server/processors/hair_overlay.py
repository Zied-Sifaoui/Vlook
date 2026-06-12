import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .base import BaseProcessor

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')
_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'images')


class HairOverlayProcessor(BaseProcessor):
    name = "hair_overlay"
    description = "Virtual hairstyle overlay"

    def __init__(self):
        hair_path = os.path.join(_ASSETS, "hair.png")
        self.hair_png = cv2.imread(hair_path, cv2.IMREAD_UNCHANGED)

        face_base_options = python.BaseOptions(
            model_asset_path=os.path.join(_MODELS, "face_landmarker.task")
        )
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        hair_base_options = python.BaseOptions(
            model_asset_path=os.path.join(_MODELS, "hair_segmenter.tflite")
        )
        hair_options = vision.ImageSegmenterOptions(
            base_options=hair_base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True
        )
        self.hair_segmenter = vision.ImageSegmenter.create_from_options(hair_options)

    def process(self, frame: np.ndarray) -> np.ndarray:
        try:
            if self.hair_png is None or self.hair_png.shape[2] != 4:
                return frame

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            face_result = self.face_landmarker.detect(mp_image)

            if not face_result.face_landmarks:
                return frame

            landmarks = face_result.face_landmarks[0]

            left = landmarks[234]
            right = landmarks[454]
            top = landmarks[10]

            lx, ly = int(left.x * w), int(left.y * h)
            rx, ry = int(right.x * w), int(right.y * h)
            tx, ty = int(top.x * w), int(top.y * h)

            seg_result = self.hair_segmenter.segment(mp_image)

            mask = seg_result.category_mask.numpy_view()
            hair_mask = (mask == 1).astype(np.uint8) * 255
            hair_mask = cv2.GaussianBlur(hair_mask, (15, 15), 0)

            head_width = rx - lx
            hair_width = int(head_width * 1.5)
            hair_height = int(hair_width * self.hair_png.shape[0] / self.hair_png.shape[1])

            if hair_width <= 0 or hair_height <= 0:
                return frame

            hair_resized = cv2.resize(self.hair_png, (hair_width, hair_height))

            x_offset = lx - int(0.25 * hair_width)
            y_offset = ty - int(0.85 * hair_height)

            x_offset = max(0, min(w - hair_width, x_offset))
            y_offset = max(0, min(h - hair_height, y_offset))

            roi = frame[y_offset:y_offset + hair_height,
                        x_offset:x_offset + hair_width]

            if roi.shape[0] != hair_height or roi.shape[1] != hair_width:
                return frame

            alpha_hair = hair_resized[:, :, 3] / 255.0
            alpha_mask = hair_mask[y_offset:y_offset + hair_height,
                                   x_offset:x_offset + hair_width] / 255.0

            alpha = alpha_hair * alpha_mask

            result = frame.copy()
            for c in range(3):
                result[y_offset:y_offset + hair_height,
                       x_offset:x_offset + hair_width][:, :, c] = (
                    alpha * hair_resized[:, :, c] +
                    (1 - alpha) * roi[:, :, c]
                )

            return result
        except Exception as e:
            return frame

    def release(self):
        try:
            self.face_landmarker.close()
        except Exception:
            pass
        try:
            self.hair_segmenter.close()
        except Exception:
            pass
