import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .base import BaseProcessor

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')

MASK_BLUR = 21


class HairColorProcessor(BaseProcessor):
    name = "hair_color"
    description = "Change hair color"

    def __init__(self, target_hue: int = 25, saturation_boost: float = 1.3):
        self.target_hue = target_hue
        self.saturation_boost = saturation_boost

        base_options = python.BaseOptions(
            model_asset_path=os.path.join(_MODELS, 'hair_segmenter.tflite')
        )
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True,
            running_mode=vision.RunningMode.IMAGE
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def process(self, frame: np.ndarray) -> np.ndarray:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = self.segmenter.segment(mp_image)
            mask = result.category_mask.numpy_view()

            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask.astype(np.float32),
                                  (frame.shape[1], frame.shape[0]))

            hair_mask = (mask > 0.3).astype(np.uint8) * 255

            hair_mask = cv2.medianBlur(hair_mask, 5)
            kernel = np.ones((3, 3), np.uint8)
            hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hair_mask_f = hair_mask.astype(np.float32) / 255.0
            hair_mask_f = cv2.GaussianBlur(hair_mask_f, (MASK_BLUR, MASK_BLUR), 0)

            hsv[:, :, 0] = np.where(hair_mask_f > 0, self.target_hue, hsv[:, :, 0])
            hsv[:, :, 1] = np.where(
                hair_mask_f > 0,
                hsv[:, :, 1] * self.saturation_boost,
                hsv[:, :, 1]
            )
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

            output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            return output
        except Exception as e:
            return frame

    def release(self):
        try:
            self.segmenter.close()
        except Exception:
            pass
