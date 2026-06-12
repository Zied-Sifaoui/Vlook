import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .base import BaseProcessor

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')


class ScarProcessor(BaseProcessor):
    name = "scar"
    description = "Detect and conceal scars"

    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path=os.path.join(_MODELS, "blaze_face_short_range.tflite")
        )
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def process(self, frame: np.ndarray) -> np.ndarray:
        try:
            h, w, _ = frame.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            detection_result = self.detector.detect(mp_image)

            output = frame.copy()

            for detection in detection_result.detections:
                bbox = detection.bounding_box

                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                bw = int(bbox.width)
                bh = int(bbox.height)

                x = max(0, x)
                y = max(0, y)

                face = frame[y:y + bh, x:x + bw]

                if face.size == 0:
                    continue

                hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
                lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

                kernel = np.ones((3, 3), np.uint8)
                skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
                skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 40, 120)
                scar_mask = cv2.bitwise_and(edges, skin_mask)
                scar_mask = cv2.dilate(scar_mask, kernel, iterations=1)

                fixed_face = cv2.inpaint(face, scar_mask, 3, cv2.INPAINT_TELEA)
                output[y:y + bh, x:x + bw] = fixed_face

            return output
        except Exception as e:
            return frame

    def release(self):
        try:
            self.detector.close()
        except Exception:
            pass
