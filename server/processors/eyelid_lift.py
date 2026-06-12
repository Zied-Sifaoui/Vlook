import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .base import BaseProcessor

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')


def _create_displacement_map(h, w, eyes_data, lift_strength):
    map_y = np.zeros((h, w), dtype=np.float32)

    y_coords = np.arange(h).reshape(-1, 1)
    x_coords = np.arange(w).reshape(1, -1)

    for eye in eyes_data:
        cx, cy, radius = eye

        dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

        mask = dist < radius

        if not np.any(mask):
            continue

        norm_dist = dist[mask] / radius
        falloff = 0.5 * (1 + np.cos(np.pi * norm_dist))

        local_lift = lift_strength * falloff
        map_y[mask] -= local_lift

    return map_y


class EyelidLiftProcessor(BaseProcessor):
    name = "eyelid_lift"
    description = "Natural eyelid enhancement"

    def __init__(self):
        model_path = os.path.join(_MODELS, "face_landmarker.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray) -> np.ndarray:
        try:
            h, w = frame.shape[:2]

            x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
            map_x = x_grid.astype(np.float32)
            map_y_base = y_grid.astype(np.float32)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_img)

            eyes_data = []

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                left_eye_indices = [159, 158, 157, 155, 153, 33]
                right_eye_indices = [386, 385, 384, 382, 380, 362]

                for indices in [left_eye_indices, right_eye_indices]:
                    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
                    center = np.mean(pts, axis=0)
                    eye_width = np.linalg.norm(pts[0] - pts[-1])
                    radius = eye_width * 2.5 * 1.5
                    eyes_data.append((center[0], center[1], radius))

            if eyes_data:
                lift_amount = (h * 30) / 1000.0
                delta_y = _create_displacement_map(h, w, eyes_data, lift_amount)
                final_map_y = map_y_base + delta_y
                warped = cv2.remap(frame, map_x, final_map_y,
                                   interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
                return warped

            return frame
        except Exception as e:
            return frame

    def release(self):
        try:
            self.landmarker.close()
        except Exception:
            pass
