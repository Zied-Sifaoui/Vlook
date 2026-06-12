import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .base import BaseProcessor

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')


def _v_jaw_ultra(frame, landmarks, strength=0.7):
    h, w = frame.shape[:2]

    try:
        jaw_idx = [152] + list(range(148, 177)) + list(range(377, 401)) + [
            17, 336, 245, 392, 93, 132, 58, 172, 136, 150, 149
        ]
        jaw_idx = list(set(jaw_idx))

        jaw_px = []
        for idx in jaw_idx:
            lm = landmarks[idx]
            jaw_px.append((int(lm.x * w), int(lm.y * h)))
        jaw_px = np.array(jaw_px, dtype=np.int32)

        chin = landmarks[152]
        chin_px = np.array([int(chin.x * w), int(chin.y * h)])

        x_min, y_min = np.min(jaw_px, axis=0)
        x_max, y_max = np.max(jaw_px, axis=0)

        pad = int(max(x_max - x_min, y_max - y_min) * 1.8)
        x1, y1 = max(0, x_min - pad), max(0, y_min - pad)
        x2, y2 = min(w, x_max + pad), min(h, y_max + pad)

        if x2 <= x1 or y2 <= y1:
            return frame

        roi = frame[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]
        yv, xv = np.indices((rh, rw), dtype=np.float32)
        cx, cy = rw // 2, rh // 2

        dx = xv - cx
        dy = yv - cy

        taper = 1 - (dy / rh) * 0.98
        taper = np.where(taper == 0, 1e-6, taper)
        ell = np.sqrt((dx / (rw * 0.85 * taper)) ** 2 + (dy / (rh * 0.9)) ** 2)
        mask = (ell < 1.2).astype(np.float32)
        mask = mask * (yv > rh * 0.1).astype(np.float32)

        if np.sum(mask) < 100:
            return frame

        dist_center = np.abs(dx)
        chin_w = np.clip(1 - (dy / rh) * 0.99, 0, 1)
        edge_w = np.clip(dist_center / (rw * 0.25), 0, 1.2)
        side_w = np.clip(1 - np.abs(dx) / (rw * 0.4), 0, 1)

        weight = chin_w * edge_w * side_w * mask

        base_shift_x = rw * 0.5
        base_shift_y = rh * 0.25

        shift_x = -np.sign(dx) * base_shift_x * strength * weight * (1 + edge_w * 0.5)

        chin_zone = (dy > rh * 0.4).astype(np.float32)
        shift_y = -base_shift_y * strength * weight * chin_zone * 1.5

        map_x = np.clip(xv - shift_x + x1, 0, w - 1).astype(np.float32)
        map_y = np.clip(yv - shift_y + y1, 0, h - 1).astype(np.float32)

        warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

        blend = cv2.GaussianBlur(mask, (61, 61), 0)
        blend = np.clip(blend * 1.2, 0, 1)
        blend_3ch = np.repeat(blend[:, :, np.newaxis], 3, axis=2)

        result = (warped * 0.95 + frame * 0.05).astype(np.uint8)
        result = (result * blend_3ch + frame * (1 - blend_3ch)).astype(np.uint8)

        shadow_mask = mask.copy()
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_ERODE, np.ones((5, 5), np.uint8))
        shadow_mask = cv2.GaussianBlur(shadow_mask, (31, 31), 5)

        shadow = result[y1:y2, x1:x2].copy()
        shadow = cv2.addWeighted(shadow, 0.7, np.zeros_like(shadow), 0.3, 8)

        edge_mask = cv2.Canny((mask * 255).astype(np.uint8), 30, 100)
        edge_mask = cv2.dilate(edge_mask, np.ones((7, 7), np.uint8), iterations=3)
        edge_mask = cv2.GaussianBlur(edge_mask.astype(np.float32), (21, 21), 0)
        edge_mask = np.clip(edge_mask / 255, 0, 0.6)

        edge_3ch = np.repeat(edge_mask[:, :, np.newaxis], 3, axis=2)

        shadow_region = result[y1:y2, x1:x2]
        shadow_region = (shadow_region * (1 - edge_3ch) + shadow * edge_3ch).astype(np.uint8)
        result[y1:y2, x1:x2] = shadow_region

        return result

    except Exception as e:
        return frame


class JawProcessor(BaseProcessor):
    name = "jaw"
    description = "Jaw slimming and contouring"

    def __init__(self):
        model_path = os.path.join(_MODELS, "face_landmarker.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray) -> np.ndarray:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self.landmarker.detect(mp_img)
            if results.face_landmarks:
                lm = results.face_landmarks[0]
                frame = _v_jaw_ultra(frame.copy(), lm, strength=0.7)
            return frame
        except Exception as e:
            return frame

    def release(self):
        try:
            self.landmarker.close()
        except Exception:
            pass
