import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .base import BaseProcessor

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models')

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def _get_finger_states(landmarks):
    fingers = []

    if landmarks[4].x > landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    tips = [8, 12, 16, 20]
    dips = [6, 10, 14, 18]

    for tip, dip in zip(tips, dips):
        if landmarks[tip].y < landmarks[dip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def _recognize_gesture(fingers):
    thumb, index, middle, ring, pinky = fingers

    if thumb == 1 and index == 1 and middle == 1 and ring == 1 and pinky == 1:
        return "HELLO"
    if thumb == 0 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
        return "NO"
    if thumb == 0 and index == 1 and middle == 0 and ring == 0 and pinky == 0:
        return "YES"
    if thumb == 1 and index == 0 and middle == 0 and ring == 0 and pinky == 1:
        return "THANKS"
    if index == 1 and middle == 1 and ring == 0:
        return "SMALLER"
    if thumb == 1 and index == 1 and pinky == 1:
        return "BIGGER"
    if thumb == 1 and index == 0 and middle == 0 and ring == 0 and pinky == 0:
        return "I DON'T LIKE IT"
    if pinky == 1 and index == 0 and middle == 0:
        return "I LIKE IT"
    if thumb == 0 and index == 1 and middle == 1 and ring == 1 and pinky == 1:
        return "BYE"

    return ""


def _draw_hand(frame, landmarks):
    h, w, _ = frame.shape
    points = []

    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    for c in HAND_CONNECTIONS:
        cv2.line(frame, points[c[0]], points[c[1]], (0, 255, 0), 2)


class SignLanguageProcessor(BaseProcessor):
    name = "sign_language"
    description = "Hand sign landmark detection"

    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path=os.path.join(_MODELS, "hand_landmarker.task")
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray) -> np.ndarray:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = self.landmarker.detect(mp_image)

            output = frame.copy()

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    _draw_hand(output, hand_landmarks)

                    fingers = _get_finger_states(hand_landmarks)
                    gesture = _recognize_gesture(fingers)

                    if gesture:
                        cv2.putText(output, gesture, (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

            return output
        except Exception as e:
            return frame

    def release(self):
        try:
            self.landmarker.close()
        except Exception:
            pass
