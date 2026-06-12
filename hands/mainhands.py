import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os as _os

_BASE = _os.path.dirname(_os.path.abspath(__file__))
_MODELS = _os.path.join(_BASE, '..', 'models')

# -----------------------------
# 1️⃣ Charger le modèle
# -----------------------------
base_options = python.BaseOptions(
    model_asset_path=_os.path.join(_MODELS, "hand_landmarker.task")
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

# -----------------------------
# 2️⃣ Ouvrir la webcam
# -----------------------------
cap = cv2.VideoCapture(0)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir en RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convertir en MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp (obligatoire en VIDEO mode)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Détection
        result = landmarker.detect_for_video(mp_image, timestamp)

        # Afficher les landmarks
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for landmark in hand_landmarks:
                    h, w, _ = frame.shape
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow("Hand Landmarker - Tasks API", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
