import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import urllib.request

_BASE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_BASE, '..', '..', 'models')

# ==========================================
# 📦 Auto Download Model
# ==========================================
MODEL_PATH = os.path.join(_MODELS, "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

# ==========================================
# 💋 Lip Enlargement (Upper + Lower Separate)
# ==========================================
def enlarge_lips(frame, landmarks, strength=0.4):
    h, w = frame.shape[:2]

    # Upper lip landmarks
    upper_idx = [61,185,40,39,37,0,267,269,270,409,291]

    # Lower lip landmarks
    lower_idx = [61,146,91,181,84,17,314,405,321,375,291]

    def process_lip(indices):
        lip_points = []
        for idx in indices:
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

        # Create coordinate grid
        map_x, map_y = np.meshgrid(np.arange(rw), np.arange(rh))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        cx = rw // 2
        cy = rh // 2

        dx = map_x - cx
        dy = map_y - cy

        dist = np.sqrt(dx**2 + dy**2)
        radius = max(ww, hh)

        mask_float = roi_mask.astype(np.float32) / 255.0

        # Radial scaling only inside lip mask
        scale = 1 - strength * mask_float * (1 - dist / radius)
        scale = np.clip(scale, 0.65, 1.0)

        map_x = cx + dx * scale
        map_y = cy + dy * scale

        warped = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        # Smooth blending
        blur_mask = cv2.GaussianBlur(mask_float, (31, 31), 0)
        blur_mask = blur_mask[:, :, np.newaxis]

        blended = warped * blur_mask + roi * (1 - blur_mask)

        frame[y1:y2, x1:x2] = blended.astype(np.uint8)

    # Apply to upper and lower lips separately
    process_lip(upper_idx)
    process_lip(lower_idx)

    return frame

# ==========================================
# 🎥 MAIN
# ==========================================
def main():
    ensure_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    strength = 0.4
    print("Controls: +  -  q")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = landmarker.detect_for_video(mp_img, int(time.time()*1000))

        if results.face_landmarks:
            frame = enlarge_lips(frame, results.face_landmarks[0], strength)

        cv2.putText(frame, f"Strength: {strength:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Lip Enlargement", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            strength = min(0.9, strength + 0.1)
        elif key in (ord('-'), ord('_')):
            strength = max(0.0, strength - 0.1)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()