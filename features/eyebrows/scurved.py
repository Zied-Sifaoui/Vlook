import cv2
import mediapipe as mp
import numpy as np
import time

MODEL_PATH = "C:/Users/pc/Desktop/FacialLandmarks2/models/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

LEFT_BROW_UPPER  = [70, 63, 105, 66, 107]
LEFT_BROW_LOWER  = [46, 53, 52,  65, 55]
RIGHT_BROW_UPPER = [300, 293, 334, 296, 336]
RIGHT_BROW_LOWER = [276, 283, 282, 295, 285]

def get_pts(face_landmarks, indices, w, h):
    return np.array(
        [(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in indices],
        dtype=np.float32
    )

def smooth_curve(pts, n=60):
    t_in  = np.linspace(0, 1, len(pts))
    t_out = np.linspace(0, 1, n)
    return np.column_stack([
        np.interp(t_out, t_in, pts[:, 0]),
        np.interp(t_out, t_in, pts[:, 1])
    ])

def sample_brow_color(frame, face_landmarks, w, h):
    samples = []
    for idx in [234, 454, 10]:
        lm = face_landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        patch = frame[max(0, cy-12):cy+12, max(0, cx-12):cx+12]
        if patch.size > 0:
            samples.append(np.mean(patch, axis=(0, 1)))
    if not samples:
        return np.array([25, 35, 50], dtype=np.float32)
    skin = np.mean(samples, axis=0)
    gray = np.dot(skin, [0.114, 0.587, 0.299])
    brow = skin * 0.25 + np.array([gray, gray, gray]) * 0.10
    return np.clip(brow, 8, 110).astype(np.float32)

def draw_scurve_brow(frame, upper_pts, lower_pts, brow_color):
    h, w = frame.shape[:2]

    upper_pts = upper_pts[np.argsort(upper_pts[:, 0])]
    lower_pts = lower_pts[np.argsort(lower_pts[:, 0])]

    n = 60
    upper_curve = smooth_curve(upper_pts, n)
    lower_curve = smooth_curve(lower_pts, n)

    # ── S-CURVE: sinusoidal wave — dips at head, rises at peak, dips at tail ──
    t = np.linspace(0, 1, n)
    brow_height = np.mean(np.abs(upper_curve[:, 1] - lower_curve[:, 1]))

    # S-shape: low start → high peak → slight dip at tail
    s_wave = (
        np.sin(t * np.pi) * 0.65           # main arch
        - np.sin(t * 2 * np.pi) * 0.22     # S counter-curve
    )
    lift_px = s_wave * brow_height * 0.9

    arched_upper = upper_curve.copy()
    arched_upper[:, 1] -= lift_px

    # Taper — thinner at both ends, fuller at arch peak
    taper = np.sin(np.linspace(0, np.pi, n)) ** 0.55
    mid_y = (arched_upper[:, 1] + lower_curve[:, 1]) / 2
    half  = (lower_curve[:, 1] - arched_upper[:, 1]) / 2
    tapered_upper = np.column_stack([upper_curve[:, 0], mid_y - half * taper])
    tapered_lower = np.column_stack([lower_curve[:, 0], mid_y + half * taper])

    contour = np.vstack([tapered_upper, tapered_lower[::-1]]).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [contour], 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5, sigmaY=4)

    core = np.zeros((h, w), dtype=np.float32)
    shrunk = (contour - contour.mean(axis=0)) * 0.70 + contour.mean(axis=0)
    cv2.fillPoly(core, [shrunk.astype(np.int32)], 1.0)
    core = cv2.GaussianBlur(core, (0, 0), sigmaX=2, sigmaY=2)

    combined = np.clip(mask * 0.50 + core * 0.80, 0, 1)
    alpha3 = np.stack([combined * 0.80] * 3, axis=-1)  # bolder for fashion
    color_layer = np.full_like(frame, brow_color.astype(np.uint8))
    frame[:] = np.clip(frame * (1 - alpha3) + color_layer * alpha3, 0, 255).astype(np.uint8)
    return frame

start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_timestamp_ms = int((time.time() - start_time) * 1000)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            h, w, _ = frame.shape
            brow_color = sample_brow_color(frame, face_landmarks, w, h)
            for upper_idx, lower_idx in [
                (LEFT_BROW_UPPER, LEFT_BROW_LOWER),
                (RIGHT_BROW_UPPER, RIGHT_BROW_LOWER)
            ]:
                upper_pts = get_pts(face_landmarks, upper_idx, w, h)
                lower_pts = get_pts(face_landmarks, lower_idx, w, h)
                frame = draw_scurve_brow(frame, upper_pts, lower_pts, brow_color)

    cv2.putText(frame, "Style: S-Curve (Bold Fashion)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("S-Curve Brow", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()