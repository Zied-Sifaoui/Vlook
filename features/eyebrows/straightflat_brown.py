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

# ── Light brown in BGR ────────────────────────────────────────────────────────
LIGHT_BROWN = np.array([20, 50, 80], dtype=np.float32)  # BGR = RGB(139, 90, 40)

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

def draw_straight_brow(frame, upper_pts, lower_pts, brow_color):
    h, w = frame.shape[:2]

    upper_pts = upper_pts[np.argsort(upper_pts[:, 0])]
    lower_pts = lower_pts[np.argsort(lower_pts[:, 0])]

    mean_upper_y = np.mean(upper_pts[:, 1])
    mean_lower_y = np.mean(lower_pts[:, 1])
    brow_gap     = mean_lower_y - mean_upper_y

    upper_flat = upper_pts.copy()
    lower_flat = lower_pts.copy()
    upper_flat[:, 1] = mean_upper_y
    lower_flat[:, 1] = mean_upper_y + brow_gap * 0.85

    n = 60
    upper_curve = smooth_curve(upper_flat, n)
    lower_curve = smooth_curve(lower_flat, n)
    taper = np.sin(np.linspace(0, np.pi, n)) ** 0.5
    mid_y = (upper_curve[:, 1] + lower_curve[:, 1]) / 2
    half  = (lower_curve[:, 1] - upper_curve[:, 1]) / 2
    tapered_upper = np.column_stack([upper_curve[:, 0], mid_y - half * taper])
    tapered_lower = np.column_stack([lower_curve[:, 0], mid_y + half * taper])

    contour = np.vstack([tapered_upper, tapered_lower[::-1]]).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [contour], 1.0)
    mask_blur = cv2.GaussianBlur(mask, (0, 0), sigmaX=5, sigmaY=4)

    core = np.zeros((h, w), dtype=np.float32)
    shrunk = (contour - contour.mean(axis=0)) * 0.72 + contour.mean(axis=0)
    cv2.fillPoly(core, [shrunk.astype(np.int32)], 1.0)
    core_blur = cv2.GaussianBlur(core, (0, 0), sigmaX=2, sigmaY=2)

    combined = np.clip(mask_blur * 0.5 + core_blur * 0.75, 0, 1)
    alpha3 = np.stack([combined * 0.75] * 3, axis=-1)
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
            for upper_idx, lower_idx in [
                (LEFT_BROW_UPPER, LEFT_BROW_LOWER),
                (RIGHT_BROW_UPPER, RIGHT_BROW_LOWER)
            ]:
                upper_pts = get_pts(face_landmarks, upper_idx, w, h)
                lower_pts = get_pts(face_landmarks, lower_idx, w, h)
                frame = draw_straight_brow(frame, upper_pts, lower_pts, LIGHT_BROWN)

    cv2.putText(frame, "Style: Straight & Flat (Korean) | Light Brown", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Straight Brow - Light Brown", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()