import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import speech_recognition as sr

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
FACE_MODEL = r"C:\Users\pc\Desktop\FacialLandmarks2\models\face_landmarker.task"
HAND_MODEL = r"C:\Users\pc\Desktop\FacialLandmarks2\models\hand_landmarker.task"

# ─────────────────────────────────────────────
# Shared speech state (thread-safe)
# ─────────────────────────────────────────────
speech_lock    = threading.Lock()
speech_text    = ""
speech_status  = "Listening..."   # shown in HUD

# ─────────────────────────────────────────────
# Speech thread
# ─────────────────────────────────────────────
def speech_worker():
    global speech_text, speech_status
    recognizer = sr.Recognizer()
    mic        = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    while True:
        try:
            with mic as source:
                with speech_lock:
                    speech_status = "Listening..."
                audio = recognizer.listen(source, phrase_time_limit=5)

            with speech_lock:
                speech_status = "Processing..."

            text = recognizer.recognize_google(audio)

            with speech_lock:
                speech_text   = text
                speech_status = "Got it!"

        except sr.UnknownValueError:
            with speech_lock:
                speech_status = "Couldn't understand"
        except sr.RequestError:
            with speech_lock:
                speech_status = "API error"
        except Exception:
            pass

# Start speech thread as daemon so it dies when main exits
t = threading.Thread(target=speech_worker, daemon=True)
t.start()

# ─────────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
FaceLandmarker     = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

face_options = FaceLandmarkerOpts(
    base_options=BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

hand_options = HandLandmarkerOpts(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# ─────────────────────────────────────────────
# Eyebrow landmark indices
# ─────────────────────────────────────────────
LEFT_BROW_UPPER  = [70, 63, 105, 66, 107]
LEFT_BROW_LOWER  = [46, 53,  52, 65,  55]
RIGHT_BROW_UPPER = [300, 293, 334, 296, 336]
RIGHT_BROW_LOWER = [276, 283, 282, 295, 285]

# ─────────────────────────────────────────────
# Hand connections
# ─────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ─────────────────────────────────────────────
# Eyebrow helpers
# ─────────────────────────────────────────────
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

def draw_rounded_brow(frame, upper_pts, lower_pts, brow_color):
    h, w = frame.shape[:2]
    upper_pts = upper_pts[np.argsort(upper_pts[:, 0])]
    lower_pts = lower_pts[np.argsort(lower_pts[:, 0])]

    n = 60
    upper_curve = smooth_curve(upper_pts, n)
    lower_curve = smooth_curve(lower_pts, n)

    t = np.linspace(0, 1, n)
    round_lift  = np.exp(-((t - 0.42) ** 2) / (2 * 0.18 ** 2))
    brow_height = np.mean(np.abs(upper_curve[:, 1] - lower_curve[:, 1]))
    lift_px     = round_lift * brow_height * 0.55

    arched_upper        = upper_curve.copy()
    arched_upper[:, 1] -= lift_px

    taper         = 0.6 + 0.4 * np.sin(np.linspace(0, np.pi, n)) ** 0.4
    mid_y         = (arched_upper[:, 1] + lower_curve[:, 1]) / 2
    half          = (lower_curve[:, 1] - arched_upper[:, 1]) / 2
    tapered_upper = np.column_stack([upper_curve[:, 0], mid_y - half * taper])
    tapered_lower = np.column_stack([lower_curve[:, 0], mid_y + half * taper])

    contour = np.vstack([tapered_upper, tapered_lower[::-1]]).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [contour], 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=6, sigmaY=5)

    core   = np.zeros((h, w), dtype=np.float32)
    shrunk = (contour - contour.mean(axis=0)) * 0.68 + contour.mean(axis=0)
    cv2.fillPoly(core, [shrunk.astype(np.int32)], 1.0)
    core = cv2.GaussianBlur(core, (0, 0), sigmaX=3, sigmaY=2)

    combined     = np.clip(mask * 0.50 + core * 0.72, 0, 1)
    alpha3       = np.stack([combined * 0.70] * 3, axis=-1)
    color_layer  = np.full_like(frame, brow_color.astype(np.uint8))
    frame[:]     = np.clip(frame * (1 - alpha3) + color_layer * alpha3, 0, 255).astype(np.uint8)
    return frame

# ─────────────────────────────────────────────
# Hand helpers
# ─────────────────────────────────────────────
def get_finger_states(landmarks):
    fingers = []
    fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
    for tip, dip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(1 if landmarks[tip].y < landmarks[dip].y else 0)
    return fingers

def recognize_gesture(fingers):
    thumb, index, middle, ring, pinky = fingers
    if   thumb==1 and index==1 and middle==1 and ring==1 and pinky==1: return "HELLO"
    elif thumb==0 and index==0 and middle==0 and ring==0 and pinky==0: return "NO"
    elif thumb==0 and index==1 and middle==0 and ring==0 and pinky==0: return "YES"
    elif thumb==1 and index==0 and middle==0 and ring==0 and pinky==1: return "THANKS"
    elif index==1 and middle==1 and ring==0:                            return "SMALLER"
    elif thumb==1 and index==1 and pinky==1:                            return "BIGGER"
    elif thumb==1 and index==0 and middle==0 and ring==0 and pinky==0: return "I DON'T LIKE IT"
    elif pinky==1 and index==0 and middle==0:                           return "I LIKE IT"
    elif thumb==0 and index==1 and middle==1 and ring==1 and pinky==1: return "BYE"
    return ""

def draw_hand(frame, landmarks):
    h, w, _ = frame.shape
    points = []
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)

# ─────────────────────────────────────────────
# Text-wrap helper for speech subtitle bar
# ─────────────────────────────────────────────
def wrap_text(text, max_chars=55):
    """Split text into lines of max_chars."""
    words  = text.split()
    lines  = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current += (" " if current else "") + word
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [""]

# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
cap        = cv2.VideoCapture(0)
start_time = time.time()
prev_time  = start_time

# Height of the subtitle bar at the bottom (px)
SUBTITLE_H = 90

with FaceLandmarker.create_from_options(face_options) as face_lm, \
     HandLandmarker.create_from_options(hand_options) as hand_lm:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cam_h, cam_w = frame.shape[:2]
        frame_timestamp_ms = int((time.time() - start_time) * 1000)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        face_result = face_lm.detect_for_video(mp_image, frame_timestamp_ms)
        hand_result = hand_lm.detect_for_video(mp_image, frame_timestamp_ms)

        # ── Eyebrows ──────────────────────────────────────────────────────
        if face_result.face_landmarks:
            for face_landmarks in face_result.face_landmarks:
                h, w, _ = frame.shape
                brow_color = sample_brow_color(frame, face_landmarks, w, h)
                for upper_idx, lower_idx in [
                    (LEFT_BROW_UPPER,  LEFT_BROW_LOWER),
                    (RIGHT_BROW_UPPER, RIGHT_BROW_LOWER)
                ]:
                    upper_pts = get_pts(face_landmarks, upper_idx, w, h)
                    lower_pts = get_pts(face_landmarks, lower_idx, w, h)
                    frame = draw_rounded_brow(frame, upper_pts, lower_pts, brow_color)

        # ── Hand gestures ─────────────────────────────────────────────────
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                draw_hand(frame, hand_landmarks)
                fingers = get_finger_states(hand_landmarks)
                gesture = recognize_gesture(fingers)
                if gesture:
                    cv2.putText(frame, gesture, (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

        # ── HUD (top-left) ────────────────────────────────────────────────
        now       = time.time()
        fps       = 1 / (now - prev_time + 1e-9)
        prev_time = now

        cv2.putText(frame, "Style: Rounded Brow", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2, cv2.LINE_AA)

        # ── Subtitle bar (bottom) ─────────────────────────────────────────
        # Read shared speech state
        with speech_lock:
            current_text   = speech_text
            current_status = speech_status

        # Semi-transparent dark bar
        bar = np.zeros((SUBTITLE_H, cam_w, 3), dtype=np.uint8)
        bar[:] = (20, 20, 20)

        # Mic status dot (green = listening, yellow = processing)
        dot_color = (0, 220, 0) if current_status == "Listening..." else (0, 200, 255)
        cv2.circle(bar, (18, SUBTITLE_H // 2), 7, dot_color, -1)
        cv2.putText(bar, current_status, (32, SUBTITLE_H // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Wrap and render the spoken text
        lines = wrap_text(current_text, max_chars=70)
        # Show last 2 lines max
        lines = lines[-2:]
        y_offsets = [28, 58] if len(lines) == 2 else [45]
        for line, yo in zip(lines, y_offsets):
            cv2.putText(bar, line, (120, yo),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 180), 2, cv2.LINE_AA)

        # Blend bar onto frame bottom
        alpha = 0.82
        combined = frame.copy()
        combined[cam_h - SUBTITLE_H:cam_h, :] = cv2.addWeighted(
            frame[cam_h - SUBTITLE_H:cam_h, :], 1 - alpha,
            bar, alpha, 0
        )
        frame = combined

        cv2.imshow("Brow + Gesture + Speech", frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()