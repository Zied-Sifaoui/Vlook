import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =============================
# PATHS
# =============================
MODEL_PATH = "C:/Users/pc/Desktop/FacialLandmarks2/models/face_landmarker.task"
MOUTH_PATH = "C:/Users/pc/Desktop/FacialLandmarks2/assets/images/mouth.jp" \
"g"

# =============================
# Tuning knobs
# =============================
PAD_X       = 0.30
PAD_Y       = 0.40
MAX_OPACITY = 0.82

# =============================
# Manual offset controls (WASD + - =)
# =============================
offset_x    = 0      # A = left,  D = right
offset_y    = 0      # W = up,    S = down
scale_delta = 0      # - = smaller, = (plus) = bigger
MOVE_STEP   = 3      # pixels per keypress
SCALE_STEP  = 0.05   # scale factor per keypress

# =============================
# Load & prepare mouth image
# =============================
mouth_src = cv2.imread(MOUTH_PATH, cv2.IMREAD_UNCHANGED)
if mouth_src is None:
    print("Error: mouth image not found at", MOUTH_PATH)
    exit()
if mouth_src.shape[2] == 3:
    mouth_src = cv2.cvtColor(mouth_src, cv2.COLOR_BGR2BGRA)

# =============================
# Load Face Landmarker
# =============================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

# =============================
# Webcam
# =============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

MOUTH_IDS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
             324, 318, 402, 317, 14, 87, 178, 88, 95, 78]


# =============================
# Color transfer in LAB space
# =============================
def match_color_lab(src_bgr, dst_bgr, alpha_mask):
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    visible = alpha_mask > 0.1
    if not visible.any():
        return src_bgr
    for c in range(3):
        s_vals = src_lab[:, :, c][visible]
        d_vals = dst_lab[:, :, c][visible]
        s_mean, s_std = s_vals.mean(), s_vals.std() + 1e-6
        d_mean, d_std = d_vals.mean(), d_vals.std() + 1e-6
        src_lab[:, :, c] = (src_lab[:, :, c] - s_mean) * (d_std / s_std) + d_mean
    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


# =============================
# Soft ellipse alpha mask
# =============================
def make_ellipse_alpha(h, w, feather):
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    cv2.ellipse(mask, (cx, cy), (max(cx - 4, 1), max(cy - 4, 1)),
                0, 0, 360, 1.0, -1)
    if feather > 0:
        ksize = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather * 0.5)
    return mask


# =============================
# Safe alpha blend
# =============================
def alpha_blend(background, overlay_bgr, alpha_2d, x, y):
    H, W = background.shape[:2]
    oh, ow = overlay_bgr.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, W), min(y + oh, H)
    if x2 <= x1 or y2 <= y1:
        return background
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    roi = background[y1:y2, x1:x2].astype(np.float32)
    ov  = overlay_bgr[sy1:sy2, sx1:sx2].astype(np.float32)
    a   = alpha_2d[sy1:sy2, sx1:sx2, np.newaxis]
    blended = (ov * a + roi * (1.0 - a)).clip(0, 255).astype(np.uint8)
    out = background.copy()
    out[y1:y2, x1:x2] = blended
    return out


# =============================
# Draw HUD overlay
# =============================
def draw_hud(frame, offset_x, offset_y, scale_delta):
    H, W = frame.shape[:2]

    # Semi-transparent dark pill at bottom
    bar_h = 52
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H - bar_h), (W, H), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    controls = [
        ("W/S", "up/down"),
        ("A/D", "left/right"),
        ("-/=", "size"),
        ("R",   "reset"),
        ("ESC", "quit"),
    ]

    x_cursor = 18
    y_mid    = H - bar_h // 2 + 1

    for key, desc in controls:
        # Key badge
        (kw, kh), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        pad = 6
        bx1, by1 = x_cursor, y_mid - kh // 2 - pad
        bx2, by2 = x_cursor + kw + pad * 2, y_mid + kh // 2 + pad
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (70, 70, 70), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (130, 130, 130), 1)
        cv2.putText(frame, key, (x_cursor + pad, y_mid + kh // 2 - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        x_cursor = bx2 + 5

        # Description
        (dw, _), _ = cv2.getTextSize(desc, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(frame, desc, (x_cursor, y_mid + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
        x_cursor += dw + 22

    # Live values (top-left)
    scale_pct = int((1.0 + scale_delta) * 100)
    info = f"offset  x:{offset_x:+d}  y:{offset_y:+d}    scale:{scale_pct}%"
    cv2.putText(frame, info, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 180), 1, cv2.LINE_AA)


# =============================
# MAIN LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        pts = np.array([
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in MOUTH_IDS
        ])

        bx, by, bw, bh = cv2.boundingRect(pts)

        px = int(bw * PAD_X)
        py = int(bh * PAD_Y)
        rx = max(bx - px, 0)
        ry = max(by - py, 0)
        rw = min(bw + 2 * px, w - rx)
        rh = min(bh + 2 * py, h - ry)

        if rw < 8 or rh < 8:
            draw_hud(frame, offset_x, offset_y, scale_delta)
            cv2.imshow("Mouth Swap", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # --- Apply scale delta ---
        scale  = max(0.3, 1.0 + scale_delta)
        rw_s   = max(int(rw * scale), 8)
        rh_s   = max(int(rh * scale), 8)
        # Re-center after scaling
        rx_s   = rx - (rw_s - rw) // 2
        ry_s   = ry - (rh_s - rh) // 2

        # --- Apply position offset ---
        rx_s += offset_x
        ry_s += offset_y

        # --- Aspect-ratio-preserving resize ---
        src_h, src_w = mouth_src.shape[:2]
        aspect = src_w / src_h
        if rw_s / rh_s > aspect:
            fit_h = rh_s
            fit_w = int(fit_h * aspect)
        else:
            fit_w = rw_s
            fit_h = int(fit_w / aspect)
        fit_w = max(fit_w, 4)
        fit_h = max(fit_h, 4)

        resized   = cv2.resize(mouth_src, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
        fake_bgr  = resized[:, :, :3]
        png_alpha = resized[:, :, 3].astype(np.float32) / 255.0

        # --- Soft ellipse alpha ---
        feather = max(int(min(fit_w, fit_h) * 0.18), 3)
        if feather % 2 == 0:
            feather += 1
        ellipse_alpha  = make_ellipse_alpha(fit_h, fit_w, feather)
        combined_alpha = ellipse_alpha * png_alpha * MAX_OPACITY

        # --- Final placement (centered in scaled+offset region) ---
        off_x = rx_s + (rw_s - fit_w) // 2
        off_y = ry_s + (rh_s - fit_h) // 2

        roi_region = frame[
            max(off_y, 0):max(off_y, 0) + fit_h,
            max(off_x, 0):max(off_x, 0) + fit_w
        ]
        if roi_region.shape[:2] == (fit_h, fit_w):
            fake_bgr = match_color_lab(fake_bgr, roi_region, combined_alpha)

        frame = alpha_blend(frame, fake_bgr, combined_alpha, off_x, off_y)

    draw_hud(frame, offset_x, offset_y, scale_delta)
    cv2.imshow("Mouth Swap", frame)

    # =============================
    # Keyboard input
    # =============================
    key = cv2.waitKey(1) & 0xFF

    if key == 27:                        # ESC — quit
        break
    elif key == ord('w') or key == ord('W'):
        offset_y -= MOVE_STEP
    elif key == ord('s') or key == ord('S'):
        offset_y += MOVE_STEP
    elif key == ord('a') or key == ord('A'):
        offset_x -= MOVE_STEP
    elif key == ord('d') or key == ord('D'):
        offset_x += MOVE_STEP
    elif key == ord('-'):                # shrink
        scale_delta = round(max(-0.7, scale_delta - SCALE_STEP), 2)
    elif key == ord('='):                # grow  (= key, no shift needed)
        scale_delta = round(min(2.0,  scale_delta + SCALE_STEP), 2)
    elif key == ord('r') or key == ord('R'):   # reset all
        offset_x, offset_y, scale_delta = 0, 0, 0

cap.release()
cv2.destroyAllWindows()