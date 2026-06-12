import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque


# ─────────────────────────────────────────────
# 1. U-Net Architecture
# ─────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
        return self.final_conv(x)


# ─────────────────────────────────────────────
# 2. MediaPipe Face Landmarker (Task API)
# ─────────────────────────────────────────────
LANDMARKER_PATH = r"C:\Users\pc\Desktop\FacialLandmarks2\models\face_landmarker.task"

# Landmark index groups (478-point model)
LIPS_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415,
    310, 311, 312, 13, 82, 81, 42, 183, 78
]
LEFT_EYE_IDX  = [33, 7, 163, 144, 145, 153, 154, 155,
                  133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249,
                  263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW_IDX  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW_IDX = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
NOSE_IDX       = [1, 2, 98, 327, 326, 97, 99, 240,
                  235, 64, 294, 460, 370, 94, 141]


def build_landmarker():
    base_opts = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    opts = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def draw_exclusion_zone(mask, landmarks, indices, h, w, expand_px):
    pts = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)]
         for i in indices if i < len(landmarks)],
        dtype=np.int32
    )
    if len(pts) < 3:
        return
    hull = cv2.convexHull(pts)
    M    = cv2.moments(hull)
    if M["m00"] == 0:
        cv2.fillConvexPoly(mask, hull, 255)
        return
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    expanded = []
    for p in hull:
        px, py = p[0]
        dx, dy = px - cx, py - cy
        norm   = max(1.0, np.sqrt(dx**2 + dy**2))
        expanded.append([[
            px + int(expand_px * dx / norm),
            py + int(expand_px * dy / norm)
        ]])
    cv2.fillConvexPoly(mask, np.array(expanded, dtype=np.int32), 255)


def get_face_zone_masks(frame, landmarker, timestamp_ms):
    """
    Returns (face_mask, exclusion_mask) using the Task API landmarker.
    """
    h, w      = frame.shape[:2]
    face_mask = np.zeros((h, w), dtype=np.uint8)
    ex_mask   = np.zeros((h, w), dtype=np.uint8)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.face_landmarks:
        return face_mask, ex_mask

    landmarks = result.face_landmarks[0]   # first face

    # Full face convex hull
    all_pts = np.array(
        [[int(lm.x * w), int(lm.y * h)] for lm in landmarks],
        dtype=np.int32
    )
    cv2.fillConvexPoly(face_mask, cv2.convexHull(all_pts), 255)

    # Exclusion zones
    draw_exclusion_zone(ex_mask, landmarks, LIPS_IDX,       h, w, expand_px=14)
    draw_exclusion_zone(ex_mask, landmarks, LEFT_EYE_IDX,   h, w, expand_px=10)
    draw_exclusion_zone(ex_mask, landmarks, RIGHT_EYE_IDX,  h, w, expand_px=10)
    draw_exclusion_zone(ex_mask, landmarks, LEFT_BROW_IDX,  h, w, expand_px=6)
    draw_exclusion_zone(ex_mask, landmarks, RIGHT_BROW_IDX, h, w, expand_px=6)
    draw_exclusion_zone(ex_mask, landmarks, NOSE_IDX,       h, w, expand_px=6)

    return face_mask, ex_mask


# ─────────────────────────────────────────────
# 3. Adaptive skin calibrator
# ─────────────────────────────────────────────
class SkinCalibrator:
    def __init__(self, update_interval=30):
        self.interval    = update_interval
        self.frame_count = 0
        self.skin_a_mean = 138.0
        self.skin_l_mean = 160.0
        self.skin_a_std  =   6.0
        self.skin_l_std  =  12.0

    def update(self, frame, face_mask, ex_mask):
        self.frame_count += 1
        if self.frame_count % self.interval != 0:
            return
        safe = cv2.bitwise_and(face_mask, cv2.bitwise_not(ex_mask))
        if cv2.countNonZero(safe) < 100:
            return
        lab    = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        pixels = lab[safe > 0].astype(np.float32)
        self.skin_a_mean = float(np.median(pixels[:, 1]))
        self.skin_l_mean = float(np.median(pixels[:, 0]))
        self.skin_a_std  = float(np.std(pixels[:, 1]))
        self.skin_l_std  = float(np.std(pixels[:, 0]))

    def scar_lab_mask(self, frame):
        lab      = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        a_ch     = lab[:, :, 1].astype(np.float32)
        l_ch     = lab[:, :, 0].astype(np.float32)
        a_thresh = self.skin_a_mean + 2.2 * max(self.skin_a_std, 3.0)
        l_min    = self.skin_l_mean - 3.0 * max(self.skin_l_std, 5.0)
        
        l_max    = self.skin_l_mean + 2.0 * max(self.skin_l_std, 5.0)
        return ((a_ch > a_thresh) &
                (l_ch > l_min)    &
                (l_ch < l_max)).astype(np.uint8) * 255


# ─────────────────────────────────────────────
# 4. Persistence tracker
# ─────────────────────────────────────────────
class PersistenceTracker:
    def __init__(self, history_len=5, min_persist=3):
        self.history     = deque(maxlen=history_len)
        self.min_persist = min_persist

    def update(self, binary_mask):
        self.history.append(binary_mask.astype(np.uint8))

    def get_stable_mask(self):
        if len(self.history) < self.min_persist:
            return np.zeros_like(self.history[0]) if self.history \
                   else None
        stack = np.stack(list(self.history), axis=0).astype(np.float32) / 255.0
        count = stack.sum(axis=0)
        return (count >= self.min_persist).astype(np.uint8) * 255


# ─────────────────────────────────────────────
# 5. Scar detector
# ─────────────────────────────────────────────
def detect_scar_mask(frame, calibrator, face_mask, ex_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    m1 = cv2.inRange(hsv, np.array([0,   50, 20]), np.array([12,  255, 210]))
    m2 = cv2.inRange(hsv, np.array([163, 50, 20]), np.array([180, 255, 210]))
    hsv_mask = cv2.bitwise_or(m1, m2)

    # Exclude dark pixels — beard/hair (V < 40)
    v_ch      = hsv[:, :, 2]
    dark_mask = (v_ch < 40).astype(np.uint8) * 255
    hsv_mask  = cv2.bitwise_and(hsv_mask, cv2.bitwise_not(dark_mask))

    lab_mask = calibrator.scar_lab_mask(frame)
    mask     = cv2.bitwise_and(hsv_mask, lab_mask)

    # Restrict to face, remove exclusion zones
    if cv2.countNonZero(face_mask) > 0:
        mask = cv2.bitwise_and(mask, face_mask)
    if cv2.countNonZero(ex_mask) > 0:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(ex_mask))

    # Clean noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8), iterations=2)

    # Keep only elongated blobs (scar lines)
    filtered   = np.zeros_like(mask)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40 or area > 5000:
            continue
        if len(cnt) >= 5:
            _, axes, _ = cv2.fitEllipse(cnt)
            ratio = max(axes) / (min(axes) + 1e-5)
            if ratio > 2.2:
                cv2.drawContours(filtered, [cnt], -1, 255, -1)
        else:
            if area > 150:
                cv2.drawContours(filtered, [cnt], -1, 255, -1)

    filtered = cv2.dilate(filtered, np.ones((4, 4), np.uint8), iterations=2)
    return filtered


# ─────────────────────────────────────────────
# 6. U-Net prediction
# ─────────────────────────────────────────────
IMG_SIZE  = (256, 256)
UV_THRESHOLD = 0.5
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def unet_scar_mask(model, image_bgr, device):
    h, w = image_bgr.shape[:2]
    pil  = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    t    = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
    mask = (prob > UV_THRESHOLD).astype(np.uint8) * 255
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


# ─────────────────────────────────────────────
# 7. Main loop
# ─────────────────────────────────────────────
def main():
    WEIGHTS_PATH = "unet_scar.pth"
    CAMERA_ID    = 0

    # ── Camera ──
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── U-Net ──
    model       = UNet().to(device)
    has_weights = False
    try:
        model.load_state_dict(torch.load(
            WEIGHTS_PATH, map_location=device, weights_only=False))
        model.eval()
        has_weights = True
        print("[INFO] Loaded U-Net weights")
    except FileNotFoundError:
        print("[WARN] No weights → adaptive HSV + MediaPipe landmarker")

    source = "U-Net" if has_weights else "HSV+Landmarker"

    # ── MediaPipe landmarker ──
    landmarker = build_landmarker()

    # ── Helpers ──
    calibrator  = SkinCalibrator(update_interval=30)
    tracker     = PersistenceTracker(history_len=5, min_persist=3)
    kernel      = np.ones((5, 5), np.uint8)
    ALPHA       = 0.4
    THRESHOLD_S = 0.35
    smoothed    = None
    saved_count = 0
    show_mask   = True
    timestamp   = 0

    print("\n[Controls]  M=toggle mask | S=save | Q=quit")
    print("[INFO] Keep face visible — calibrating skin tone...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp += 33   # ~30 fps in milliseconds

        # ── Landmarks & zone masks ──
        face_mask, ex_mask = get_face_zone_masks(frame, landmarker, timestamp)

        # ── Skin calibration ──
        calibrator.update(frame, face_mask, ex_mask)

        # ── Raw scar mask ──
        if has_weights:
            raw_mask = unet_scar_mask(model, frame, device)
            if cv2.countNonZero(ex_mask) > 0:
                raw_mask = cv2.bitwise_and(
                    raw_mask, cv2.bitwise_not(ex_mask))
        else:
            raw_mask = detect_scar_mask(
                frame, calibrator, face_mask, ex_mask)

        # ── Morphological refinement ──
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        raw_mask = cv2.dilate(raw_mask, kernel, iterations=1)

        # ── Temporal smoothing ──
        raw_f    = raw_mask.astype(np.float32) / 255.0
        smoothed = raw_f if smoothed is None \
                   else ALPHA * raw_f + (1 - ALPHA) * smoothed
        smooth_binary = (smoothed > THRESHOLD_S).astype(np.uint8) * 255

        # ── Persistence filter ──
        tracker.update(smooth_binary)
        stable_mask = tracker.get_stable_mask()

        # ── Inpaint ──
        if stable_mask is not None and cv2.countNonZero(stable_mask) > 0:
            result = cv2.inpaint(frame, stable_mask,
                                 inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        else:
            result = frame.copy()

        # ── Display ──
        cv2.putText(frame,  "Original",     (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, "Scar Removed", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Original",     frame)
        cv2.imshow("Scar Removed", result)

        if show_mask and stable_mask is not None:
            mask_vis = cv2.cvtColor(stable_mask, cv2.COLOR_GRAY2BGR)
            mask_vis[ex_mask > 0] = [200, 80, 0]   # exclusion zones in blue
            cv2.putText(
                mask_vis,
                f"px:{cv2.countNonZero(stable_mask)} "
                f"a={calibrator.skin_a_mean:.1f}±{calibrator.skin_a_std:.1f}",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)
            cv2.imshow(f"Mask ({source})", mask_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow(f"Mask ({source})")
        elif key == ord('s'):
            cv2.imwrite(f"snap_orig_{saved_count:03d}.jpg",   frame)
            cv2.imwrite(f"snap_result_{saved_count:03d}.jpg", result)
            if stable_mask is not None:
                cv2.imwrite(f"snap_mask_{saved_count:03d}.jpg", stable_mask)
            print(f"[INFO] Saved set {saved_count:03d}")
            saved_count += 1

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()