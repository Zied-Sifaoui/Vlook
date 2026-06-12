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
import os
import firebase_admin
from firebase_admin import credentials, firestore

# ═══════════════════════════════════════════════════════════════════
# 🔥  FIREBASE SETUP
# ═══════════════════════════════════════════════════════════════════
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
def get_active_patient_id():
    try:
        doc = db.collection('jetson_control').document('active_filter').get()
        if doc.exists:
            return doc.to_dict().get('patientId', None)
    except Exception as e:
        print(f"[Firebase] Error reading patientId: {e}")
    return None

def send_skin_readiness_to_firebase(patient_id, ready, count, coverage, reason):
    result = "Ready" if ready else "Not Ready"
    try:
        sessions = db.collection('sessions') \
            .where('patientId', '==', patient_id) \
            .order_by('timestamp', direction=firestore.Query.DESCENDING) \
            .limit(1) \
            .stream()

        session_updated = False
        for session in sessions:
            session.reference.update({
                'skinReadiness': result,
                'skinDetails': {
                    'severeLesionCount': count,
                    'coveragePercent': round(coverage, 2),
                    'reason': reason,
                }
            })
            print(f"[Firebase] Session updated -> skinReadiness: {result}")
            session_updated = True

        if not session_updated:
            print("[Firebase] No session found for this patient")

        db.collection('vitals').document(patient_id).set({
            'skinReadiness': result,
            'skinDetails': {
                'severeLesionCount': count,
                'coveragePercent': round(coverage, 2),
                'reason': reason,
            }
        }, merge=True)
        print(f"[Firebase] Vitals updated -> skinReadiness: {result}")

    except Exception as e:
        print(f"[Firebase] Error sending skin readiness: {e}")


# ═══════════════════════════════════════════════════════════════════
# ⚙️  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
USE_WEBCAM      = True
IMAGE_PATH      = r"patients/patient1.jpg"
LANDMARKER_PATH = r"models/face_landmarker.task"
WEIGHTS_PATH    = "unet_acne.pth"

MIN_NODULE_AREA         = 60
MAX_NODULE_AREA         = 8000
SAT_ABOVE_SKIN          = 40
MIN_BRIGHTNESS_CONTRAST = 10
SEVERE_COUNT_TO_FAIL    = 1

FIREBASE_UPDATE_INTERVAL = 30

# ═══════════════════════════════════════════════════════════════════
# 1.  U-Net
# ═══════════════════════════════════════════════════════════════════
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for f in features:
            self.downs.append(DoubleConv(in_channels, f)); in_channels = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1); x = self.ups[i + 1](x)
        return self.final_conv(x)

unet_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def run_unet(model, frame, device):
    h, w = frame.shape[:2]
    pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    t    = unet_transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        prob   = torch.sigmoid(model(t))[0, 0]
        binary = (prob > 0.45).cpu().numpy().astype(np.uint8) * 255
    return cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)

# ═══════════════════════════════════════════════════════════════════
# 2.  MediaPipe face + exclusion masks
# ═══════════════════════════════════════════════════════════════════
LIPS_IDX       = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,42,183,78]
LEFT_EYE_IDX   = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE_IDX  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
LEFT_BROW_IDX  = [70,63,105,66,107,55,65,52,53,46]
RIGHT_BROW_IDX = [336,296,334,293,300,276,283,282,295,285]
NOSE_IDX       = [1,2,98,327,326,97,99,240,235,64,294,460,370,94,141]
FACE_OVAL_IDX  = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]

def build_landmarker():
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(opts)

def get_face_mask(frame, landmarker):
    h, w   = frame.shape[:2]
    f_mask = np.zeros((h, w), dtype=np.uint8)
    e_mask = np.zeros((h, w), dtype=np.uint8)
    result = landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB,
                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    )
    if not result.face_landmarks:
        return f_mask, e_mask, 0, None

    lm  = result.face_landmarks[0]
    pts = np.array([[int(l.x * w), int(l.y * h)] for l in lm], dtype=np.int32)
    cv2.fillConvexPoly(f_mask, cv2.convexHull(pts), 255)

    for idx_group in [LIPS_IDX, LEFT_EYE_IDX, RIGHT_EYE_IDX,
                      LEFT_BROW_IDX, RIGHT_BROW_IDX, NOSE_IDX]:
        ex_pts = np.array(
            [[int(lm[i].x * w), int(lm[i].y * h)] for i in idx_group if i < len(lm)],
            dtype=np.int32
        )
        if len(ex_pts) > 2:
            cv2.fillConvexPoly(e_mask, cv2.convexHull(ex_pts), 255)

    e_mask = cv2.dilate(e_mask, np.ones((7, 7), np.uint8), iterations=1)

    oval_pts = np.array(
        [[int(lm[i].x * w), int(lm[i].y * h)] for i in FACE_OVAL_IDX if i < len(lm)],
        dtype=np.int32
    )
    (cx, cy), radius = cv2.minEnclosingCircle(oval_pts)
    face_circle = (int(cx), int(cy), int(radius))

    return f_mask, e_mask, 1, face_circle

# ═══════════════════════════════════════════════════════════════════
# 3.  Severe lesion detector
# ═══════════════════════════════════════════════════════════════════
def compute_skin_baseline(hsv, lab, face_mask, ex_mask):
    skin_region = cv2.bitwise_and(face_mask, cv2.bitwise_not(ex_mask))
    _, s_ch, v_ch = cv2.split(hsv)
    _, a_ch, _    = cv2.split(lab)

    not_red = cv2.bitwise_not(
        cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,   30, 30]), np.array([15,  255, 255])),
            cv2.inRange(hsv, np.array([165, 30, 30]), np.array([180, 255, 255]))
        )
    )
    clean = cv2.bitwise_and(skin_region, not_red)

    sats  = s_ch[clean > 0]
    avals = a_ch[clean > 0]

    mean_sat   = float(np.mean(sats))  if len(sats)  > 100 else 50.0
    mean_astar = float(np.mean(avals)) if len(avals) > 100 else 128.0
    return mean_sat, mean_astar


def detect_severe_lesions(frame, face_mask, ex_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    _, s_ch, v_ch = cv2.split(hsv)
    _, a_ch, _    = cv2.split(lab)

    mean_sat, mean_astar = compute_skin_baseline(hsv, lab, face_mask, ex_mask)

    red1    = cv2.inRange(hsv, np.array([0,   60, 40]), np.array([14,  255, 220]))
    red2    = cv2.inRange(hsv, np.array([166,  60, 40]), np.array([180, 255, 220]))
    hsv_red = cv2.bitwise_or(red1, red2)

    a_thresh = int(min(mean_astar + 10, 165))
    lab_red  = (a_ch > a_thresh).astype(np.uint8) * 255

    raw_red = cv2.bitwise_or(hsv_red, lab_red)

    sat_gate = (s_ch > (mean_sat + SAT_ABOVE_SKIN)).astype(np.uint8) * 255
    raw_red  = cv2.bitwise_and(raw_red, sat_gate)
    raw_red  = cv2.bitwise_and(raw_red, (v_ch > 35).astype(np.uint8) * 255)
    raw_red  = cv2.bitwise_and(raw_red, face_mask)

    if cv2.countNonZero(ex_mask) > 0:
        raw_red = cv2.bitwise_and(raw_red, cv2.bitwise_not(ex_mask))

    raw_red = cv2.morphologyEx(raw_red, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    raw_red = cv2.morphologyEx(raw_red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    severe_mask = np.zeros_like(raw_red)
    lesion_list = []

    contours, _ = cv2.findContours(raw_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_NODULE_AREA < area < MAX_NODULE_AREA):
            continue

        cnt_mask   = np.zeros_like(raw_red)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

        lesion_sat = float(np.mean(s_ch[cnt_mask > 0]))
        if lesion_sat < (mean_sat + SAT_ABOVE_SKIN * 0.75):
            continue

        surr_mask = cv2.dilate(cnt_mask, np.ones((13, 13), np.uint8)) - cnt_mask
        surr_mask = cv2.bitwise_and(surr_mask, face_mask)
        lesion_v  = v_ch[cnt_mask > 0]
        surr_v    = v_ch[surr_mask > 0]
        if len(lesion_v) == 0 or len(surr_v) == 0:
            continue
        if abs(float(np.mean(lesion_v)) - float(np.mean(surr_v))) < MIN_BRIGHTNESS_CONTRAST:
            continue

        lesion_a = float(np.mean(a_ch[cnt_mask > 0]))
        if lesion_a < (mean_astar + 8):
            continue

        if area > 800:
            grade = "CYST / NODULE"
        elif area > 300:
            grade = "LARGE PUSTULE"
        else:
            grade = "INFLAMED PAPULE"

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        lesion_list.append((cx, cy, area, grade))
        cv2.drawContours(severe_mask, [cnt], -1, 255, -1)

    return severe_mask, lesion_list


# ═══════════════════════════════════════════════════════════════════
# 4.  Optional U-Net integration
# ═══════════════════════════════════════════════════════════════════
def detect_acne(frame, face_mask, ex_mask, model, device, has_weights):
    severe_mask, lesion_list = detect_severe_lesions(frame, face_mask, ex_mask)

    if has_weights:
        unet_raw  = run_unet(model, frame, device)
        unet_mask = cv2.bitwise_and(unet_raw, face_mask)
        if cv2.countNonZero(ex_mask) > 0:
            unet_mask = cv2.bitwise_and(unet_mask, cv2.bitwise_not(ex_mask))
        severe_mask = cv2.bitwise_or(severe_mask, unet_mask)

    return severe_mask, lesion_list


# ═══════════════════════════════════════════════════════════════════
# 5.  Surgery readiness decision
# ═══════════════════════════════════════════════════════════════════
def assess_readiness(lesion_list, severe_mask, face_mask):
    face_px  = cv2.countNonZero(face_mask)
    acne_px  = cv2.countNonZero(severe_mask)
    coverage = (acne_px / face_px * 100) if face_px > 0 else 0.0

    severe_count = len(lesion_list)
    worst_grade  = ""
    if lesion_list:
        rank = {"CYST / NODULE": 3, "LARGE PUSTULE": 2, "INFLAMED PAPULE": 1}
        worst_grade = max(lesion_list, key=lambda x: rank.get(x[3], 0))[3]

    ready  = severe_count < SEVERE_COUNT_TO_FAIL
    reason = f"Active: {worst_grade}" if not ready else "No grade 3-4 lesions detected"
    return ready, severe_count, coverage, reason


# ═══════════════════════════════════════════════════════════════════
# 6.  Overlay rendering — circle only, no text
# ═══════════════════════════════════════════════════════════════════
def draw_overlay(frame, severe_mask, lesion_list, ready, count, coverage, reason, face_circle):
    out = frame.copy()

    # ── Green or red circle around face ──────────────────────────────────
    if face_circle is not None:
        cx, cy, radius = face_circle
        circle_color   = (0, 220, 0) if ready else (0, 0, 220)
        overlay = out.copy()
        cv2.circle(overlay, (cx, cy), radius + 10, circle_color, 18)
        cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)
        cv2.circle(out, (cx, cy), radius + 10, circle_color, 4)

    # ── Red highlights on lesions ─────────────────────────────────────────
    if cv2.countNonZero(severe_mask) > 0:
        hl = out.copy()
        hl[severe_mask > 0] = [0, 0, 255]
        cv2.addWeighted(hl, 0.5, out, 0.5, 0, out)

    rank_color = {"CYST / NODULE": (0, 0, 255),
                  "LARGE PUSTULE": (0, 80, 255),
                  "INFLAMED PAPULE": (0, 140, 255)}
    for (lx, ly, area, grade) in lesion_list:
        radius_l = max(12, int(np.sqrt(area / np.pi) * 1.4))
        color    = rank_color.get(grade, (0, 0, 255))
        cv2.circle(out, (lx, ly), radius_l, color, 2)
        cv2.putText(out, grade, (lx - radius_l, ly - radius_l - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    # ── Top banner ────────────────────────────────────────────────────────
    banner_color = (0, 200, 0) if ready else (0, 0, 220)
    status_text  = "READY FOR SURGERY" if ready else "NOT READY - SURGERY CONTRAINDICATED"
    cv2.rectangle(out, (0, 0), (out.shape[1], 90), (15, 15, 15), -1)
    cv2.putText(out, status_text, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, banner_color, 2, cv2.LINE_AA)
    cv2.putText(out, reason, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(out, f"Severe lesions: {count}   Coverage: {coverage:.2f}%",
                (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

    return out


# ═══════════════════════════════════════════════════════════════════
# 7.  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model       = UNet().to(device)
    has_weights = False
    if os.path.exists(WEIGHTS_PATH):
        try:
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
            model.eval(); has_weights = True
            print("[INFO] U-Net weights loaded.")
        except Exception as e:
            print(f"[WARNING] U-Net load failed: {e}. Using HSV/LAB only.")
    else:
        print(f"[WARNING] '{WEIGHTS_PATH}' not found. Using HSV/LAB detection only.")

    landmarker = build_landmarker()
    print("[INFO] System ready. Press Q to quit.\n")

    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Webcam not found"); landmarker.close(); return
    else:
        if not os.path.exists(IMAGE_PATH):
            print(f"Error: Image not found at {IMAGE_PATH}"); landmarker.close(); return
        frame = cv2.imread(IMAGE_PATH)
        if frame is None:
            print("Error: Could not load image"); landmarker.close(); return

    frame_count = 0
    last_ready  = None

    def check_scan_requested():
        try:
            doc = db.collection('jetson_control').document('active_filter').get()
            if doc.exists:
                return doc.to_dict().get('scanRequested', False)
        except:
            pass
        return False

    def clear_scan_request():
        try:
            db.collection('jetson_control').document('active_filter').update({
                'scanRequested': False
            })
        except:
            pass

    while True:
        if USE_WEBCAM:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame grab failed."); break

        face_mask, ex_mask, face_found, face_circle = get_face_mask(frame, landmarker)

        if face_found:
            severe_mask, lesion_list = detect_acne(
                frame, face_mask, ex_mask, model, device, has_weights)
            ready, count, coverage, reason = assess_readiness(
                lesion_list, severe_mask, face_mask)
            display = draw_overlay(
                frame, severe_mask, lesion_list, ready, count, coverage, reason, face_circle)

            frame_count += 1
            if frame_count % FIREBASE_UPDATE_INTERVAL == 0:
                if check_scan_requested():
                    patient_id = get_active_patient_id()
                    if patient_id:
                        print(f"[Firebase] Scan requested! Sending result for: {patient_id}")
                        send_skin_readiness_to_firebase(
                            patient_id, ready, count, coverage, reason)
                        clear_scan_request()
                        last_ready = ready
                    else:
                        print("[Firebase] No active patient found")

        else:
            display = frame.copy()
            cv2.putText(display, "No face detected - please look at the camera",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Surgery Readiness - Severe Acne Detector", display)

        if not USE_WEBCAM:
            cv2.waitKey(0); break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if USE_WEBCAM:
        cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()