"""
server.py — V-Look Web Server
Run: python -m uvicorn server:app --host 0.0.0.0 --port 8000

Requirements:
    pip install fastapi uvicorn websockets opencv-python mediapipe numpy moderngl
"""

import asyncio
import base64
import json
import threading
import time
import traceback

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

from core.processor import FaceProcessor
from core.renderer  import ARRenderer
from core.geometry  import GeometryEngine
from core.face_mask import build_face_depth_map

app = FastAPI()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# App state
# ─────────────────────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.lock      = threading.Lock()
        self.face_data = {"found": False, "R_mat": None, "tvec": None, "landmarks": None}
        self.cap       = None
        self.processor = None
        self.renderer  = None
        self.geo       = None
        self.W = self.H = 0
        self.K         = None
        self._t0       = time.perf_counter()
        self.running   = False

        # ── Defaults ────────────────────────────────────────────────────────
        # After the renderer fix, Z=0 means mesh straddles the face plane.
        # offset_y: positive Y = DOWN in OpenCV, so negative = moves hair UP.
        # offset_z: positive = toward camera (in front), negative = behind face.
        #           Start at 0 — mesh is centered on face plane.
        self.params = {
            "style_idx":  0,
            "offset_x":   0.0,
            "offset_y": -350.0,
            "offset_z":   0.0,
            "mesh_scale": 450.0,
            "occ_margin":  15.0,   # hair↔face blend margin in mm
        }
        self.presets = [
            {"name": "Silver", "frame": (200, 200, 200), "lens": (55,  55,  55)},
            {"name": "Gold",   "frame": ( 30, 130, 200), "lens": (15,  75, 135)},
            {"name": "Black",  "frame": ( 15,  15,  15), "lens": ( 5,   5,   5)},
            {"name": "Rose",   "frame": (110,  85, 205), "lens": (55,  35, 115)},
            {"name": "Chrome", "frame": (185, 215, 235), "lens": (85, 115, 145)},
        ]

    def get_style(self):
        return self.presets[self.params["style_idx"] % len(self.presets)]


STATE = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe callback
# ─────────────────────────────────────────────────────────────────────────────
def on_result(result, image, ts_ms):
    geo  = STATE.geo
    W, H = STATE.W, STATE.H
    K    = STATE.K

    if not result.face_landmarks:
        with STATE.lock:
            STATE.face_data["found"] = False
        return

    dist  = np.zeros((4, 1), np.float32)
    lms   = result.face_landmarks[0]
    pts2d = np.array([[lms[i].x * W, lms[i].y * H]
                      for i in geo.SOLVE_IDX], dtype=np.float32)

    with STATE.lock:
        prev_r = None
        prev_t = STATE.face_data["tvec"]
        if STATE.face_data["R_mat"] is not None:
            prev_r, _ = cv2.Rodrigues(STATE.face_data["R_mat"])

    ok, rvec, tvec = geo.solve_pose(pts2d, K, dist, prev_rvec=prev_r, prev_tvec=prev_t)
    if not ok:
        return

    rvec *= -1
    R_new, _ = cv2.Rodrigues(rvec)

    with STATE.lock:
        STATE.face_data["R_mat"] = geo.smooth_rotation_matrix(
            STATE.face_data["R_mat"], R_new, alpha=0.12)
        prev_t = STATE.face_data["tvec"]
        if prev_t is not None:
            jump = np.linalg.norm(tvec - prev_t)
            if jump < 80.0:
                STATE.face_data["tvec"] = 0.15 * tvec + 0.85 * prev_t
        else:
            STATE.face_data["tvec"] = tvec.copy()
        STATE.face_data["landmarks"] = lms
        STATE.face_data["found"]     = True


# ─────────────────────────────────────────────────────────────────────────────
# Camera init
# ─────────────────────────────────────────────────────────────────────────────
def init_camera():
    STATE.geo       = GeometryEngine()
    STATE.renderer  = ARRenderer("assets/hair.obj", normalize_span=4.0)
    STATE.processor = FaceProcessor("models/face_landmarker.task", on_result)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera 0.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    STATE.cap     = cap
    STATE.W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    STATE.H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    STATE.K       = STATE.geo.get_camera_matrix(STATE.W, STATE.H)
    STATE._t0     = time.perf_counter()
    STATE.running = True
    print(f"[V-Look Web] Camera {STATE.W}x{STATE.H} ready")


# ─────────────────────────────────────────────────────────────────────────────
# Frame grab + render
# ─────────────────────────────────────────────────────────────────────────────
def grab_processed_frame():
    cap = STATE.cap
    if cap is None:
        return None, False

    ret, frame = cap.read()
    if not ret:
        return None, False

    frame = cv2.flip(frame, 1)
    ts_ms = int((time.perf_counter() - STATE._t0) * 1000)
    STATE.processor.process_frame(frame, ts_ms)

    face_found = False
    lms_f = p = None

    with STATE.lock:
        fd = STATE.face_data
        if fd["found"] and fd["R_mat"] is not None:
            face_found = True
            rvec_f, _ = cv2.Rodrigues(fd["R_mat"])
            tvec_f    = fd["tvec"]
            lms_f     = fd["landmarks"]
            p         = STATE.params.copy()

        if fd["found"] and fd["landmarks"] is not None:
            lms = fd["landmarks"]
            W, H = STATE.W, STATE.H
            nx = int(lms[1].x * W)
            ny = int(lms[1].y * H)
            cv2.circle(frame, (nx, ny), 4, (0, 255, 180), -1)
            fx = int(lms[10].x * W)
            fy = int(lms[10].y * H)
            cv2.circle(frame, (fx, fy), 4, (0, 200, 255), -1)

    if face_found:
        face_depth = None
        if lms_f is not None:
            face_depth = build_face_depth_map(
                lms_f, STATE.W, STATE.H, rvec_f, tvec_f, STATE.K)

        frame = STATE.renderer.render(
            frame, rvec_f, tvec_f, STATE.K,
            style          = STATE.get_style(),
            offset_x       = p["offset_x"],
            offset_y       = p["offset_y"],
            offset_z       = p["offset_z"],
            mesh_scale     = p["mesh_scale"],
            face_depth_map = face_depth,
            occ_margin     = 15.0,
        )

    dbg_lines = [
        f"X:{p['offset_x']:+.0f}  Y:{p['offset_y']:+.0f}  Z:{p['offset_z']:+.0f}  S:{p['mesh_scale']:.0f}" if face_found else "NO FACE",
    ]
    for i, line in enumerate(dbg_lines):
        cv2.putText(frame, line, (8, STATE.H - 12 - i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 210, 170), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes(), face_found


# ─────────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/presets")
async def get_presets():
    return {"presets": [p["name"] for p in STATE.presets]}


@app.get("/defaults")
async def get_defaults():
    """Let the browser know the server-side defaults on load."""
    return STATE.params


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — video
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/video")
async def ws_video(ws: WebSocket):
    await ws.accept()
    if not STATE.running:
        try:
            init_camera()
        except Exception as e:
            await ws.send_text(json.dumps({"error": str(e)}))
            await ws.close()
            return

    loop = asyncio.get_event_loop()
    try:
        while True:
            try:
                frame_bytes, face_found = await loop.run_in_executor(None, grab_processed_frame)
            except Exception as e:
                print(f"[V-Look Web] Frame grab error: {e}")
                await asyncio.sleep(0.1)
                continue
            if frame_bytes is None:
                await asyncio.sleep(0.01)
                continue
            b64 = base64.b64encode(frame_bytes).decode("ascii")
            await ws.send_text(json.dumps({"frame": b64, "face_found": face_found}))
            await asyncio.sleep(0.001)
    except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError):
        print("[V-Look Web] Client disconnected")
    except asyncio.CancelledError:
        print("[V-Look Web] Server shutting down")
    except Exception as e:
        print(f"[V-Look Web] WS error: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — controls
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/controls")
async def ws_controls(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            msg  = json.loads(data)
            with STATE.lock:
                for key in ("offset_x", "offset_y", "offset_z", "mesh_scale", "style_idx"):
                    if key in msg:
                        STATE.params[key] = float(msg[key]) if key != "style_idx" else int(msg[key])
            await ws.send_text(json.dumps({"ok": True}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[Controls WS] error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, init_camera)


@app.on_event("shutdown")
async def shutdown():
    STATE.running = False
    if STATE.processor:
        STATE.processor.close()
    if STATE.cap:
        STATE.cap.release()
    SCAR.close()
    BROW.close()
    print("[V-Look Web] Shutdown complete.")


# ═════════════════════════════════════════════════════════════════════════════
# SCAR REMOVAL — U-Net + HSV/LAB hybrid
# ═════════════════════════════════════════════════════════════════════════════
import sys, os
import mediapipe as mp
sys.path.insert(0, os.path.dirname(__file__))

from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

# ── U-Net Architecture ──────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms as _transforms
    from PIL import Image as _PILImage
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class _UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)
        for f in features:
            self.downs.append(_DoubleConv(in_channels, f))
            in_channels = f
        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(_DoubleConv(f * 2, f))
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

# ── Landmark index groups ───────────────────────────────────────────────────
_LIPS_IDX = [
    61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,
    14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,
    312,13,82,81,42,183,78
]
_LEFT_EYE  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
_RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
_LEFT_BROW  = [70,63,105,66,107,55,65,52,53,46]
_RIGHT_BROW = [336,296,334,293,300,276,283,282,295,285]
_NOSE_IDX   = [1,2,98,327,326,97,99,240,235,64,294,460,370,94,141]


def _draw_excl(mask, lms, indices, h, w, expand_px):
    pts = np.array([[int(lms[i].x*w), int(lms[i].y*h)]
                    for i in indices if i < len(lms)], dtype=np.int32)
    if len(pts) < 3: return
    hull = cv2.convexHull(pts)
    M = cv2.moments(hull)
    if M["m00"] == 0: cv2.fillConvexPoly(mask, hull, 255); return
    cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
    exp = []
    for p in hull:
        px,py = p[0]; dx,dy = px-cx, py-cy
        n = max(1.0, np.sqrt(dx**2+dy**2))
        exp.append([[px+int(expand_px*dx/n), py+int(expand_px*dy/n)]])
    cv2.fillConvexPoly(mask, np.array(exp, dtype=np.int32), 255)


class _SkinCalibrator:
    def __init__(self, update_interval=30):
        self.interval    = update_interval
        self.frame_count = 0
        self.skin_a_mean = 138.0
        self.skin_l_mean = 160.0
        self.skin_a_std  =   6.0
        self.skin_l_std  =  12.0
    def update(self, frame, face_mask, ex_mask):
        self.frame_count += 1
        if self.frame_count % self.interval != 0: return
        safe = cv2.bitwise_and(face_mask, cv2.bitwise_not(ex_mask))
        if cv2.countNonZero(safe) < 100: return
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        px  = lab[safe > 0].astype(np.float32)
        self.skin_a_mean = float(np.median(px[:, 1]))
        self.skin_l_mean = float(np.median(px[:, 0]))
        self.skin_a_std  = float(np.std(px[:, 1]))
        self.skin_l_std  = float(np.std(px[:, 0]))
    def scar_lab_mask(self, frame):
        lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        a_ch = lab[:,:,1].astype(np.float32)
        l_ch = lab[:,:,0].astype(np.float32)
        a_th = self.skin_a_mean + 2.2 * max(self.skin_a_std, 3.0)
        l_lo = self.skin_l_mean - 3.0 * max(self.skin_l_std, 5.0)
        l_hi = self.skin_l_mean + 2.0 * max(self.skin_l_std, 5.0)
        return ((a_ch > a_th) & (l_ch > l_lo) & (l_ch < l_hi)).astype(np.uint8) * 255


class _PersistenceTracker:
    def __init__(self, history_len=5, min_persist=3):
        self.history     = __import__("collections").deque(maxlen=history_len)
        self.min_persist = min_persist
    def update(self, m):
        self.history.append(m.astype(np.uint8))
    def get_stable(self):
        if len(self.history) < self.min_persist: return None
        s = np.stack(list(self.history), 0).astype(np.float32) / 255.0
        return (s.sum(0) >= self.min_persist).astype(np.uint8) * 255


def _detect_scar_mask(frame, calibrator, face_mask, ex_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1  = cv2.inRange(hsv, np.array([0,   50, 20]), np.array([12,  255, 210]))
    m2  = cv2.inRange(hsv, np.array([163, 50, 20]), np.array([180, 255, 210]))
    hsv_mask = cv2.bitwise_or(m1, m2)
    dark = (hsv[:,:,2] < 40).astype(np.uint8) * 255
    hsv_mask = cv2.bitwise_and(hsv_mask, cv2.bitwise_not(dark))

    lab_mask = calibrator.scar_lab_mask(frame)
    mask     = cv2.bitwise_and(hsv_mask, lab_mask)

    if cv2.countNonZero(face_mask) > 0: mask = cv2.bitwise_and(mask, face_mask)
    if cv2.countNonZero(ex_mask)  > 0: mask = cv2.bitwise_and(mask, cv2.bitwise_not(ex_mask))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((2,2), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    filtered = np.zeros_like(mask)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        if a < 40 or a > 5000: continue
        if len(cnt) >= 5:
            _, axes, _ = cv2.fitEllipse(cnt)
            if max(axes) / (min(axes) + 1e-5) > 2.2:
                cv2.drawContours(filtered, [cnt], -1, 255, -1)
        elif a > 150:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)
    return cv2.dilate(filtered, np.ones((4,4), np.uint8), iterations=2)


def _unet_scar_mask(model, img_bgr, device):
    h, w = img_bgr.shape[:2]
    tf = _transforms.Compose([
        _transforms.Resize((256,256)),
        _transforms.ToTensor(),
        _transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    pil = _PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    t   = tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
    m = (prob > 0.5).astype(np.uint8) * 255
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


# ── Scar session state ──────────────────────────────────────────────────────
class ScarState:
    def __init__(self):
        self.cap        = None
        self.landmarker = None
        self.running    = False
        self._t0        = time.perf_counter()
        self.timestamp  = 0
        self.calibrator = _SkinCalibrator(update_interval=30)
        self.tracker    = _PersistenceTracker(history_len=5, min_persist=3)
        self.model      = None
        self.device     = None
        self.has_weights = False
        self.smoothed   = None
        self.params = {
            "alpha":      0.4,
            "threshold":  0.35,
            "inpaint_r":  5,
            "show_mask":  False,
        }

    def _build_landmarker(self):
        base = _mp_python.BaseOptions(model_asset_path="models/face_landmarker.task")
        opts = _mp_vision.FaceLandmarkerOptions(
            base_options=base, running_mode=_mp_vision.RunningMode.VIDEO,
            num_faces=1, min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
            output_face_blendshapes=False, output_facial_transformation_matrixes=False)
        return _mp_vision.FaceLandmarker.create_from_options(opts)

    def init(self):
        self.landmarker = self._build_landmarker()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): raise RuntimeError("Cannot open camera for scar mode.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.running = True

        if _HAS_TORCH:
            try:
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                m = _UNet().to(dev)
                m.load_state_dict(torch.load("unet_scar.pth", map_location=dev, weights_only=False))
                m.eval()
                self.model = m
                self.device = dev
                self.has_weights = True
                print("[Scar] U-Net loaded")
            except Exception as e:
                print(f"[Scar] No U-Net weights ({e}) — using HSV+LAB")
        else:
            print("[Scar] torch not available — using HSV+LAB")
        print("[Scar] Camera ready")

    def _face_zones(self, frame):
        h, w = frame.shape[:2]
        face_mask = np.zeros((h,w), np.uint8)
        ex_mask   = np.zeros((h,w), np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.timestamp += 33
        result = self.landmarker.detect_for_video(mp_img, self.timestamp)
        if not result.face_landmarks: return face_mask, ex_mask
        lms = result.face_landmarks[0]
        all_pts = np.array([[int(lm.x*w), int(lm.y*h)] for lm in lms], np.int32)
        cv2.fillConvexPoly(face_mask, cv2.convexHull(all_pts), 255)
        for idx, exp in [(_LIPS_IDX,14),(_LEFT_EYE,10),(_RIGHT_EYE,10),
                          (_LEFT_BROW,6),(_RIGHT_BROW,6),(_NOSE_IDX,6)]:
            _draw_excl(ex_mask, lms, idx, h, w, exp)
        return face_mask, ex_mask

    def grab(self):
        ret, frame = self.cap.read()
        if not ret: return None, None
        frame = cv2.flip(frame, 1)

        face_mask, ex_mask = self._face_zones(frame)
        self.calibrator.update(frame, face_mask, ex_mask)

        if self.has_weights:
            raw_mask = _unet_scar_mask(self.model, frame, self.device)
            if cv2.countNonZero(ex_mask) > 0:
                raw_mask = cv2.bitwise_and(raw_mask, cv2.bitwise_not(ex_mask))
        else:
            raw_mask = _detect_scar_mask(frame, self.calibrator, face_mask, ex_mask)

        k = np.ones((5,5), np.uint8)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, k)
        raw_mask = cv2.dilate(raw_mask, k, iterations=1)

        raw_f = raw_mask.astype(np.float32) / 255.0
        a = self.params["alpha"]
        self.smoothed = raw_f if self.smoothed is None else a*raw_f + (1-a)*self.smoothed
        smooth_bin = (self.smoothed > self.params["threshold"]).astype(np.uint8) * 255

        self.tracker.update(smooth_bin)
        stable = self.tracker.get_stable()

        if self.calibrator.frame_count % 30 == 0:
            nz = cv2.countNonZero(stable) if stable is not None else 0
            print(f"[Scar] frame={self.calibrator.frame_count} "
                  f"a={self.calibrator.skin_a_mean:.1f} l={self.calibrator.skin_l_mean:.1f} "
                  f"mask_px={nz}")

        if self.params.get("show_mask", False):
            overlay = frame.copy()
            if stable is not None:
                red = np.full_like(frame, (0,0,255), dtype=np.uint8)
                overlay = cv2.addWeighted(cv2.bitwise_and(red, red, mask=stable), 0.6, frame, 0.4, 0)
                cnts,_ = cv2.findContours(stable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1, (0,0,255), 2)
            result = overlay
        elif stable is not None and cv2.countNonZero(stable) > 0:
            result = cv2.inpaint(frame, stable, inpaintRadius=self.params["inpaint_r"], flags=cv2.INPAINT_TELEA)
        else:
            result = frame.copy()

        def enc(img):
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            return base64.b64encode(buf.tobytes()).decode("ascii")
        return enc(frame), enc(result)

    def close(self):
        self.running = False
        if self.cap: self.cap.release()
        if self.landmarker: self.landmarker.close()


SCAR = ScarState()


@app.get("/scar")
async def scar_page():
    p = os.path.join(os.path.dirname(__file__), "scar.html")
    with open(p, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws/scar")
async def ws_scar(ws: WebSocket):
    await ws.accept()
    if not SCAR.running:
        try:
            loop2 = asyncio.get_event_loop()
            await loop2.run_in_executor(None, SCAR.init)
        except Exception as e:
            await ws.send_text(json.dumps({"error": str(e)}))
            await ws.close()
            return

    loop2 = asyncio.get_event_loop()
    try:
        while True:
            try:
                orig_b64, result_b64 = await loop2.run_in_executor(None, SCAR.grab)
            except Exception as e:
                print(f"[Scar] Frame grab error: {e}")
                await asyncio.sleep(0.1)
                continue
            if orig_b64 is None:
                await asyncio.sleep(0.02)
                continue
            await ws.send_text(json.dumps({
                "original": orig_b64,
                "result":   result_b64,
            }))
            await asyncio.sleep(0.001)
    except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError):
        print("[Scar] Client disconnected")
    except asyncio.CancelledError:
        print("[Scar] Server shutting down")
    except Exception as e:
        print(f"[Scar WS] error: {e}")


@app.websocket("/ws/scar/controls")
async def ws_scar_ctrl(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = json.loads(await ws.receive_text())
            for k in ("alpha","threshold","inpaint_r"):
                if k in data:
                    SCAR.params[k] = float(data[k])
            if "show_mask" in data:
                SCAR.params["show_mask"] = bool(data["show_mask"])
            await ws.send_text(json.dumps({"ok": True}))
    except WebSocketDisconnect:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# BROW SHAPER MODE
# ═════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.dirname(__file__))

from modes.brow_shaper import BrowShaperMode as _BrowShaperMode


class BrowState:
    def __init__(self):
        self.cap        = None
        self.landmarker = None
        self.running    = False
        self._t0        = time.perf_counter()
        self.timestamp  = 0
        self.brow       = _BrowShaperMode()

    def _build_landmarker(self):
        base = _mp_python.BaseOptions(
            model_asset_path="models/face_landmarker.task")
        opts = _mp_vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return _mp_vision.FaceLandmarker.create_from_options(opts)

    def init(self):
        self.landmarker = self._build_landmarker()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera for brow mode.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap     = cap
        self.running = True
        print("[Brow] Camera ready")

    def grab(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.timestamp += 33
        result = self.landmarker.detect_for_video(mp_img, self.timestamp)
        lms = result.face_landmarks[0] if result.face_landmarks else None
        if lms is not None:
            frame = self.brow.process(frame, lms, w, h)

        def enc(img):
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            return base64.b64encode(buf.tobytes()).decode("ascii")

        return enc(frame), enc(frame)

    def close(self):
        self.running = False
        if self.cap: self.cap.release()
        if self.landmarker: self.landmarker.close()


BROW = BrowState()


@app.get("/brow")
async def brow_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-Look Brow Shaper</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:sans-serif;
         display:flex; flex-direction:column; align-items:center; height:100vh; }
  h2 { margin:12px 0 6px; font-weight:300; letter-spacing:2px; }
  canvas { display:block; max-width:95vw; max-height:80vh; border-radius:8px;
           box-shadow:0 0 30px rgba(0,200,150,0.15); }
  .status { margin:8px 0; font-size:13px; opacity:0.6; }
</style>
</head>
<body>
<h2>&#x2728; Brow Shaper</h2>
<div class="status" id="status">Connecting...</div>
<canvas id="c"></canvas>
<script>
const c = document.getElementById('c'), ctx = c.getContext('2d');
let ws = null;
function connect() {
  ws = new WebSocket('ws://' + location.host + '/ws/brow');
  ws.onmessage = e => {
    const d = JSON.parse(e.data);
    const img = new Image();
    img.onload = () => { c.width = img.width; c.height = img.height; ctx.drawImage(img,0,0); };
    img.src = 'data:image/jpeg;base64,' + (d.result || d.original);
    document.getElementById('status').textContent = 'Live';
  };
  ws.onclose = () => { document.getElementById('status').textContent = 'Disconnected. Reconnecting...';
                        setTimeout(connect, 1000); };
  ws.onerror = () => ws.close();
}
connect();
</script>
</body>
</html>
""")


@app.websocket("/ws/brow")
async def ws_brow(ws: WebSocket):
    await ws.accept()
    if not BROW.running:
        try:
            loop2 = asyncio.get_event_loop()
            await loop2.run_in_executor(None, BROW.init)
        except Exception as e:
            await ws.send_text(json.dumps({"error": str(e)}))
            await ws.close()
            return

    loop2 = asyncio.get_event_loop()
    try:
        while True:
            try:
                _, result_b64 = await loop2.run_in_executor(None, BROW.grab)
            except Exception as e:
                print(f"[Brow] Frame grab error: {e}")
                await asyncio.sleep(0.1)
                continue
            if result_b64 is None:
                await asyncio.sleep(0.02)
                continue
            await ws.send_text(json.dumps({"result": result_b64}))
            await asyncio.sleep(0.001)
    except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError):
        print("[Brow] Client disconnected")
    except asyncio.CancelledError:
        print("[Brow] Server shutting down")
    except Exception as e:
        print(f"[Brow WS] error: {e}")
        traceback.print_exc()