"""
server.py — V-Look Web Server
Run: python -m uvicorn server:app --host 0.0.0.0 --port 8000

Requirements:
    pip install fastapi uvicorn websockets opencv-python mediapipe numpy moderngl
"""

import asyncio
import base64
import concurrent.futures
import faulthandler
import json
import threading
import time
import traceback
faulthandler.enable()

# Dedicated single-thread executor for all OpenGL/camera work
_gl_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gl")

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
from firestore_watcher import FirestoreWatcher

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
        self.ready     = threading.Event()

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

    def close(self):
        self.running = False
        self.ready.clear()
        if self.processor:
            try:
                self.processor.close()
            except Exception:
                pass
            self.processor = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


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
    if STATE.processor is None:
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
    STATE.ready.set()
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
    if STATE.processor is not None:
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

        try:
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
        except Exception as e:
            print(f"[Render] Error: {e}")
            traceback.print_exc()
            face_found = False

    dbg_lines = [
        f"X:{p['offset_x']:+.0f}  Y:{p['offset_y']:+.0f}  Z:{p['offset_z']:+.0f}  S:{p['mesh_scale']:.0f}" if face_found else "NO FACE",
    ]
    for i, line in enumerate(dbg_lines):
        cv2.putText(frame, line, (8, STATE.H - 12 - i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 210, 170), 1, cv2.LINE_AA)

    small = cv2.resize(frame, (320, 240))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
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
def close_camera():
    STATE.running = False
    if STATE.cap:
        try:
            STATE.cap.release()
        except Exception:
            pass


@app.websocket("/ws/video")
async def ws_video(ws: WebSocket):
    await ws.accept()
    for _ in range(50):
        if STATE.running:
            break
        await asyncio.sleep(0.1)
    if not STATE.running:
        await ws.send_text(json.dumps({"error": "Camera not ready"}))
        await ws.close()
        return

    loop = asyncio.get_event_loop()
    try:
        while True:
            try:
                frame_bytes, face_found = await loop.run_in_executor(_gl_executor, grab_processed_frame)
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
        close_camera()
        print("[V-Look Web] Client disconnected")
    except asyncio.CancelledError:
        print("[V-Look Web] Server shutting down")
    except Exception as e:
        print(f"[V-Look Web] WS error: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — controls
# ─────────────────────────────────────────────────────────────────────────────
_ctrl_conn_count = 0
_ctrl_conn_lock = threading.Lock()

@app.websocket("/ws/controls")
async def ws_controls(ws: WebSocket):
    global _ctrl_conn_count
    with _ctrl_conn_lock:
        if _ctrl_conn_count >= 3:
            await ws.close(1013, "Too many control connections")
            return
        _ctrl_conn_count += 1
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
    finally:
        with _ctrl_conn_lock:
            _ctrl_conn_count -= 1


# ─────────────────────────────────────────────────────────────────────────────
# Firestore mode control
# ─────────────────────────────────────────────────────────────────────────────
SERVICE_ACCOUNT = os.path.join(os.path.dirname(__file__),
                               "medar-3214d-firebase-adminsdk-fbsvc-679e0072ab.json")
FW = FirestoreWatcher(SERVICE_ACCOUNT, "jetson_control", "active_filter", "option")
CURRENT_MODE = "hair"
_mode_lock = threading.Lock()


def _close_all_modes():
    STATE.close()
    SCAR.close()
    BROW.close()
    NOSE.close()
    MOUTH.close()
    # Let the camera settle
    time.sleep(0.3)


def _switch_mode(raw_mode):
    global CURRENT_MODE
    # Normalize aliases to canonical mode names
    _ALIASES = {"style_1": "hair", "brow_1": "brow", "brow_2": "brow", "mouth_full": "mouth"}
    mode = _ALIASES.get(raw_mode, raw_mode)
    with _mode_lock:
        if mode == CURRENT_MODE:
            return
        print(f"[Mode] Switching: {CURRENT_MODE} -> {raw_mode} (canonical: {mode})")
        _close_all_modes()
        if mode == "hair":
            init_camera()
            mode = "hair"
        elif mode == "nose":
            NOSE.init()
            mode = "nose"
        elif mode == "scar":
            SCAR.init()
            mode = "scar"
        elif mode == "brow":
            BROW.init()
            mode = "brow"
        elif mode == "mouth":
            MOUTH.init()
            mode = "mouth"
        else:
            print(f"[Mode] Unknown mode {mode!r}, falling back to hair")
            init_camera()
            mode = "hair"
        CURRENT_MODE = mode


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()

    def _on_firestore_change(val):
        mode = (val or "hair").lower()
        if mode != CURRENT_MODE:
            asyncio.run_coroutine_threadsafe(
                _switch_mode_async(mode), loop)

    async def _switch_mode_async(mode):
        await loop.run_in_executor(_gl_executor, _switch_mode, mode)

    FW.on_change(_on_firestore_change)
    FW.start()
    await loop.run_in_executor(_gl_executor, init_camera)


@app.on_event("shutdown")
async def shutdown():
    FW.stop()
    STATE.close()
    SCAR.close()
    BROW.close()
    NOSE.close()
    MOUTH.close()
    print("[V-Look Web] Shutdown complete.")


@app.get("/mode")
async def get_mode():
    return {"mode": CURRENT_MODE, "desired": FW.get() or CURRENT_MODE}


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
        if not hasattr(self, 'landmarker') or self.landmarker is None:
            self.landmarker = self._build_landmarker()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): raise RuntimeError("Cannot open camera for scar mode.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.running = True
        STATE.ready.set()

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
            s = cv2.resize(img, (320, 240))
            _, buf = cv2.imencode(".jpg", s, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return base64.b64encode(buf.tobytes()).decode("ascii")
        return enc(frame), enc(result)

    def close(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


SCAR = ScarState()


@app.get("/scar")
async def scar_page():
    p = os.path.join(os.path.dirname(__file__), "scar.html")
    with open(p, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws/scar")
async def ws_scar(ws: WebSocket):
    await ws.accept()
    for _ in range(50):
        if SCAR.running:
            break
        await asyncio.sleep(0.1)
    if not SCAR.running:
        await ws.send_text(json.dumps({"error": "Scar filter not ready"}))
        await ws.close()
        return

    loop2 = asyncio.get_event_loop()
    try:
        while True:
            try:
                orig_b64, result_b64 = await loop2.run_in_executor(_gl_executor, SCAR.grab)
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
        SCAR.close()
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

from modes.nose_filter import NoseRenderer as _NoseRenderer

from modes.mouth_correction import MouthCorrection as _MouthCorrection

from core.geometry import GeometryEngine as _GeometryEngine
from core.processor import FaceProcessor as _FaceProcessor


class BrowState:
    def __init__(self):
        self.cap        = None
        self.running    = False
        self._t0        = time.perf_counter()
        self.brow       = _BrowShaperMode()

    def init(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera for brow mode.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap     = cap
        self.running = True
        STATE.ready.set()
        self.brow._init_models()
        self.brow.start_speech()
        print("[Brow] Camera + Speech ready")

    def grab(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, "", ""
        frame = cv2.flip(frame, 1)
        frame, gesture, speech_text, speech_status = self.brow.process(frame)

        def enc(img):
            s = cv2.resize(img, (320, 240))
            _, buf = cv2.imencode(".jpg", s, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return base64.b64encode(buf.tobytes()).decode("ascii")

        return enc(frame), gesture, speech_text, speech_status

    def close(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


BROW = BrowState()

# ─────────────────────────────────────────────────────────────────────────────
# MOUTH CORRECTION MODE
# ─────────────────────────────────────────────────────────────────────────────

class MouthState:
    def __init__(self):
        self.cap        = None
        self.running    = False
        self._t0        = time.perf_counter()
        self.mouth      = _MouthCorrection()

    def init(self):
        self.mouth.reset_smooth()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera for mouth mode.")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap     = cap
        self.running = True
        STATE.ready.set()
        print("[Mouth] Camera ready")

    def grab(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
        frame = cv2.flip(frame, 1)
        result, face_found = self.mouth.process(frame)
        small = cv2.resize(result, (320, 240))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return b64, b64, face_found

    def close(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None


MOUTH = MouthState()

class NoseState:
    def __init__(self):
        self.cap        = None
        self.processor  = None
        self.renderer   = None
        self.geo        = None
        self.running    = False
        self._t0        = time.perf_counter()
        self.timestamp  = 0
        self.W = self.H = 0
        self.K          = None
        self.lock       = threading.Lock()
        self._frame_count = 0
        self.face_data  = {"found": False, "R_mat": None, "tvec": None, "landmarks": None}
        self.params = {
            "offset_x":  -15.0,
            "offset_y":  -110.0,
            "offset_z":  -100.0,
            "mesh_scale": 135.0,
            "color_r":   255,
            "color_g":   219,
            "color_b":   172,
        }

    def _on_result(self, result, image, ts_ms):
        if not result.face_landmarks:
            with self.lock:
                self.face_data["found"] = False
            return
        lms = result.face_landmarks[0]
        W, H = self.W, self.H

        # -- Use nose landmarks directly for yaw/pitch/roll --
        tip  = np.array([lms[1].x * W,   lms[1].y * H])
        mid  = np.array([lms[168].x * W, lms[168].y * H])
        leye = np.array([lms[33].x * W,  lms[33].y * H])
        reye = np.array([lms[263].x * W, lms[263].y * H])

        eye_dist = np.linalg.norm(reye - leye) + 1e-6
        nd = tip - mid                        # nose direction in image (px)

        # Yaw: horizontal offset of tip relative to bridge
        #   Mirrored frame: user turns left → nd.x < 0
        #   → yaw < 0  (nose rotates left in the rendered view)
        yaw = np.arctan2(nd[0], eye_dist * 2.5)

        # Pitch: vertical deviation from rest nose length
        rest_len = 0.08 * W
        pitch = np.arctan2(rest_len - nd[1], eye_dist * 2.5)

        # Roll: eye-line angle
        roll = np.arctan2(reye[1] - leye[1], reye[0] - leye[0])

        # Build rotation: Ry(yaw) @ Rx(pitch) @ Rz(roll) @ R_base
        # R_base = 180° around X — matches solvePnP model→camera convention
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        R = np.array([
            [cy*cr + sy*sp*sr,  -cy*sr + sy*sp*cr,  sy*cp],
            [cp*sr,              cp*cr,             -sp   ],
            [-sy*cr + cy*sp*sr,  sy*sr + cy*sp*cr,  cy*cp],
        ], dtype=np.float32)

        # Base rotation: model +z → camera −z (nose toward camera)
        R_base = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        R = R @ R_base

        # -- Translation: solvePnP for tvec (full solve, then discard its R) --
        dist = np.zeros((4, 1), np.float32)
        pts2d = np.array([[lms[i].x * W, lms[i].y * H]
                          for i in self.geo.SOLVE_IDX], dtype=np.float32)
        with self.lock:
            prev_r = cv2.Rodrigues(self.face_data["R_mat"])[0] if self.face_data["R_mat"] is not None else None
            prev_t = self.face_data["tvec"]
        ok, _, tvec = self.geo.solve_pose(
            pts2d, self.K, dist, prev_rvec=prev_r, prev_tvec=prev_t)
        if not ok:
            return

        with self.lock:
            self.face_data["R_mat"] = self.geo.smooth_rotation_matrix(
                self.face_data["R_mat"], R, alpha=0.25)
            prev_t = self.face_data["tvec"]
            self.face_data["tvec"] = tvec if prev_t is None else (0.3 * prev_t + 0.7 * tvec)
            self.face_data["landmarks"] = lms
            self.face_data["found"] = True

    def init(self):
        self.geo = _GeometryEngine()
        self.renderer = _NoseRenderer("assets/nose.obj", normalize_span=1.5, baseline=(10, 50, -950))
        if not hasattr(self, 'processor') or self.processor is None:
            self.processor = _FaceProcessor("models/face_landmarker.task", self._on_result)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera for nose mode.")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap     = cap
        self.W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.K       = self.geo.get_camera_matrix(self.W, self.H)
        self._t0     = time.perf_counter()
        self.running = True
        STATE.ready.set()
        print(f"[Nose] Camera ready {self.W}x{self.H}")

    def grab(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
        frame = cv2.flip(frame, 1)
        self.timestamp += 33

        self._frame_count += 1
        small = cv2.resize(frame, (320, 240))
        if self._frame_count % 2 == 0:
            self.processor.process_frame(small, self.timestamp)

        face_found = False

        with self.lock:
            fd = self.face_data
            if fd["found"] and fd["R_mat"] is not None:
                face_found = True
                rvec_f, _ = cv2.Rodrigues(fd["R_mat"])
                tvec_f = fd["tvec"]
                p = self.params.copy()

        if face_found:
            color = (p["color_b"], p["color_g"], p["color_r"])
            frame = self.renderer.render(
                frame, rvec_f, tvec_f, self.K,
                base_color  = color,
                offset_x    = p["offset_x"],
                offset_y    = p["offset_y"],
                offset_z    = p["offset_z"],
                mesh_scale  = p["mesh_scale"],
            )

        def enc(img):
            s = cv2.resize(img, (320, 240))
            _, buf = cv2.imencode(".jpg", s, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return base64.b64encode(buf.tobytes()).decode("ascii")

        return enc(frame), enc(frame), face_found

    def close(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


NOSE = NoseState()


@app.get("/nose")
async def nose_page():
    import os
    base = os.path.dirname(__file__)
    path = os.path.join(base, "nose.html")
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws/nose")
async def ws_nose(ws: WebSocket):
    await ws.accept()
    for _ in range(50):
        if NOSE.running:
            break
        await asyncio.sleep(0.1)
    if not NOSE.running:
        await ws.send_text(json.dumps({"error": "Nose filter not ready"}))
        await ws.close()
        return

    loop2 = asyncio.get_event_loop()
    try:
        while True:
            try:
                _, result_b64, face_found = await loop2.run_in_executor(_gl_executor, NOSE.grab)
            except Exception as e:
                print(f"[Nose] Frame grab error: {e}")
                await asyncio.sleep(0.1)
                continue
            if result_b64 is None:
                await asyncio.sleep(0.02)
                continue
            await ws.send_text(json.dumps({"result": result_b64, "face_found": face_found}))
            await asyncio.sleep(0.001)
    except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError):
        NOSE.close()
        print("[Nose] Client disconnected")
    except asyncio.CancelledError:
        print("[Nose] Server shutting down")
    except Exception as e:
        print(f"[Nose WS] error: {e}")
        traceback.print_exc()


@app.websocket("/ws/nose/controls")
async def ws_nose_ctrl(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = json.loads(await ws.receive_text())
            with NOSE.lock:
                for key in ("offset_x", "offset_y", "offset_z", "mesh_scale"):
                    if key in data:
                        NOSE.params[key] = float(data[key])
                for key in ("color_r", "color_g", "color_b"):
                    if key in data:
                        NOSE.params[key] = int(data[key])
            await ws.send_text(json.dumps({"ok": True}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[Nose Ctrl WS] error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MOUTH CORRECTION MODE — routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/mouth")
async def mouth_page():
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>V-Look &mdash; Mouth Correction</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#0e0f14;--panel:#16181f;--panel2:#1c1f2b;--border:#32374a;--accent:#e0a050;--align:#1e9fff;--warn:#dc5050;--orange:#ff8c1a;--pink:#c850d4;--green:#28c868;--cyan:#14d4e0;--red:#e03232;--text-hi:#eaeaf0;--text-lo:#646878;--text-mid:#9096ac;--radius:10px;--mono:'Space Mono',monospace;--ui:'Rajdhani',sans-serif}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text-hi);font-family:var(--ui);min-height:100vh;display:flex;flex-direction:column;overflow-x:hidden}
.layout{display:grid;grid-template-columns:270px 1fr 220px;grid-template-rows:52px 1fr 44px;min-height:100vh}
.topbar{grid-column:1/-1;background:var(--panel);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 18px;gap:14px}
.logo{font-family:var(--mono);font-size:.9rem;font-weight:700;color:var(--accent);letter-spacing:.15em}
.logo span{color:var(--text-lo);font-weight:400}
.sep{width:1px;height:22px;background:var(--border)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--warn);box-shadow:0 0 6px var(--warn);transition:.3s}
.dot.on{background:var(--accent);box-shadow:0 0 8px var(--accent)}
.st{font-family:var(--mono);font-size:.58rem;color:var(--warn);letter-spacing:.1em;transition:.3s}
.st.on{color:var(--accent)}
.ml{margin-left:auto;display:flex;gap:10px;align-items:center}
.badge{font-family:var(--mono);font-size:.58rem;color:var(--text-lo);background:var(--panel2);border:1px solid var(--border);border-radius:4px;padding:2px 8px}
.pl{background:var(--panel);border-right:1px solid var(--border);padding:14px 12px;display:flex;flex-direction:column;gap:16px;overflow-y:auto}
.pr{background:var(--panel);border-left:1px solid var(--border);padding:14px 12px;display:flex;flex-direction:column;gap:14px;overflow-y:auto}
.va{background:#08090c;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden}
#vc{max-width:100%;max-height:100%;display:block}
.brackets{position:absolute;inset:0;pointer-events:none}
.brackets svg{width:100%;height:100%}
.br{stroke:var(--accent);stroke-width:2;fill:none;opacity:.5;transition:.3s}
.br.warn{stroke:var(--warn);animation:blink 1s step-start infinite}
@keyframes blink{50%{opacity:0}}
.scanline{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:.18;pointer-events:none;animation:scan 5s linear infinite}
@keyframes scan{0%{top:0}100%{top:100%}}
#overlay{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:rgba(8,9,12,.92);gap:10px;z-index:10}
#overlay .ot{font-family:var(--mono);font-size:.7rem;color:var(--accent);letter-spacing:.15em}
#overlay .os{font-family:var(--mono);font-size:.6rem;color:var(--text-lo)}
.spin{width:26px;height:26px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.bot{grid-column:1/-1;background:var(--panel);border-top:1px solid var(--border);display:flex;align-items:center;padding:0 18px;gap:18px}
.hi{display:flex;align-items:center;gap:6px;font-size:.68rem;color:var(--text-lo);font-family:var(--mono)}
.hk{background:var(--panel2);border:1px solid var(--border);border-radius:3px;padding:1px 5px;font-size:.58rem;color:var(--accent)}
.sec{font-family:var(--mono);font-size:.56rem;color:var(--text-lo);letter-spacing:.15em;text-transform:uppercase;margin-bottom:7px;display:flex;align-items:center;gap:8px}
.sec::after{content:'';flex:1;height:1px;background:var(--border)}
.sg{display:flex;flex-direction:column;gap:11px}
.sr{display:flex;flex-direction:column;gap:4px}
.sh{display:flex;justify-content:space-between;align-items:baseline}
.sl{font-family:var(--mono);font-size:.56rem;color:var(--text-lo);letter-spacing:.1em;text-transform:uppercase}
.sv{font-family:var(--mono);font-size:.6rem;color:var(--text-hi);min-width:68px;text-align:right}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:4px;border-radius:2px;background:var(--border);outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;border-radius:50%;background:var(--accent);box-shadow:0 0 6px var(--accent);transition:.1s}
input[type=range]::-webkit-slider-thumb:active{transform:scale(1.4)}
.btn{width:100%;padding:8px;background:transparent;border:1px solid var(--border);border-radius:var(--radius);color:var(--text-lo);font-family:var(--mono);font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;cursor:pointer;transition:.2s}
.btn:hover{border-color:var(--align);color:var(--align);background:rgba(30,159,255,.06)}
.btn-snap{width:100%;padding:10px;background:linear-gradient(135deg,rgba(224,160,80,.15),rgba(224,160,80,.05));border:1px solid var(--accent);border-radius:var(--radius);color:var(--accent);font-family:var(--mono);font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;cursor:pointer;transition:.2s}
.btn-snap:hover{box-shadow:0 0 16px rgba(224,160,80,.25)}
.btn-snap:active{transform:scale(.98)}
.ib{background:var(--panel2);border:1px solid var(--border);border-radius:var(--radius);padding:10px 12px;font-family:var(--mono);font-size:.54rem;color:var(--text-lo);line-height:1.85;letter-spacing:.04em}
.ib b{color:var(--text-mid)}
.lp{background:rgba(224,160,80,.06);border:1px solid rgba(224,160,80,.25);border-radius:var(--radius);padding:8px 10px;font-family:var(--mono);font-size:.54rem;color:var(--accent);line-height:1.9}
.err{background:rgba(220,80,80,.12);border:1px solid var(--warn);border-radius:var(--radius);padding:8px 12px;font-family:var(--mono);font-size:.57rem;color:var(--warn);display:none}
</style>
</head>
<body>
<div class="layout">
  <header class="topbar">
    <div class="logo">V-Look <span>Mouth</span></div>
    <div class="sep"></div>
    <a href="/" style="font-family:var(--mono);font-size:.58rem;color:var(--text-lo);background:var(--panel2);border:1px solid var(--border);border-radius:6px;padding:5px 12px;cursor:pointer;letter-spacing:.08em;text-decoration:none;transition:.2s">Hair AR</a>
    <a href="/nose" style="font-family:var(--mono);font-size:.58rem;color:var(--text-lo);background:var(--panel2);border:1px solid var(--border);border-radius:6px;padding:5px 12px;cursor:pointer;letter-spacing:.08em;text-decoration:none;transition:.2s">Nose Filter</a>
    <a href="/scar" style="font-family:var(--mono);font-size:.58rem;color:var(--text-lo);background:var(--panel2);border:1px solid var(--border);border-radius:6px;padding:5px 12px;cursor:pointer;letter-spacing:.08em;text-decoration:none;transition:.2s">Scar Removal</a>
    <a href="/brow" style="font-family:var(--mono);font-size:.58rem;color:var(--text-lo);background:var(--panel2);border:1px solid var(--border);border-radius:6px;padding:5px 12px;cursor:pointer;letter-spacing:.08em;text-decoration:none;transition:.2s">Brow Shaper</a>
    <a href="/mouth" style="font-family:var(--mono);font-size:.58rem;color:var(--accent);background:rgba(224,160,80,.08);border:1px solid var(--accent);border-radius:6px;padding:5px 12px;cursor:pointer;letter-spacing:.08em;text-decoration:none;transition:.2s">Mouth</a>
    <div class="sep"></div>
    <div class="dot" id="dot"></div>
    <span class="st" id="st">SCANNING...</span>
    <div class="ml">
      <div class="badge"><span id="fps" style="color:var(--green)">0.0</span> FPS</div>
      <div class="badge" id="wsb" style="color:var(--warn)">WS OFF</div>
    </div>
  </header>

  <aside class="pl">
    <div>
      <div class="sec">Position</div>
      <div class="sg">
        <div class="sr">
          <div class="sh"><span class="sl">Offset X</span><span class="sv" id="vox">+0</span></div>
          <input type="range" id="ox" class="pink" min="-200" max="200" value="0" step="5">
        </div>
        <div class="sr">
          <div class="sh"><span class="sl">Offset Y</span><span class="sv" id="voy">+0</span></div>
          <input type="range" id="oy" class="orange" min="-200" max="200" value="0" step="5">
        </div>
        <div class="sr">
          <div class="sh"><span class="sl">Scale</span><span class="sv" id="vsc">100%</span></div>
          <input type="range" id="sc" class="green" min="30" max="300" value="100" step="1">
        </div>
      </div>
    </div>
    <button class="btn" onclick="resetSliders()">&hookleftarrow; Reset</button>
    <div class="lp" id="lp">X: +0 &nbsp; Y: +0 &nbsp; S: 100%</div>
    <div class="err" id="err"></div>
  </aside>

  <main class="va">
    <div id="overlay">
      <div class="spin"></div>
      <div class="ot">Connecting to mouth filter server&hellip;</div>
      <div class="os">python -m uvicorn server:app --port 8000</div>
    </div>
    <canvas id="vc" width="640" height="480"></canvas>
    <div class="brackets">
      <svg viewBox="0 0 640 480" xmlns="http://www.w3.org/2000/svg">
        <polyline class="br" id="bTL" points="80,110 58,110 58,88"/>
        <polyline class="br" id="bTR" points="560,88 582,88 582,110"/>
        <polyline class="br" id="bBL" points="58,370 58,392 80,392"/>
        <polyline class="br" id="bBR" points="582,370 582,392 560,392"/>
      </svg>
    </div>
    <div class="scanline"></div>
  </main>

  <aside class="pr">
    <div>
      <div class="sec">Capture</div>
      <button class="btn-snap" onclick="snap()">&#9679; Screenshot</button>
    </div>
    <div>
      <div class="sec">Controls</div>
      <div class="ib">
        <b>W/S</b> up / down<br>
        <b>A/D</b> left / right<br>
        <b>-/=</b> shrink / grow<br>
        <b>R</b> reset position
      </div>
    </div>
    <div>
      <div class="sec">Tip</div>
      <div class="ib">
        Mouth too high:<br>
        &rarr; <b>W</b> or drag Y more negative<br><br>
        Mouth too small:<br>
        &rarr; <b>=</b> or drag Scale up<br><br>
        Adjust with sliders<br>
        or keyboard keys
      </div>
    </div>
  </aside>

  <footer class="bot">
    <div class="hi"><span class="hk">W</span><span class="hk">S</span> up/down</div>
    <div class="hi"><span class="hk">A</span><span class="hk">D</span> left/right</div>
    <div class="hi"><span class="hk">-</span><span class="hk">=</span> size</div>
    <div class="hi"><span class="hk">R</span> reset</div>
    <div class="hi"><span class="hk">S</span> snapshot</div>
  </footer>
</div>
<script>
const HOST=location.host;let vidWs=null,ctrlWs=null;
const DEFAULTS={ox:0,oy:0,sc:100};
const canvas=document.getElementById('vc'),ctx=canvas.getContext('2d'),img=new Image();
let fpsBuf=[],fpsTs=performance.now();
let offX=0,offY=0,scalePct=100;
const MOVE_STEP=3;

function connectCtl(){
  ctrlWs=new WebSocket('ws://'+HOST+'/ws/mouth/controls');
  ctrlWs.onclose=()=>setTimeout(connectCtl,5000);
}
function sendCtl(){
  if(ctrlWs&&ctrlWs.readyState===WebSocket.OPEN)
    ctrlWs.send(JSON.stringify({offset_x:offX,offset_y:offY,scale_delta:(scalePct-100)/100}));
}
function updateLive(){
  document.getElementById('vox').textContent=(offX>=0?'+':'')+offX;
  document.getElementById('voy').textContent=(offY>=0?'+':'')+offY;
  document.getElementById('vsc').textContent=scalePct+'%';
  document.getElementById('lp').innerHTML='X: '+(offX>=0?'+':'')+offX+' &nbsp; Y: '+(offY>=0?'+':'')+offY+' &nbsp; S: '+scalePct+'%';
}
function sliderToVals(){
  offX=+document.getElementById('ox').value;
  offY=+document.getElementById('oy').value;
  scalePct=+document.getElementById('sc').value;
  updateLive();sendCtl();
}
function resetSliders(){
  offX=DEFAULTS.ox;offY=DEFAULTS.oy;scalePct=DEFAULTS.sc;
  document.getElementById('ox').value=offX;
  document.getElementById('oy').value=offY;
  document.getElementById('sc').value=scalePct;
  updateLive();sendCtl();
}
['ox','oy','sc'].forEach(id=>document.getElementById(id).addEventListener('input',sliderToVals));
function connectVid(){
  vidWs=new WebSocket('ws://'+HOST+'/ws/mouth');
  vidWs.onopen=()=>{
    document.getElementById('wsb').textContent='WS ON';
    document.getElementById('wsb').style.color='var(--green)';
    document.getElementById('overlay').style.display='none';
  };
  vidWs.onmessage=ev=>{
    const d=JSON.parse(ev.data);
    if(d.error){showErr(d.error);return}
    img.onload=()=>ctx.drawImage(img,0,0,canvas.width,canvas.height);
    const fd=d.result||d.frame;
    if(fd)img.src='data:image/jpeg;base64,'+fd;
    const on=d.face_found;
    if(on!==undefined){
      document.getElementById('dot').className='dot'+(on?' on':'');
      document.getElementById('st').className='st'+(on?' on':'');
      document.getElementById('st').textContent=on?'FACE LOCKED':'SCANNING...';
      ['bTL','bTR','bBL','bBR'].forEach(id=>{
        document.getElementById(id).className='br'+(on?'':' warn');
      });
    }
    const now=performance.now();
    fpsBuf.push(1000/Math.max(now-fpsTs,1));fpsTs=now;
    if(fpsBuf.length>45)fpsBuf.shift();
    const f=fpsBuf.reduce((a,b)=>a+b,0)/fpsBuf.length;
    const fe=document.getElementById('fps');
    fe.textContent=f.toFixed(1);
    fe.style.color=f>=28?'var(--green)':f>=18?'var(--cyan)':'var(--red)';
  };
  vidWs.onerror=()=>showErr('Cannot reach server. Is it running on port 8000?');
  vidWs.onclose=()=>{
    document.getElementById('wsb').textContent='WS OFF';
    document.getElementById('wsb').style.color='#dc5050';
    document.getElementById('overlay').style.display='flex';
    setTimeout(connectVid,2500);
  };
}
function showErr(msg){
  document.getElementById('err').textContent='\\u26a0 '+msg;
  document.getElementById('err').style.display='block';
}
function snap(){
  const a=document.createElement('a');
  a.download='mouth_'+Date.now()+'.png';
  a.href=canvas.toDataURL('image/png');a.click();
}
document.addEventListener('keydown',e=>{
  const k=e.key;
  if(k==='w'||k==='W'){offY-=MOVE_STEP;document.getElementById('oy').value=offY;updateLive();sendCtl()}
  else if(k==='s'||k==='S'){offY+=MOVE_STEP;document.getElementById('oy').value=offY;updateLive();sendCtl()}
  else if(k==='a'||k==='A'){offX-=MOVE_STEP;document.getElementById('ox').value=offX;updateLive();sendCtl()}
  else if(k==='d'||k==='D'){offX+=MOVE_STEP;document.getElementById('ox').value=offX;updateLive();sendCtl()}
  else if(k==='-'){scalePct=Math.max(30,scalePct-5);document.getElementById('sc').value=scalePct;updateLive();sendCtl()}
  else if(k==='='){scalePct=Math.min(300,scalePct+5);document.getElementById('sc').value=scalePct;updateLive();sendCtl()}
  else if(k==='r'||k==='R'){resetSliders()}
  else if(k.toLowerCase()==='s')snap();
});
connectVid();connectCtl();
setInterval(async()=>{
  try{const r=await fetch('/mode');const d=await r.json();if(d.mode!=='mouth')window.location.href='/'}catch{}
},1000);
</script>
</body>
</html>""")


@app.websocket("/ws/mouth")
async def ws_mouth(ws: WebSocket):
    await ws.accept()
    for _ in range(50):
        if MOUTH.running:
            break
        await asyncio.sleep(0.1)
    if not MOUTH.running:
        await ws.send_text(json.dumps({"error": "Mouth correction not ready"}))
        await ws.close()
        return
    loop2 = asyncio.get_event_loop()
    try:
        while True:
            try:
                _, result_b64, face_found = await loop2.run_in_executor(_gl_executor, MOUTH.grab)
            except Exception as e:
                print(f"[Mouth] Frame grab error: {e}")
                await asyncio.sleep(0.1)
                continue
            if result_b64 is None:
                await asyncio.sleep(0.02)
                continue
            await ws.send_text(json.dumps({"result": result_b64, "face_found": face_found}))
            await asyncio.sleep(0.001)
    except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError):
        MOUTH.close()
        print("[Mouth] Client disconnected")
    except asyncio.CancelledError:
        print("[Mouth] Server shutting down")
    except Exception as e:
        print(f"[Mouth WS] error: {e}")


@app.websocket("/ws/mouth/controls")
async def ws_mouth_ctrl(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = json.loads(await ws.receive_text())
            m = MOUTH.mouth
            if "offset_x" in data:
                m.offset_x = int(data["offset_x"])
            if "offset_y" in data:
                m.offset_y = int(data["offset_y"])
            if "scale_delta" in data:
                m.scale_delta = float(data["scale_delta"])
            await ws.send_text(json.dumps({"ok": True}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[Mouth Ctrl WS] error: {e}")


@app.get("/brow")
async def brow_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-Look — Brow Shaper + Gesture + Speech</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#0e0f14;--panel:#16181f;--panel2:#1c1f2b;--border:#32374a;--accent:#e0a050;--align:#1e9fff;--warn:#dc5050;--green:#28c868;--cyan:#14d4e0;--red:#e03232;--text-hi:#eaeaf0;--text-lo:#646878;--text-mid:#9096ac;--radius:10px;--mono:'Space Mono',monospace;--ui:'Rajdhani',sans-serif}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text-hi);font-family:var(--ui);min-height:100vh;display:flex;flex-direction:column;overflow-x:hidden}
.layout{display:grid;grid-template-columns:270px 1fr 220px;grid-template-rows:52px 1fr 44px;min-height:100vh}
.topbar{grid-column:1/-1;background:var(--panel);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 18px;gap:14px}
.logo{font-family:var(--mono);font-size:.9rem;font-weight:700;color:var(--accent);letter-spacing:.15em}
.logo span{color:var(--text-lo);font-weight:400}
.sep{width:1px;height:22px;background:var(--border)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--warn);box-shadow:0 0 6px var(--warn);transition:.3s}
.dot.on{background:var(--accent);box-shadow:0 0 8px var(--accent)}
.st{font-family:var(--mono);font-size:.58rem;color:var(--warn);letter-spacing:.1em;transition:.3s}
.st.on{color:var(--accent)}
.ml{margin-left:auto;display:flex;gap:10px;align-items:center}
.badge{font-family:var(--mono);font-size:.58rem;color:var(--text-lo);background:var(--panel2);border:1px solid var(--border);border-radius:4px;padding:2px 8px}
.pl{background:var(--panel);border-right:1px solid var(--border);padding:14px 12px;display:flex;flex-direction:column;gap:16px;overflow-y:auto}
.pr{background:var(--panel);border-left:1px solid var(--border);padding:14px 12px;display:flex;flex-direction:column;gap:14px;overflow-y:auto}
.va{background:#08090c;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden}
#vc{max-width:100%;max-height:100%;display:block}
.brackets{position:absolute;inset:0;pointer-events:none}
.brackets svg{width:100%;height:100%}
.br{stroke:var(--accent);stroke-width:2;fill:none;opacity:.5;transition:.3s}
.br.warn{stroke:var(--warn);animation:blink 1s step-start infinite}
@keyframes blink{50%{opacity:0}}
.scanline{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:.18;pointer-events:none;animation:scan 5s linear infinite}
@keyframes scan{0%{top:0}100%{top:100%}}
#overlay{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:rgba(8,9,12,.92);gap:10px;z-index:10}
#overlay .ot{font-family:var(--mono);font-size:.7rem;color:var(--accent);letter-spacing:.15em}
#overlay .os{font-family:var(--mono);font-size:.6rem;color:var(--text-lo)}
.spin{width:26px;height:26px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.bot{grid-column:1/-1;background:var(--panel);border-top:1px solid var(--border);display:flex;align-items:center;padding:0 18px;gap:18px}
.hi{display:flex;align-items:center;gap:6px;font-size:.68rem;color:var(--text-lo);font-family:var(--mono)}
.hk{background:var(--panel2);border:1px solid var(--border);border-radius:3px;padding:1px 5px;font-size:.58rem;color:var(--accent)}
.sec{font-family:var(--mono);font-size:.56rem;color:var(--text-lo);letter-spacing:.15em;text-transform:uppercase;margin-bottom:7px;display:flex;align-items:center;gap:8px}
.sec::after{content:'';flex:1;height:1px;background:var(--border)}
.ib{background:var(--panel2);border:1px solid var(--border);border-radius:var(--radius);padding:10px 12px;font-family:var(--mono);font-size:.54rem;color:var(--text-lo);line-height:1.85;letter-spacing:.04em}
.ib b{color:var(--text-mid)}
.ib .gl{color:var(--accent);font-size:.65rem;font-weight:700}
.ib .sp{color:var(--cyan);font-size:.65rem}
.ib .st2{color:var(--green);font-size:.55rem}
.lp{background:rgba(30,159,255,.06);border:1px solid rgba(30,159,255,.25);border-radius:var(--radius);padding:8px 10px;font-family:var(--mono);font-size:.54rem;color:var(--align);line-height:1.9}
.btn-snap{width:100%;padding:10px;background:linear-gradient(135deg,rgba(224,160,80,.15),rgba(224,160,80,.05));border:1px solid var(--accent);border-radius:var(--radius);color:var(--accent);font-family:var(--mono);font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;cursor:pointer;transition:.2s}
.btn-snap:hover{box-shadow:0 0 16px rgba(224,160,80,.25)}
.err{background:rgba(220,80,80,.12);border:1px solid var(--warn);border-radius:var(--radius);padding:8px 12px;font-family:var(--mono);font-size:.57rem;color:var(--warn);display:none}
</style>
</head>
<body>
<div class="layout">
  <header class="topbar">
    <div class="logo">V-Look <span>Brow+Sign+Speech</span></div>
    <div class="sep"></div>
    <a href="/" style="font-family:var(--mono);font-size:.58rem;color:var(--text-lo);background:var(--panel2);border:1px solid var(--border);border-radius:6px;padding:5px 12px;cursor:pointer;letter-spacing:.08em;text-decoration:none;transition:.2s">Hair AR</a>
    <a href="/nose" style="font-family:var(--mono);font-size:.58rem;color:var(--text-lo);background:var(--panel2);border:1px solid var(--border);border-radius:6px;padding:5px 12px;cursor:pointer;letter-spacing:.08em;text-decoration:none;transition:.2s">Nose Filter</a>
    <a href="/scar" style="font-family:var(--mono);font-size:.58rem;color:var(--text-lo);background:var(--panel2);border:1px solid var(--border);border-radius:6px;padding:5px 12px;cursor:pointer;letter-spacing:.08em;text-decoration:none;transition:.2s">Scar Removal</a>
    <a href="/brow" style="font-family:var(--mono);font-size:.58rem;color:var(--accent);background:rgba(224,160,80,.08);border:1px solid var(--accent);border-radius:6px;padding:5px 12px;cursor:pointer;letter-spacing:.08em;text-decoration:none;transition:.2s">Brow Shaper</a>
    <a href="/mouth" style="font-family:var(--mono);font-size:.58rem;color:var(--text-lo);background:var(--panel2);border:1px solid var(--border);border-radius:6px;padding:5px 12px;cursor:pointer;letter-spacing:.08em;text-decoration:none;transition:.2s">Mouth</a>
    <div class="sep"></div>
    <div class="dot" id="dot"></div>
    <span class="st" id="st">SCANNING...</span>
    <div class="ml">
      <div class="badge"><span id="fps" style="color:var(--green)">0.0</span> FPS</div>
      <div class="badge" id="wsb" style="color:var(--warn)">WS OFF</div>
    </div>
  </header>

  <aside class="pl">
    <div>
      <div class="sec">Features</div>
      <div class="ib">
        <span class="gl">&#9679; Brow Shaper</span><br>
        Rounded eyebrow overlay<br>
        with natural color matching<br><br>
        <span class="gl">&#9758; Sign Language</span><br>
        HELLO &bull; YES &bull; NO &bull; BYE<br>
        THANKS &bull; LIKE IT &bull; DON'T LIKE<br><br>
        <span class="gl">&#9835; Speech-to-Text</span><br>
        Google Speech Recognition<br>
        shown as subtitles
      </div>
    </div>
    <div>
      <div class="sec">Gesture</div>
      <div class="lp" id="gesture_display">
        <span style="color:var(--text-lo);font-size:.85rem">&#x1f590;</span><br>
        <span id="gesture_text" style="font-weight:700;letter-spacing:.1em">---</span>
      </div>
    </div>
    <div>
      <div class="sec">Speech</div>
      <div class="ib">
        <span class="st2" id="speech_status">Idle</span><br>
        <span class="sp" id="speech_text"></span>
      </div>
    </div>
  </aside>

  <main class="va">
    <div id="overlay">
      <div class="spin"></div>
      <div class="ot">Connecting to brow shaper server&hellip;</div>
      <div class="os">python -m uvicorn server:app --port 8000</div>
    </div>
    <canvas id="vc" width="640" height="480"></canvas>
    <div class="brackets">
      <svg viewBox="0 0 640 480" xmlns="http://www.w3.org/2000/svg">
        <polyline class="br" id="bTL" points="80,110 58,110 58,88"/>
        <polyline class="br" id="bTR" points="560,88 582,88 582,110"/>
        <polyline class="br" id="bBL" points="58,370 58,392 80,392"/>
        <polyline class="br" id="bBR" points="582,370 582,392 560,392"/>
      </svg>
    </div>
    <div class="scanline"></div>
  </main>

  <aside class="pr">
    <div>
      <div class="sec">Capture</div>
      <button class="btn-snap" onclick="snap()">&#9679; Screenshot</button>
    </div>
    <div>
      <div class="sec">How it works</div>
      <div class="ib">
        <b>Eyebrows</b> &mdash; automatically<br>
        shaped with rounded fill using<br>
        MediaPipe face landmarks<br><br>
        <b>Hand gestures</b> &mdash; recognized<br>
        via MediaPipe hand landmarks<br><br>
        <b>Speech</b> &mdash; captured from mic<br>
        and transcribed in real time
      </div>
    </div>
    <div>
      <div class="sec">Tip</div>
      <div class="ib">
        Face the camera directly.<br>
        Show hand signs clearly.<br>
        Speak near the microphone.
      </div>
    </div>
  </aside>

  <footer class="bot">
    <div class="hi"><span class="hk">S</span> snapshot</div>
    <div class="hi"><span class="hk">R</span> reset speech</div>
  </footer>
</div>
<script>
const c=document.getElementById('vc'),ctx=c.getContext('2d');
const img=new Image();
let fpsBuf=[],fpsTs=performance.now();
let ws=null;
let lastGesture="---",lastSpeech="",lastSpeechStatus="Idle";

function connect(){
  ws=new WebSocket('ws://'+location.host+'/ws/brow');
  ws.onopen=()=>{
    document.getElementById('wsb').textContent='WS ON';
    document.getElementById('wsb').style.color='var(--green)';
    document.getElementById('overlay').style.display='none';
  };
  ws.onmessage=e=>{
    const d=JSON.parse(e.data);
    if(d.error){document.getElementById('err').textContent='\\u26a0 '+d.error;document.getElementById('err').style.display='block';return}
    img.onload=()=>ctx.drawImage(img,0,0,c.width,c.height);
    if(d.result)img.src='data:image/jpeg;base64,'+d.result;

    if(d.gesture!==undefined){
      lastGesture=d.gesture||'---';
      document.getElementById('gesture_text').textContent=lastGesture;
    }
    if(d.speech_text!==undefined){
      lastSpeech=d.speech_text;
      document.getElementById('speech_text').textContent=d.speech_text;
    }
    if(d.speech_status!==undefined){
      lastSpeechStatus=d.speech_status;
      const se=document.getElementById('speech_status');
      se.textContent=d.speech_status;
      if(d.speech_status==='Listening...')se.style.color='var(--green)';
      else if(d.speech_status==='Processing...')se.style.color='var(--cyan)';
      else if(d.speech_status==='Got it!')se.style.color='var(--accent)';
      else se.style.color='var(--text-lo)';
    }

    const on=d.result!=null;
    document.getElementById('dot').className='dot'+(on?' on':'');
    document.getElementById('st').className='st'+(on?' on':'');
    document.getElementById('st').textContent=on?'STREAMING':'WAITING...';
    ['bTL','bTR','bBL','bBR'].forEach(id=>{
      document.getElementById(id).className='br'+(on?'':' warn');
    });

    const now=performance.now();
    fpsBuf.push(1000/Math.max(now-fpsTs,1));fpsTs=now;
    if(fpsBuf.length>45)fpsBuf.shift();
    const f=fpsBuf.reduce((a,b)=>a+b,0)/fpsBuf.length;
    const fe=document.getElementById('fps');
    fe.textContent=f.toFixed(1);
    fe.style.color=f>=28?'var(--green)':f>=18?'var(--cyan)':'var(--red)';
  };
  ws.onerror=()=>{document.getElementById('overlay').style.display='flex'};
  ws.onclose=()=>{
    document.getElementById('wsb').textContent='WS OFF';
    document.getElementById('wsb').style.color='#dc5050';
    document.getElementById('overlay').style.display='flex';
    setTimeout(connect,2500);
  };
}
connect();

setInterval(async()=>{
  try{const r=await fetch('/mode');const d=await r.json();if(d.mode!=='brow')window.location.href='/'}catch{}
},1000);

function snap(){
  const a=document.createElement('a');
  a.download='brow_'+Date.now()+'.png';
  a.href=c.toDataURL('image/png');a.click();
}
document.addEventListener('keydown',e=>{
  if(e.key.toLowerCase()==='s')snap();
});
</script>
</body>
</html>
""")


@app.websocket("/ws/brow")
async def ws_brow(ws: WebSocket):
    await ws.accept()
    for _ in range(50):
        if BROW.running:
            break
        await asyncio.sleep(0.1)
    if not BROW.running:
        await ws.send_text(json.dumps({"error": "Brow shaper not ready"}))
        await ws.close()
        return

    loop2 = asyncio.get_event_loop()
    try:
        while True:
            try:
                result_b64, gesture, speech_text, speech_status = await loop2.run_in_executor(_gl_executor, BROW.grab)
            except Exception as e:
                print(f"[Brow] Frame grab error: {e}")
                await asyncio.sleep(0.1)
                continue
            if result_b64 is None:
                await asyncio.sleep(0.02)
                continue
            await ws.send_text(json.dumps({
                "result": result_b64,
                "gesture": gesture,
                "speech_text": speech_text,
                "speech_status": speech_status,
            }))
            await asyncio.sleep(0.001)
    except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError):
        BROW.close()
        print("[Brow] Client disconnected")
    except asyncio.CancelledError:
        print("[Brow] Server shutting down")
    except Exception as e:
        print(f"[Brow WS] error: {e}")
        traceback.print_exc()