"""
FaceAR Flask Server  —  PC & Jetson Nano
Captures from local camera, processes with AI, streams MJPEG to phone.
"""

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import base64
import cv2
import numpy as np
import logging
import traceback
import threading
import time
from typing import Optional
from processors import FEATURES, get_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app  = Flask(__name__)
CORS(app)

# ── Jetson CSI GStreamer pipeline ─────────────────────────────────────────────
def _csi_pipeline(w=1280, h=720, fps=30):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h}, framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )

# ── Camera streaming state ────────────────────────────────────────────────────
_cam_thread:   Optional[threading.Thread] = None
_cam_running   = False
_cam_paused    = False
_frame_lock    = threading.Lock()
_latest_frame: Optional[np.ndarray] = None
_active_feature: str = ""

# ── Processor cache ───────────────────────────────────────────────────────────
_processor_cache = {}

def get_cached_processor(feature_id: str):
    if feature_id not in _processor_cache:
        logger.info(f"Loading processor: {feature_id}")
        _processor_cache[feature_id] = get_processor(feature_id)
    return _processor_cache[feature_id]


def _resolve_camera(cam_src):
    """Convert camera source to OpenCV-compatible value."""
    if cam_src == "csi":
        return _csi_pipeline()
    try:
        return int(cam_src)
    except (ValueError, TypeError):
        return str(cam_src)  # path like /dev/video1


def _camera_loop(feature_id: str, cam_src):
    global _cam_running, _latest_frame

    src = _resolve_camera(cam_src)
    is_gst = isinstance(src, str) and "nvargus" in src

    cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER if is_gst else cv2.CAP_ANY)
    if not is_gst:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        logger.error(f"Cannot open camera: {src}")
        _cam_running = False
        return

    logger.info(f"Camera opened: {src}  |  feature: {feature_id}")
    processor = get_cached_processor(feature_id)

    while _cam_running:
        if _cam_paused:
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        try:
            processed = processor.process(frame)
            out = processed if processed is not None else frame
        except Exception as e:
            logger.warning(f"Processor error: {e}")
            out = frame

        with _frame_lock:
            _latest_frame = out

    cap.release()
    logger.info("Camera loop stopped")


def _stop_camera():
    global _cam_running, _cam_thread, _latest_frame
    _cam_running = False
    if _cam_thread and _cam_thread.is_alive():
        _cam_thread.join(timeout=3.0)
    _cam_thread = None
    with _frame_lock:
        _latest_frame = None


# ── Frame encode helpers ──────────────────────────────────────────────────────
def decode_frame(b64: str) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def encode_frame(frame: np.ndarray, quality: int = 75) -> str:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('utf-8')


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "message": "FaceAR Server is running"})


@app.route('/features', methods=['GET'])
def list_features():
    return jsonify({"features": FEATURES})


@app.route('/launch', methods=['POST'])
def launch_feature():
    """
    Start capturing from camera and processing with the given feature.
    POST JSON:
    {
        "feature": "nose",
        "camera": 0          // 0=default webcam, 1=second USB, "csi"=Jetson CSI, "/dev/video1"=path
    }
    """
    global _cam_thread, _cam_running, _cam_paused, _active_feature
    try:
        data = request.get_json(force=True)
        feature_id = data.get('feature', '')
        cam_src    = data.get('camera', 0)

        if not feature_id:
            return jsonify({"success": False, "error": "Missing feature"}), 400

        _stop_camera()

        _cam_running   = True
        _cam_paused    = False
        _active_feature = feature_id
        _cam_thread = threading.Thread(
            target=_camera_loop,
            args=(feature_id, cam_src),
            daemon=True
        )
        _cam_thread.start()
        logger.info(f"Launched '{feature_id}' on camera '{cam_src}'")
        return jsonify({"success": True, "feature": feature_id})

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop_feature():
    _stop_camera()
    return jsonify({"success": True})


@app.route('/pause', methods=['POST'])
def pause_feature():
    global _cam_paused
    _cam_paused = True
    return jsonify({"success": True})


@app.route('/resume', methods=['POST'])
def resume_feature():
    global _cam_paused
    _cam_paused = False
    return jsonify({"success": True})


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "running": _cam_running,
        "paused":  _cam_paused,
        "feature": _active_feature,
    })


@app.route('/stream')
def stream():
    """MJPEG stream — open in browser or consume on Android."""
    def generate():
        while True:
            with _frame_lock:
                frame = _latest_frame

            if frame is None:
                time.sleep(0.033)
                continue

            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                time.sleep(0.033)
                continue

            jpg = buf.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(jpg)).encode() + b'\r\n'
                b'\r\n' + jpg + b'\r\n'
            )
            time.sleep(0.033)   # ~30 fps cap

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/process', methods=['POST'])
def process_frame():
    """Legacy single-frame endpoint (kept for backwards compatibility)."""
    try:
        data       = request.get_json(force=True)
        feature_id = data.get('feature', '')
        frame_b64  = data.get('frame', '')

        if not feature_id or not frame_b64:
            return jsonify({"success": False, "error": "Missing feature or frame"}), 400

        frame = decode_frame(frame_b64)
        if frame is None:
            return jsonify({"success": False, "error": "Failed to decode frame"}), 400

        result = get_cached_processor(feature_id).process(frame) or frame
        return jsonify({"success": True, "frame": encode_frame(result)})

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    print("=" * 50)
    print("  FaceAR Server  —  PC / Jetson Nano")
    print("  Stream:  http://localhost:5000/stream")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
