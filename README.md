# V-Look — AR Pre-Op Visualization System

An augmented reality system for cosmetic surgery planning with real-time face tracking, 3D hair overlay, scar/acne detection and removal, and eyebrow shaping.

## Modes

| Mode | Description |
|------|-------------|
| **Hair AR** | Real-time virtual hair overlay on the face using MediaPipe face landmarks + 3D rendering (OpenGL GPU or CPU fallback) |
| **Scar Removal** | Detects scars/acne via HSV+LAB color analysis (or optional U-Net), then inpaints them using TELEA |
| **Brow Shaper** | Enhances eyebrows with smooth curved overlays, auto-sampled skin color |
| **Glasses Viewer** (standalone) | Interactive 3D OBJ viewer with smooth yaw-based fade, drag-to-rotate |

## How It Works

1. **Camera** captures video at 640×480
2. **MediaPipe Face Landmarker** detects 468 facial landmarks (async live stream)
3. **solvePnP** computes 3D head pose (rotation + translation) from 6 key landmarks
4. **3D mesh** (OBJ) is transformed to camera space, projected, and rendered
5. **Face depth map** creates a radial dome approximation for occlusion handling
6. **Result** is composited onto the camera feed and streamed via WebSocket to the browser (or shown in an OpenCV window)

## Project Structure

```
├── main.py              # Desktop app (OpenCV window)
├── server.py            # Web server (FastAPI + WebSocket) — serves all modes
├── Index.html           # Hair AR web UI
├── scar.html            # Scar removal web UI
├── v2.py                # Glasses viewer v2 (standalone)
├── test.py              # Glasses viewer with diagonal tip-to-hinge fade
├── test_occlusion.py    # Occlusion test viewer (no camera needed)
├── check_gpu.py         # GPU diagnostics script
├── core/
│   ├── processor.py     # MediaPipe face landmark async processor
│   ├── geometry.py      # PnP pose solver with 3D face reference
│   ├── renderer.py      # 3D mesh renderer (OpenGL/GPU or CPU fallback)
│   └── face_mask.py     # Face depth map for occlusion
├── ui/
│   └── controls.py      # Window manager, HUD, keyboard input, performance panel
├── modes/
│   └── brow_shaper.py   # Eyebrow enhancement logic
├── models/
│   └── face_landmarker.task  # MediaPipe model
├── assets/
│   ├── hair.obj         # Default hair mesh
│   ├── Glasses.obj      # Glasses mesh
│   └── New folder/      # Alternative meshes
└── version control/     # Design notes and diagnostics
```

## Installation

### Python Dependencies

```bash
# Core (required)
pip install opencv-python numpy mediapipe fastapi uvicorn websockets

# GPU rendering (optional, recommended)
pip install moderngl

# Performance monitoring (optional)
pip install psutil

# GPU stats (optional, NVIDIA only)
pip install pynvml

# Scar removal U-Net (optional, for deep learning scar detection)
pip install torch torchvision Pillow

# Blender / 3D tools
pip install bpy  # optional, for mesh processing
```

### Node Dependencies

```bash
npm install   # installs bleak (Bluetooth LE, currently unused by main app)
```

## How to Run

### Web Server (recommended)

```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser.

- **Hair AR mode** — face tracking + 3D hair overlay with sliders for X/Y/Z offset, scale, style presets
- **Scar Removal mode** — click "Scar Removal" in the top bar, use the before/after slider
- **Brow Shaper mode** — click "Brow Shaper" in the top bar

### Desktop App

```bash
python main.py
```

Controls:
- `1`-`5` — switch color preset
- `C` — cycle color preset
- `R` — reset alignment sliders
- `H` — hide/show HUD panels
- `S` — save screenshot
- `TAB` — toggle alignment tuning panel
- `Q` / `ESC` — quit

### Standalone Viewers

```bash
python v2.py        # Glasses viewer with yaw-based fade
python test.py      # Diagonal tip-to-hinge fade version
python test_occlusion.py   # Test occlusion without camera
python check_gpu.py # See what acceleration is available
```

## Good / Working Well

- **Robust face tracking** — MediaPipe provides stable 468-point landmarks at high speed
- **Temporal smoothing** — rotation matrix SVD blending + translation EMA keeps the overlay stable
- **GPU rendering path** — moderngl provides fast OpenGL-based rendering with depth testing
- **CPU fallback** — works without a GPU (just slower, ~15 FPS instead of 28-30)
- **Face occlusion** — radial depth dome prevents hair from bleeding over the face
- **Web UI** — responsive, dark theme, before/after slider for scar mode, real-time parameter adjustment
- **Performance monitoring** — FPS, CPU, RAM, GPU utilization shown in HUD
- **Scar detection** — HSV+LAB hybrid works well for reddish marks; optional U-Net for deep learning

## Bad / Needs Improvement

- **No requirements.txt** — dependencies are scattered across server.py docstrings and this README; use `pip freeze > requirements.txt` to lock versions
- **server.py is too long** (837 lines) — AR rendering, scar removal, and brow shaper are all in one file; should be split into separate modules/routers
- **Missing error recovery** — if the camera fails mid-stream, the server doesn't auto-restart
- **U-Net weights path** — hardcoded as `unet_scar.pth`; should be configurable via env var
- **No automated tests** — no pytest or unit tests for core components
- **Mixed concerns** — `node_modules/` and `package.json` include `bleak` (Bluetooth) with no apparent use in the main app
- **No Docker support** — would simplify deployment across machines
- **Hardware requirements** — the app expects a webcam; no fallback for headless/offline mode (except test_occlusion.py)
- **Scar mode detection** — sometimes over-detects on skin texture, needs better spatial filtering

## Making It Better

### Short-term
1. `pip freeze > requirements.txt` to lock dependencies
2. Split `server.py` into `routes/hair.py`, `routes/scar.py`, `routes/brow.py`
3. Add `--camera` CLI argument to select camera index
4. Add `--no-webcam` flag that uses a static image for demo

### Medium-term
5. Add Dockerfile + docker-compose for easy deployment
6. Write unit tests with pytest for `core/geometry.py` and `core/renderer.py`
7. Add GitHub Actions CI for linting + testing
8. Make U-Net model path configurable via environment variable or web UI upload

### Long-term
9. Replace OBJ loading with glTF/GLB (modern 3D format with PBR materials)
10. Add multi-face support for group AR
11. Integrate with ESP32 health module (heart rate, SpO₂) via Bluetooth/pyserial
12. Add real-time speech-to-text for hands-free control
13. Port rendering pipeline to WebGL via Three.js to offload GPU work to the browser

## Tech Stack

- **Face tracking:** MediaPipe Face Landmarker (468 landmarks)
- **Pose estimation:** OpenCV solvePnP
- **3D rendering:** ModernGL (OpenGL) or CPU rasterizer fallback
- **Web server:** FastAPI + Uvicorn + WebSockets
- **Frontend:** Vanilla JS, Canvas 2D
- **Deep learning (optional):** PyTorch U-Net for scar segmentation
- **Hardware target:** Jetson Nano / laptop with webcam
