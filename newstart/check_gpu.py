"""
check_gpu.py  —  Run this to see what acceleration is available
================================================================
python check_gpu.py
"""

import sys

print("=" * 55)
print("  V-Look GPU / Acceleration Diagnostic")
print("=" * 55)

# ── Python ────────────────────────────────────────────────────
print(f"\nPython : {sys.version.split()[0]}")

# ── OpenCV ────────────────────────────────────────────────────
import cv2
print(f"OpenCV : {cv2.__version__}")

cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
print(f"OpenCV CUDA devices : {cuda_count}")
if cuda_count > 0:
    for i in range(cuda_count):
        info = cv2.cuda.DeviceInfo(i)
        print(f"  GPU {i}: {info.name()}  "
              f"{info.totalMemory() // 1024**2} MB")
else:
    print("  → OpenCV was NOT built with CUDA.")
    print("    Install 'opencv-contrib-python' with CUDA support for GPU fillPoly.")

# OpenCL (lighter GPU path, often available)
ocl = cv2.ocl.haveOpenCL()
print(f"OpenCV OpenCL : {'YES' % () if ocl else 'NO'}")
if ocl:
    cv2.ocl.setUseOpenCL(True)
    print(f"  OpenCL in use : {cv2.ocl.useOpenCL()}")

# ── NumPy ─────────────────────────────────────────────────────
import numpy as np
print(f"\nNumPy  : {np.__version__}")
cfg = np.__config__
blas = getattr(cfg, 'blas_opt_info', {}) or getattr(cfg, 'blas_ilp64_opt_info', {})
libs = blas.get('libraries', ['unknown'])
print(f"BLAS   : {libs}")

# ── ONNX Runtime (fast CPU/GPU inference) ────────────────────
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"\nONNX Runtime : {ort.__version__}")
    print(f"  Providers  : {providers}")
    if 'CUDAExecutionProvider' in providers:
        print("  → CUDA available for ONNX models!")
    elif 'DmlExecutionProvider' in providers:
        print("  → DirectML available (Windows GPU)!")
    else:
        print("  → CPU only for ONNX.")
except ImportError:
    print("\nONNX Runtime : NOT installed")

# ── PyTorch (if installed) ────────────────────────────────────
try:
    import torch
    print(f"\nPyTorch : {torch.__version__}")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
except ImportError:
    print("\nPyTorch : NOT installed")

# ── CuPy (GPU numpy drop-in) ─────────────────────────────────
try:
    import cupy as cp
    print(f"\nCuPy   : {cp.__version__}  — GPU numpy available!")
    print(f"  Device : {cp.cuda.Device().attributes}")
except ImportError:
    print("\nCuPy   : NOT installed  (GPU numpy drop-in, optional)")

# ── MediaPipe ─────────────────────────────────────────────────
try:
    import mediapipe as mp
    print(f"\nMediaPipe : {mp.__version__}")
except ImportError:
    print("\nMediaPipe : NOT installed")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  RECOMMENDATION")
print("=" * 55)
if cuda_count > 0:
    print("  OpenCV CUDA is available — GPU rendering possible.")
elif ocl:
    print("  OpenCV OpenCL is available — partial GPU speedup.")
    print("  For full GPU: install opencv-python built with CUDA.")
else:
    print("  No GPU path found in OpenCV.")
    print("  Best options to speed up rendering:")
    print("  1. pip install cupy-cuda11x  (if NVIDIA GPU + CUDA 11)")
    print("     or cupy-cuda12x           (if CUDA 12)")
    print("  2. pip install onnxruntime-gpu")
    print("  3. Reduce camera resolution: 640x480 instead of 1280x720")
    print("     — halves the pixel count → ~2x faster rendering")
print("=" * 55)