"""
core/processor.py — MediaPipe face landmarker wrappers
✅ Fixed: Added input type validation to prevent cv2.cvtColor crash
"""

import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode as VisionTaskRunningMode


class FaceProcessor:
    """Async live-stream processor — used by Hair AR mode."""

    def __init__(self, model_path: str, callback):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}")
        
        print(f"[FaceProcessor] Loading async model: {os.path.basename(model_path)}")
        
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            result_callback=callback,
        )
        try:
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to create async FaceLandmarker: {e}")
        
        self._last_ts_ms = -1

    def process_frame(self, frame, timestamp_ms: int) -> bool:
        if timestamp_ms <= self._last_ts_ms:
            return False
        self._last_ts_ms = timestamp_ms
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.landmarker.detect_async(mp_image, timestamp_ms)
            return True
        except Exception as e:
            print(f"[WARN] Async detect failed: {e}")
            return False

    def close(self):
        if hasattr(self, 'landmarker'):
            try:
                self.landmarker.close()
                print("[FaceProcessor] Async landmarker closed")
            except Exception as e:
                print(f"[WARN] Error closing async landmarker: {e}")


class SyncProcessor:
    """Synchronous VIDEO-mode processor — used by Brow / Scar / Mouth modes."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}")
            
        print(f"[SyncProcessor] Loading sync model: {os.path.basename(model_path)}")
        
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
        )
        try:
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to create sync FaceLandmarker: {e}")
        
        self._ts_ms = 0

    def detect(self, frame):
        """
        Detect face landmarks synchronously.
        ✅ Fixed: Robust type checking to handle mp.Image, None, or wrong formats.
        """
        # ✅ Guard against invalid inputs (caused the cvtColor crash)
        if frame is None:
            return None
            
        # Convert mp.Image → numpy array if accidentally passed
        if not isinstance(frame, np.ndarray):
            try:
                if hasattr(frame, 'numpy_view'):  # mp.Image fallback
                    frame = cv2.cvtColor(frame.numpy_view(), cv2.COLOR_RGB2BGR)
                else:
                    return None
            except Exception:
                return None

        self._ts_ms += 33
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect_for_video(mpi, self._ts_ms)
            return res.face_landmarks[0] if res.face_landmarks else None
        except Exception as e:
            print(f"[WARN] Sync detect failed: {e}")
            return None

    def close(self):
        if hasattr(self, 'landmarker'):
            try:
                self.landmarker.close()
                print("[SyncProcessor] Sync landmarker closed")
            except Exception as e:
                print(f"[WARN] Error closing sync landmarker: {e}")