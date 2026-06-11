import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


class FaceProcessor:
    def __init__(self, model_path: str, callback):
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,   # not used, saves processing time
            result_callback=callback,
        )
        self.landmarker  = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process_frame(self, frame, timestamp_ms: int) -> bool:
        # Strict monotonic guard — MediaPipe requirement
        if timestamp_ms <= self._last_ts_ms:
            return False
        self._last_ts_ms = timestamp_ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.landmarker.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            timestamp_ms)
        return True

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass