import cv2
import numpy as np
import threading
import time
from core.renderer  import ARRenderer
from core.geometry  import GeometryEngine
from core.face_mask import build_face_depth_map
from core.processor import FaceProcessor

class HairARMode:
    def __init__(self, model_path: str, obj_path: str, W: int, H: int, K: np.ndarray):
        self.W = W
        self.H = H
        self.K = K.copy()
        self.geo = GeometryEngine()
        self.renderer = ARRenderer(obj_path, normalize_span=4.0)
        self.processor = FaceProcessor(model_path, self._on_result)

        self._lock = threading.Lock()
        self._fd = {"found": False, "R_mat": None, "tvec": None, "landmarks": None}
        self._t0 = time.perf_counter()
        self._last_ts = -1

    def _on_result(self, result, image, ts_ms):
        if not result.face_landmarks:
            with self._lock:
                self._fd["found"] = False
            return

        dist  = np.zeros((4, 1), np.float32)
        lms   = result.face_landmarks[0]
        pts2d = np.array([[lms[i].x * self.W, lms[i].y * self.H]
                          for i in self.geo.SOLVE_IDX], dtype=np.float32)

        with self._lock:
            prev_r = None
            prev_t = self._fd["tvec"]
            if self._fd["R_mat"] is not None:
                prev_r, _ = cv2.Rodrigues(self._fd["R_mat"])

        ok, rvec, tvec = self.geo.solve_pose(pts2d, self.K, dist, prev_r, prev_t)
        if not ok:
            return

        # ✅ USE RAW rvec DIRECTLY (Mesh axes are already aligned in renderer.py)
        R_new, _ = cv2.Rodrigues(rvec)

        with self._lock:
            # HEAVY ROTATION SMOOTHING
            self._fd["R_mat"] = self.geo.smooth_rotation_matrix(
                self._fd["R_mat"], R_new, alpha=0.12
            )

            # HEAVY TRANSLATION SMOOTHING + JUMP REJECTION
            if prev_t is not None:
                jump = np.linalg.norm(tvec - prev_t)
                if jump < 80.0:
                    self._fd["tvec"] = 0.15 * tvec + 0.85 * prev_t
            else:
                self._fd["tvec"] = tvec.copy()

            self._fd["landmarks"] = lms
            self._fd["found"] = True

    def process(self, frame, ui_params: dict) -> np.ndarray:
        ts = int((time.perf_counter() - self._t0) * 1000)
        if ts > self._last_ts:
            self._last_ts = ts
            self.processor.process_frame(frame, ts)

        with self._lock:
            if not self._fd["found"]:
                return frame
            rvec_f, _ = cv2.Rodrigues(self._fd["R_mat"])
            tvec_f    = self._fd["tvec"]
            lms_f     = self._fd["landmarks"]

        face_depth = None
        if lms_f is not None:
            face_depth = build_face_depth_map(lms_f, self.W, self.H, rvec_f, tvec_f, self.K)

        return self.renderer.render(
            frame, rvec_f, tvec_f, self.K,
            style=ui_params.get("style"),
            offset_x=ui_params.get("offset_x", 0.0),
            offset_y=ui_params.get("offset_y", 0.0),
            offset_z=ui_params.get("offset_z", 0.0),
            mesh_scale=ui_params.get("mesh_scale", 450.0),
            face_depth_map=face_depth,
            occ_margin=15.0,
            opacity=0.92,
        )

    def close(self):
        self.processor.close()
        try: self.renderer.close()
        except Exception: pass