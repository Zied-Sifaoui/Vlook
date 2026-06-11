import cv2
import numpy as np
from typing import Optional, Tuple

# Head reference points in mm — tuned for crown/hair placement.
# These sit higher on the face than the glasses reference.
FACE_3D_HAIR = np.array([
    [  0.0,    0.0,    0.0],   # nose tip (stable anchor)
    [  0.0, -330.0,  -65.0],   # chin
    [-225.0,  170.0, -135.0],  # left temple
    [ 225.0,  170.0, -135.0],  # right temple
    [-150.0, -150.0, -125.0],  # left mouth corner
    [ 150.0, -150.0, -125.0],  # right mouth corner
], dtype=np.float32)


class GeometryEngine:
    # Mediapipe landmark indices — same 6 as before for stable solvePnP
    # but chosen to give a clean head orientation (not biased toward nose)
    SOLVE_IDX = [1, 152, 33, 263, 61, 291]

    def get_camera_matrix(self, w: int, h: int) -> np.ndarray:
        return np.array([
            [float(w), 0.0,      w / 2.0],
            [0.0,      float(w), h / 2.0],
            [0.0,      0.0,          1.0],
        ], dtype=np.float32)

    def solve_pose(self, pts2d, K, dist,
                   prev_rvec=None, prev_tvec=None):
        """Uses FACE_3D_HAIR reference — same logic, different 3-D anchor."""
        pts3d = np.ascontiguousarray(FACE_3D_HAIR).reshape(-1, 1, 3).astype(np.float32)
        pts2  = np.ascontiguousarray(pts2d).reshape(-1, 1, 2).astype(np.float32)
        have  = prev_rvec is not None and prev_tvec is not None
        r0    = prev_rvec if have else np.zeros((3, 1), np.float32)
        t0    = prev_tvec if have else np.zeros((3, 1), np.float32)
        try:
            if have:
                ok, rvec, tvec = cv2.solvePnP(
                    pts3d, pts2, K, dist,
                    rvec=r0.copy(), tvec=t0.copy(),
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                ok, rvec, tvec = cv2.solvePnP(
                    pts3d, pts2, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            return False, r0, t0
        return bool(ok), rvec, tvec

    @staticmethod
    def smooth_rotation_matrix(old_R, new_R, alpha=0.0):
        if old_R is None or alpha == 0.0:
            return new_R
        blended = alpha * old_R + (1.0 - alpha) * new_R
        u, _, vt = np.linalg.svd(blended)
        R = u @ vt
        if np.linalg.det(R) < 0:
            u[:, -1] *= -1
            R = u @ vt
        return R