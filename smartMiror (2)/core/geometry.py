"""
core/geometry.py — PnP pose estimation + smoothing
✅ Fixed: Added missing landmarks 57 & 287, robust EPNP fallback for 4-6 points
"""

import cv2
import numpy as np

# ✅ Fixed: Added 57 and 287 (mouth corners) to ensure all 6 PnP points exist
SOLVE_IDX = [
    1, 4, 6, 10, 33, 46, 52, 53, 54, 55, 57, 58, 61, 63, 65, 66, 67,
    70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 105, 107, 109, 127, 
    132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 
    157, 158, 159, 160, 161, 163, 172, 173, 176, 178, 181, 185, 234, 
    246, 249, 251, 263, 276, 282, 283, 284, 285, 287, 288, 293, 295, 
    296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 
    332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 
    379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 
    400, 402, 405, 406, 409, 415, 454, 466
]

# Standard 3D face model (mm) - Y down, Z forward
_FACE_3D_PTS = np.array([
    [  0.0,    0.0,   0.0],  # 1: Nose tip
    [  0.0,  105.0,  50.0],  # 152: Chin
    [- 68.0,  -28.0, -25.0],  # 234: Left temple
    [ 68.0,  -28.0, -25.0],  # 454: Right temple
    [- 48.0,   42.0,  18.0],  # 57:  Left mouth corner
    [ 48.0,   42.0,  18.0],  # 287: Right mouth corner
], dtype=np.float32)

_FACE_3D_LM_IDX = [1, 152, 234, 454, 57, 287]

_DEBUG_PNP = True
_debug_logged = False

class GeometryEngine:
    SOLVE_IDX = SOLVE_IDX

    @staticmethod
    def get_camera_matrix(W: int, H: int, focal_scale: float = 0.85) -> np.ndarray:
        f = focal_scale * float(max(W, H))
        return np.array([[f, 0, W/2.0], [0, f, H/2.0], [0, 0, 1.0]], dtype=np.float32)

    def solve_pose(self, pts2d: np.ndarray, K: np.ndarray,
                   dist: np.ndarray, prev_rvec=None, prev_tvec=None, flags: int = None):
        global _debug_logged
        
        if pts2d is None or len(pts2d) < len(_FACE_3D_LM_IDX):
            return False, None, None

        # Extract 2D subset matching 3D points
        subset_2d = []
        valid_3d = []
        missing = []
        
        for i, lm_idx in enumerate(_FACE_3D_LM_IDX):
            try:
                pos = SOLVE_IDX.index(lm_idx)
                pt = pts2d[pos]
                if np.isfinite(pt).all() and -100 <= pt[0] <= K[0,2]*3 and -100 <= pt[1] <= K[1,2]*3:
                    subset_2d.append(pt)
                    valid_3d.append(_FACE_3D_PTS[i])
                else:
                    missing.append(f"{lm_idx}(OOB)")
            except ValueError:
                missing.append(str(lm_idx))
            except IndexError:
                missing.append(f"{lm_idx}(IDX)")

        if missing:
            if not _debug_logged:
                print(f"[Geometry] ⚠️  Missing PnP landmarks: {', '.join(missing)}")
                _debug_logged = True
            if len(subset_2d) < 4:
                return False, None, None

        subset_2d = np.array(subset_2d, dtype=np.float32)
        valid_3d = np.array(valid_3d, dtype=np.float32)

        # Log coordinates on first run
        if not _debug_logged and _DEBUG_PNP:
            _debug_logged = True
            print(f"\n[Geometry] 📐 PnP INPUT DEBUG ({len(subset_2d)} pts):")
            for i in range(len(valid_3d)):
                print(f"  3D{valid_3d[i].round(1)} -> 2D{subset_2d[i].round(1)}")
            print(f"  K[0,0]={K[0,0]:.0f}\n")

        use_guess = prev_rvec is not None and prev_tvec is not None
        
        # ✅ Robust strategy: EPNP handles 4-6 points reliably; ITERATIVE needs 6+
        strategies = [
            ("EPNP", cv2.SOLVEPNP_EPNP, False),
            ("ITERATIVE + Guess", cv2.SOLVEPNP_ITERATIVE, use_guess),
            ("IPPE", cv2.SOLVEPNP_IPPE, False)
        ]

        for name, flag, guess in strategies:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    valid_3d, subset_2d, K, dist,
                    rvec=prev_rvec, tvec=prev_tvec,
                    useExtrinsicGuess=guess, flags=flag)
                
                if success and rvec is not None and tvec is not None:
                    depth = np.linalg.norm(tvec)
                    # Accept if face is 100mm - 1500mm from camera
                    if 100 < depth < 1500:
                        return True, rvec, tvec
            except Exception:
                continue

        return False, None, None

    @staticmethod
    def smooth_rotation_matrix(prev_R: np.ndarray, new_R: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        if prev_R is None: return new_R.copy()
        blended = (1.0 - alpha) * prev_R + alpha * new_R
        try:
            U, _, Vt = cv2.SVDecomp(blended)
            return U @ Vt
        except Exception:
            for i in range(3): blended[:, i] /= np.linalg.norm(blended[:, i]) + 1e-8
            return blended