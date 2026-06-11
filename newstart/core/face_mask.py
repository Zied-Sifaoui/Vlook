import cv2
import numpy as np

FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]


def build_face_depth_map(landmarks, W: int, H: int, rvec, tvec, K) -> np.ndarray:
    """
    3D depth buffer of the face using a radial 'dome' approximation.
    Center (nose) is closest to camera; edges curve back 50mm for skull wrap.
    Neck extension covers the throat area below the chin.
    """
    depth_map = np.full((H, W), 999999.0, dtype=np.float32)

    tvec_flat = np.array(tvec, dtype=np.float32).flatten()
    face_z_center = float(tvec_flat[2])
    depth_scale = face_z_center * 0.25

    pts2d = np.array([[lm.x * W, lm.y * H] for lm in landmarks], dtype=np.float32)
    pts_z = np.array([face_z_center + (lm.z * depth_scale) for lm in landmarks],
                     dtype=np.float32)

    # Build face oval + neck polygon (subdivided for smooth contour)
    chin_pt   = pts2d[152]
    left_jaw  = pts2d[377]
    right_jaw = pts2d[148]
    face_h    = abs(chin_pt[1] - pts2d[10][1])
    neck_y    = min(H - 1, int(chin_pt[1] + face_h * 0.35))
    neck_half = int((right_jaw[0] - left_jaw[0]) * 0.35)
    neck_l    = np.array([int(chin_pt[0] - neck_half), neck_y], dtype=np.int32)
    neck_r    = np.array([int(chin_pt[0] + neck_half), neck_y], dtype=np.int32)

    def _subdiv(pts, n=1):
        out = []
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            out.append(a)
            for j in range(1, n + 1):
                t = j / (n + 1)
                out.append(((1.0 - t) * a + t * b).astype(np.int32))
        return np.array(out)

    oval = FACE_OVAL_IDX
    split_pos = oval.index(152)
    full_pts = []
    for idx in oval[:split_pos + 1]:
        full_pts.append(pts2d[idx].astype(np.int32))
    full_pts.extend([neck_l, neck_r])
    for idx in oval[split_pos + 1:]:
        full_pts.append(pts2d[idx].astype(np.int32))

    full_hull = _subdiv(np.array(full_pts), n=1)
    face_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(face_mask, [full_hull], 255)

    # Radial dome — center (nose) closest, edges curve back smoothly
    dist_transform = cv2.distanceTransform(face_mask, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    nose_z = pts_z[1]
    forehead_z = pts_z[10]
    chin_z = pts_z[152]
    avg_edge_z = (forehead_z + chin_z) / 2.0 + 50.0

    y_idx, x_idx = np.where(face_mask > 0)
    if len(y_idx) == 0:
        return depth_map

    pixel_dists = dist_transform[y_idx, x_idx]
    depth_map[y_idx, x_idx] = nose_z + (1.0 - pixel_dists) * (avg_edge_z - nose_z)

    depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)

    return depth_map