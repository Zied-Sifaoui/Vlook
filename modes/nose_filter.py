import cv2
import numpy as np
from collections import defaultdict
from typing import Tuple


class NoseRenderer:
    def __init__(self, obj_path: str, normalize_span: float = 1.5, baseline=(0, 0, 0)):
        verts_raw, self.faces = self._load_obj(obj_path)
        centered = verts_raw - verts_raw.mean(axis=0)
        x_span = centered[:, 0].max() - centered[:, 0].min()
        if x_span > 1e-6:
            centered *= (normalize_span / x_span)
        self._verts_norm = centered.astype(np.float32)
        self.baseline = np.array(baseline, np.float32)
        self._light = np.array([0.0, -0.2, -1.0], np.float32)
        self._light /= np.linalg.norm(self._light)

    @staticmethod
    def _load_obj(path) -> Tuple[np.ndarray, np.ndarray]:
        verts, faces = [], []
        with open(path) as fh:
            for line in fh:
                tok = line.split()
                if not tok or tok[0] not in ("v", "f"):
                    continue
                if tok[0] == "v":
                    verts.append((float(tok[1]), float(tok[2]), float(tok[3])))
                elif tok[0] == "f":
                    idx = [int(t.split("/")[0]) - 1 for t in tok[1:]]
                    for i in range(1, len(idx) - 1):
                        faces.append((idx[0], idx[i], idx[i + 1]))
        return np.array(verts, np.float32), np.array(faces, np.int32)

    def _build(self, mesh_scale, ox, oy, oz):
        v = (self._verts_norm * mesh_scale).copy()
        v[:, 0] += ox + self.baseline[0]
        v[:, 1] += oy + self.baseline[1]
        v[:, 2] += oz + self.baseline[2]
        return v

    def _sample_skin_color(self, frame, pts2d) -> np.ndarray:
        H, W = frame.shape[:2]
        xs, ys = pts2d[:, 0], pts2d[:, 1]
        x0 = int(np.clip(xs.min(), 0, W - 1))
        y0 = int(np.clip(ys.min(), 0, H - 1))
        x1 = int(np.clip(xs.max(), 0, W - 1))
        y1 = int(np.clip(ys.max(), 0, H - 1))
        pad = max(10, int((x1 - x0) * 0.25))
        sy0 = max(0, y0 - pad)
        sy1 = max(0, y0 - 2)
        sx0 = max(0, x0)
        sx1 = min(W - 1, x1)
        if sy1 > sy0 and sx1 > sx0:
            patch = frame[sy0:sy1, sx0:sx1].reshape(-1, 3).astype(np.float32)
            brightness = patch.mean(axis=1)
            valid = patch[(brightness > 40) & (brightness < 220)]
            if len(valid) >= 20:
                return valid.mean(axis=0)
        return np.array([160, 185, 210], np.float32)

    def render(self, frame, rvec, tvec, K,
               base_color=None,
               offset_x=-15.0, offset_y=-110.0, offset_z=-100.0,
               mesh_scale=135.0,
               **kwargs):

        H, W = frame.shape[:2]

        verts = self._build(mesh_scale, offset_x, offset_y, offset_z)
        pts2d, _ = cv2.projectPoints(verts, rvec, tvec, K, np.zeros((4, 1)))
        pts2d = pts2d.reshape(-1, 2).astype(np.float32)

        if base_color is not None:
            base_color = np.array(base_color, np.float32)
        else:
            base_color = self._sample_skin_color(frame, pts2d)
        base_color = np.clip(base_color * 0.6 + 255 * 0.4, 0, 255)

        R, _ = cv2.Rodrigues(rvec)
        v_cam = (R @ verts.T).T + tvec.reshape(1, 3)

        f0, f1, f2 = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
        p0, p1, p2 = pts2d[f0], pts2d[f1], pts2d[f2]

        e1 = v_cam[f1] - v_cam[f0]
        e2 = v_cam[f2] - v_cam[f0]
        normals = np.cross(e1, e2)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8

        diffuse = normals @ self._light
        brightness = np.clip(diffuse * 0.15 + 0.85, 0.80, 1.0)

        cross_z = (
            (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
        )
        avg_z = (v_cam[:, 2][f0] + v_cam[:, 2][f1] + v_cam[:, 2][f2]) / 3.0

        vis_idx = np.where(cross_z > 0)[0]
        sort_ord = vis_idx[np.argsort(-avg_z[vis_idx])]

        overlay = np.zeros((H, W, 3), np.uint8)
        alpha_mask = np.zeros((H, W), np.float32)

        N_B = 20
        b_q = np.floor(np.power(brightness, 0.9) * N_B).clip(0, N_B).astype(np.int32)

        buckets = defaultdict(list)
        for i in sort_ord:
            buckets[int(b_q[i])].append(np.array([p0[i], p1[i], p2[i]], np.int32))

        for bi, tris in buckets.items():
            b_val = max(bi / N_B, 0.75)
            draw_color = tuple(int(c * b_val) for c in base_color)
            cv2.fillPoly(overlay, tris, draw_color)
            cv2.fillPoly(alpha_mask, tris, 1.0)

        small = cv2.resize(alpha_mask, (W // 2, H // 2))
        u_alpha = cv2.UMat(small)
        u_alpha = cv2.GaussianBlur(u_alpha, (5, 5), 2)
        a = cv2.resize(u_alpha.get(), (W, H))[:, :, None]

        MAX_ALPHA = 0.80
        a = a * MAX_ALPHA

        out = (
            overlay.astype(np.float32) * a
            + frame.astype(np.float32) * (1.0 - a)
        )
        return out.astype(np.uint8)
