import cv2
import numpy as np
from collections import defaultdict


class NoseRenderer:
    def __init__(self, obj_path: str):
        verts_raw, self.faces = self._load_obj(obj_path)
        self._verts_norm = verts_raw - verts_raw.mean(axis=0)
        self._light = np.array([0.0, 0.0, -1.0], np.float32)

    @staticmethod
    def _load_obj(path):
        verts, faces = [], []
        with open(path) as fh:
            for line in fh:
                tok = line.split()
                if not tok: continue
                if tok[0] == "v":
                    verts.append((float(tok[1]), float(tok[2]), float(tok[3])))
                elif tok[0] == "f":
                    idx = [int(t.split("/")[0]) - 1 for t in tok[1:]]
                    for i in range(1, len(idx) - 1):
                        faces.append((idx[0], idx[i], idx[i + 1]))
        return np.array(verts, np.float32), np.array(faces, np.int32)

    def _build(self, mesh_scale, ox, oy, oz):
        v = (self._verts_norm * mesh_scale).copy()
        v[:, 0] += ox
        v[:, 1] += oy
        v[:, 2] += oz
        return v

    def render(self, frame, rvec, tvec, K,
               base_color=None,
               offset_x=0.0, offset_y=170.0, offset_z=0.0,
               mesh_scale=3060.0):
        H, W = frame.shape[:2]
        if base_color is None:
            base_color = np.array([160, 190, 230], np.float32)
        else:
            base_color = np.array(base_color, np.float32)

        verts = self._build(mesh_scale, offset_x, offset_y, offset_z)
        pts2d, _ = cv2.projectPoints(verts, rvec, tvec, K, np.zeros((4, 1)))
        pts2d = pts2d.reshape(-1, 2).astype(np.float32)

        R, _ = cv2.Rodrigues(rvec)
        v_cam = (R @ verts.T).T + tvec.reshape(1, 3)

        f0, f1, f2 = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
        p0, p1, p2 = pts2d[f0], pts2d[f1], pts2d[f2]

        e1 = v_cam[f1] - v_cam[f0]
        e2 = v_cam[f2] - v_cam[f0]
        normals = np.cross(e1, e2)
        nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        normals /= nlen

        diffuse = normals @ self._light
        brightness = np.clip(diffuse * 0.3 + 0.7, 0.6, 1.0)

        cross_z = ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                   - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
        avg_z = (v_cam[:, 2][f0] + v_cam[:, 2][f1] + v_cam[:, 2][f2]) / 3.0

        vis_idx = np.where(cross_z > 0)[0]
        sort_ord = vis_idx[np.argsort(-avg_z[vis_idx])]

        overlay = np.zeros((H, W, 3), np.uint8)
        alpha_mask = np.zeros((H, W), np.float32)

        N_B = 20
        b_q = np.floor(np.power(brightness, 0.8) * N_B).clip(0, N_B).astype(np.int32)

        buckets = defaultdict(list)
        for i in sort_ord:
            buckets[int(b_q[i])].append(np.array([p0[i], p1[i], p2[i]], np.int32))

        for bi, tris in buckets.items():
            b_val = bi / N_B
            draw_color = tuple(int(c * b_val) for c in base_color)
            cv2.fillPoly(overlay, tris, draw_color)
            cv2.fillPoly(alpha_mask, tris, 1.0)

        u_alpha = cv2.UMat(alpha_mask)
        u_alpha = cv2.GaussianBlur(u_alpha, (3, 3), 0)
        a = u_alpha.get()[:, :, None]

        out = (overlay.astype(np.float32) * a + frame.astype(np.float32) * (1.0 - a))
        return out.astype(np.uint8)
