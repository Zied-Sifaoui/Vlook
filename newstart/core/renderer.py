"""
Drop-in replacement for core/renderer.py
Key fixes:
  1. Z is now centered around 0 (not floored to 0) so hair sits AT the face plane
  2. _build() accepts offset_z and applies it directly to mesh vertices
  3. offset_z in render() is applied to the mesh, NOT to tvec
"""

import cv2
import numpy as np
from typing import Optional, Tuple

try:
    import moderngl
    _HAS_MGL = True
except ImportError:
    _HAS_MGL = False
    print("[AR] moderngl not found — CPU fallback. Run: pip install moderngl")

_VERT = """
#version 330 core
in vec3 in_pos;
in vec3 in_nor;
out float v_bright;
out float v_depth;
uniform mat4 u_proj;
void main() {
    vec4 pos_gl = vec4(in_pos.x, -in_pos.y, -in_pos.z, 1.0);
    gl_Position = u_proj * pos_gl;
    v_depth = in_pos.z;
    vec3 L    = normalize(vec3(-0.3, -0.6, -1.0));
    vec3 n_gl = normalize(vec3(in_nor.x, -in_nor.y, -in_nor.z));
    float diff = max(dot(n_gl, -L), 0.0);
    v_bright   = clamp(diff * 0.7 + 0.4, 0.2, 1.0);
}
"""

_FRAG = """
#version 330 core
in  float v_bright;
in  float v_depth;
out vec4  f_color;
uniform vec3      u_color;
uniform float     u_opacity;
uniform sampler2D u_face_depth;
uniform vec2      u_resolution;
uniform float     u_occ_margin;
void main() {
    vec2  uv     = gl_FragCoord.xy / u_resolution;
    float face_z = texture(u_face_depth, uv).r;
    if (face_z > 0.0 && v_depth > face_z + u_occ_margin)
        discard;
    f_color = vec4(u_color * v_bright, u_opacity);
}
"""


def _make_proj(K, W, H, near=1.0, far=200000.0):
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])
    m = np.zeros((4,4), np.float32)
    m[0,0] =  2.0*fx/W
    m[1,1] =  2.0*fy/H
    m[0,2] =  2.0*cx/W - 1.0
    m[1,2] =  2.0*cy/H - 1.0
    m[2,2] = -(far+near)/(far-near)
    m[2,3] = -2.0*far*near/(far-near)
    m[3,2] = -1.0
    return m


class ARRenderer:

    def __init__(self, obj_path: str, normalize_span: float = 1.0):
        verts_raw, self.faces = self._load_obj(obj_path)

        # Center the mesh on all 3 axes
        centered = verts_raw - verts_raw.mean(axis=0)

        # Scale so X-span = normalize_span
        x_span = centered[:, 0].max() - centered[:, 0].min()
        if x_span > 1e-6:
            centered *= (normalize_span / x_span)

        # Mesh straddles Z=0 (face plane); rvec*=-1 in server.py handles axis alignment
        self._verts_norm = centered.astype(np.float32)

        # Per-face normals
        v0 = self._verts_norm[self.faces[:, 0]]
        v1 = self._verts_norm[self.faces[:, 1]]
        v2 = self._verts_norm[self.faces[:, 2]]
        n  = np.cross(v1 - v0, v2 - v0).astype(np.float32)
        self._face_normals = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)

        self._light = np.array([-0.3, -0.6, -1.0], np.float32)
        self._light /= np.linalg.norm(self._light)

        self._ctx = self._prog = self._fbo = None
        self._col_tex = self._dep_rb = self._fd_tex = None
        self._W = self._H = 0

        z_min = self._verts_norm[:, 2].min()
        z_max = self._verts_norm[:, 2].max()
        print(f"[AR] {obj_path}  verts={len(verts_raw)} faces={len(self.faces)}")
        print(f"[AR] mesh Z range: {z_min:.3f} -> {z_max:.3f}  (0 = face plane)")
        print(f"[AR] backend={'OpenGL/GPU' if _HAS_MGL else 'CPU'}")

    @staticmethod
    def _load_obj(path):
        verts, faces = [], []
        with open(path) as fh:
            for line in fh:
                tok = line.split()
                if not tok: continue
                if tok[0] == 'v':
                    verts.append((float(tok[1]), float(tok[2]), float(tok[3])))
                elif tok[0] == 'f':
                    idx = [int(t.split('/')[0])-1 for t in tok[1:]]
                    for i in range(1, len(idx)-1):
                        faces.append((idx[0], idx[i], idx[i+1]))
        return np.array(verts, np.float32), np.array(faces, np.int32)

    def _build(self, scale: float, ox: float, oy: float, oz: float) -> np.ndarray:
        v = (self._verts_norm * scale).copy()
        v[:, 0] += ox
        v[:, 1] += oy
        v[:, 2] += oz
        return v

    def _init_gl(self, W, H):
        if self._ctx is not None:
            for obj in (self._fd_tex, self._fbo, self._dep_rb, self._col_tex, self._ctx):
                try: obj.release()
                except: pass
        self._ctx     = moderngl.create_standalone_context()
        self._prog    = self._ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self._col_tex = self._ctx.texture((W,H), 4)
        self._dep_rb  = self._ctx.depth_renderbuffer((W,H))
        self._fbo     = self._ctx.framebuffer(
            color_attachments=[self._col_tex], depth_attachment=self._dep_rb)
        self._fd_tex  = self._ctx.texture((W,H), 1, dtype='f4')
        self._fd_tex.filter = moderngl.NEAREST, moderngl.NEAREST
        self._W, self._H = W, H
        print(f"[AR] OpenGL ready {W}x{H}  GPU={self._ctx.info.get('GL_RENDERER','?')}")

    def _make_vbo(self, verts_cam, normals_cam):
        pos = verts_cam[self.faces]
        nor = np.repeat(normals_cam[:,None,:], 3, axis=1)
        return np.concatenate([pos, nor], axis=2).astype(np.float32).reshape(-1, 6)

    def _render_gpu(self, frame, verts_cam, normals_cam, hair_color,
                    opacity, K, face_depth_map, occ_margin):
        H, W = frame.shape[:2]
        if self._ctx is None or self._W != W or self._H != H:
            self._init_gl(W, H)

        ctx  = self._ctx
        prog = self._prog

        if face_depth_map is not None:
            fd = face_depth_map.copy().astype(np.float32)
            fd[~np.isfinite(fd)] = 0.0
        else:
            fd = np.zeros((H,W), np.float32)
        self._fd_tex.write(fd.tobytes())

        vbo_data = self._make_vbo(verts_cam, normals_cam)
        vbo = ctx.buffer(vbo_data.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_pos', 'in_nor')])

        proj = _make_proj(K, W, H)
        prog['u_proj'].write(proj.astype(np.float32).tobytes())
        prog['u_color'].value      = tuple(float(c/255.0) for c in hair_color)
        prog['u_opacity'].value    = float(opacity)
        prog['u_resolution'].value = (float(W), float(H))
        prog['u_occ_margin'].value = float(occ_margin)
        self._fd_tex.use(location=0)
        prog['u_face_depth'].value = 0

        self._fbo.use()
        self._fbo.clear(0,0,0,0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.disable(moderngl.CULL_FACE)
        vao.render(moderngl.TRIANGLES)
        ctx.finish()

        raw  = self._fbo.read(components=4, dtype='f1')
        rgba = np.frombuffer(raw, dtype=np.uint8).reshape(H,W,4)
        rgba = np.ascontiguousarray(np.flipud(rgba))

        a      = rgba[:,:,3:4].astype(np.float32)/255.0
        result = rgba[:,:,:3].astype(np.float32)*a + frame.astype(np.float32)*(1-a)

        vao.release()
        vbo.release()
        return result.astype(np.uint8)

    def _render_cpu(self, frame, pts2d, avg_z, brightness,
                    hair_color, opacity, face_depth_map, occ_margin):
        from collections import defaultdict
        H,W = frame.shape[:2]
        f0,f1,f2 = self.faces[:,0],self.faces[:,1],self.faces[:,2]
        p0,p1,p2 = pts2d[f0],pts2d[f1],pts2d[f2]
        cross_z = ((p1[:,0]-p0[:,0])*(p2[:,1]-p0[:,1])
                  -(p1[:,1]-p0[:,1])*(p2[:,0]-p0[:,0]))
        margin  = max(H,W)*2
        max_c   = np.maximum(np.abs(p0).max(1),
                  np.maximum(np.abs(p1).max(1), np.abs(p2).max(1)))
        vis   = np.where((cross_z>0)&(max_c<margin))[0]
        order = vis[np.argsort(-avg_z[vis])]
        N_B = 16
        b_q = np.floor(brightness*N_B).clip(0,N_B).astype(np.int32)
        overlay = np.zeros((H,W,3), np.uint8)
        alpha   = np.zeros((H,W),   np.float32)
        buckets = defaultdict(list)
        for i in order:
            buckets[int(b_q[i])].append(np.array([p0[i],p1[i],p2[i]], np.int32))
        for bi, tris in buckets.items():
            bv = bi/N_B
            cv2.fillPoly(overlay, tris, tuple(int(c*bv) for c in hair_color))
            cv2.fillPoly(alpha,   tris, opacity)
        if face_depth_map is not None:
            hd = np.full((H,W), np.inf, np.float32)
            for i in order:
                tri = np.array([p0[i],p1[i],p2[i]], np.int32)
                m   = np.zeros((H,W), np.uint8)
                cv2.fillPoly(m,[tri],1)
                px = m==1
                hd[px] = np.minimum(hd[px], avg_z[i])
            behind = (hd > face_depth_map+occ_margin) & np.isfinite(hd)
            alpha[behind] = 0.0
        ab = cv2.GaussianBlur(cv2.UMat(alpha),(5,5),0).get()
        a  = ab[:,:,None]
        return (overlay.astype(np.float32)*a + frame.astype(np.float32)*(1-a)).astype(np.uint8)

    def render(self, frame, rvec, tvec, K,
               style=None,
               offset_x: float = 0.0,
                offset_y: float = -400.0,
               offset_z: float = 0.0,
               mesh_scale: float = 450.0,
               opacity: float = 0.92,
               face_depth_map=None,
               occ_margin: float = 15.0,
               **kwargs):

        hair_color = np.array(style['frame'] if style else [40,25,15], np.float32)

        R, _ = cv2.Rodrigues(rvec)
        tvec_flat = tvec.reshape(3, 1)

        verts_obj   = self._build(mesh_scale, offset_x, offset_y, offset_z)
        verts_cam   = (R @ verts_obj.T).T + tvec_flat.reshape(1, 3)
        normals_cam = (R @ self._face_normals.T).T

        if _HAS_MGL:
            return self._render_gpu(frame, verts_cam, normals_cam,
                                    hair_color, opacity, K,
                                    face_depth_map, occ_margin)

        dist  = np.zeros((4,1), np.float32)
        pts2d,_ = cv2.projectPoints(verts_obj, rvec, tvec_flat, K, dist)
        pts2d   = pts2d.reshape(-1,2).astype(np.float32)
        f0,f1,f2 = self.faces[:,0],self.faces[:,1],self.faces[:,2]
        avg_z  = (verts_cam[:,2][f0]+verts_cam[:,2][f1]+verts_cam[:,2][f2])/3.0
        bpf    = np.clip(normals_cam @ self._light * 0.7 + 0.4, 0.2, 1.0)
        bright = (bpf[f0]+bpf[f1]+bpf[f2])/3.0
        return self._render_cpu(frame, pts2d, avg_z, bright,
                                hair_color, opacity, face_depth_map, occ_margin)