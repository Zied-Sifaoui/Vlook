"""
ui/controls.py
Mesh Scale: slider 1-1000, value = mm directly (normalize_span=1.0 in renderer).
Offset X/Y/Z: slider 0-4000, real = val - 2000.
"""

import cv2
import numpy as np
import time
import os

# ── psutil ────────────────────────────────────────────────────────────────────
try:
    import psutil as _psutil
    _PROC     = _psutil.Process()
    _PROC.cpu_percent(interval=None)   # prime
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ── GPU monitoring — try pynvml first, then gputil ────────────────────────────
_GPU_BACKEND = None
_nvml_handle = None

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_BACKEND = "pynvml"
    print("[HUD] GPU monitor: pynvml OK")
except Exception as _e:
    print(f"[HUD] pynvml unavailable ({_e}), trying gputil...")

if _GPU_BACKEND is None:
    try:
        import GPUtil as _GPUtil
        if _GPUtil.getGPUs():
            _GPU_BACKEND = "gputil"
            print("[HUD] GPU monitor: gputil OK")
        else:
            print("[HUD] gputil found but no GPUs detected.")
    except Exception as _e:
        print(f"[HUD] gputil unavailable ({_e})")

if _GPU_BACKEND is None:
    print("[HUD] No GPU monitor available. pip install pynvml   (Nvidia)")


def _query_gpu():
    """Returns (util_pct, mem_used_mb, mem_total_mb, name) or Nones on error."""
    if _GPU_BACKEND == "pynvml":
        try:
            u  = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
            m  = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
            nm = pynvml.nvmlDeviceGetName(_nvml_handle)
            if isinstance(nm, bytes): nm = nm.decode()
            return float(u.gpu), m.used/1024**2, m.total/1024**2, nm
        except Exception:
            pass
    if _GPU_BACKEND == "gputil":
        try:
            gpus = _GPUtil.getGPUs()
            if gpus:
                g = gpus[0]
                return g.load*100.0, g.memoryUsed, g.memoryTotal, g.name
        except Exception:
            pass
    return None, None, None, None


# ── colour palette ────────────────────────────────────────────────────────────
C_PANEL   = (22,  24,  32)
C_ACCENT  = (0,  210, 170)
C_ALIGN   = (30, 160, 255)
C_WARN    = (30,  80, 220)
C_ORANGE  = (30, 140, 255)
C_PINK    = (180, 80, 200)
C_GREEN   = (40, 200, 100)
C_RED     = (50,  50, 220)
C_CYAN    = (200, 220,  20)
C_TEXT_HI = (235, 235, 240)
C_TEXT_LO = (100, 108, 128)
C_BORDER  = (50,   55,  72)
FONT      = cv2.FONT_HERSHEY_SIMPLEX

# ── slider definitions ────────────────────────────────────────────────────────
ALIGN_SLIDERS = [
    ("Offset X",   "OFFSET X", 2000, 4000, C_PINK),
    ("Offset Y",   "OFFSET Y", 2170, 4000, C_ORANGE),
    ("Offset Z",   "OFFSET Z", 2000, 4000, C_ALIGN),
    ("Mesh Scale", "MESH SCL",  450, 1000, C_GREEN),
]

PANEL_W  = 280
PANEL_X  = 14
PANEL_Y  = 14
CORNER_R = 12


class WindowManager:

    def __init__(self, win_name: str):
        self.win_name  = win_name
        self.presets   = [
            {"name": "Silver", "frame": (200, 200, 200), "lens": (55,  55,  55)},
            {"name": "Gold",   "frame": ( 30, 130, 200), "lens": (15,  75, 135)},
            {"name": "Black",  "frame": ( 15,  15,  15), "lens": ( 5,   5,   5)},
            {"name": "Rose",   "frame": (110,  85, 205), "lens": (55,  35, 115)},
            {"name": "Chrome", "frame": (185, 215, 235), "lens": (85, 115, 145)},
        ]
        self.color_idx  = 0
        self._fps_buf   = []
        self._last_ts   = 0.0
        self._hidden    = False
        self._show_align= False
        self._frame_no  = 0
        self._pending_ss= False
        self._ss_dir    = "screenshots"
        self._tb        = {s[0]: s[2] for s in ALIGN_SLIDERS}

        # perf cache
        self._render_ms  = 0.0
        self._cpu_pct    = 0.0
        self._ram_mb     = 0.0
        self._ram_tot_mb = 1.0
        self._gpu_util   = None   # float or None
        self._gpu_mem    = None
        self._gpu_tot    = None
        self._gpu_name   = ""
        self._perf_tick  = 0
        self._perf_every = 12     # sample every N frames

    # called by main loop each frame
    def set_render_ms(self, ms: float):
        self._render_ms = ms

    # ── system sampling (throttled) ───────────────────────────────
    def _sample_perf(self):
        self._perf_tick += 1
        if self._perf_tick % self._perf_every != 0:
            return

        if _HAS_PSUTIL:
            try:
                self._cpu_pct    = _PROC.cpu_percent(interval=None)
                self._ram_mb     = _PROC.memory_info().rss / 1024**2
                self._ram_tot_mb = _psutil.virtual_memory().total / 1024**2
            except Exception:
                pass

        gu, gm, gt, gn = _query_gpu()
        # Only update if we actually got data — don't clobber with None
        if gu is not None:
            self._gpu_util = gu
            self._gpu_mem  = gm
            self._gpu_tot  = gt
        if gn:
            self._gpu_name = gn[:26]

    # ── window / keyboard ─────────────────────────────────────────
    def create_interface(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, 1280, 720)
        for name, _, default, maximum, _ in ALIGN_SLIDERS:
            cv2.createTrackbar(name, self.win_name, default, maximum,
                               lambda x: None)

    def handle_input(self) -> bool:
        try:
            if cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except cv2.error:
            return False
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):  return False
        if key in (ord("c"), ord("C")):
            self.color_idx = (self.color_idx + 1) % len(self.presets)
        if key in (ord("r"), ord("R")):
            for name, _, default, _, _ in ALIGN_SLIDERS:
                cv2.setTrackbarPos(name, self.win_name, default)
        if key in (ord("h"), ord("H")): self._hidden     = not self._hidden
        if key in (ord("s"), ord("S")): self._pending_ss = True
        if key == 9:                    self._show_align  = not self._show_align
        for i, code in enumerate(ord(str(n)) for n in range(1, 6)):
            if key == code: self.color_idx = i % len(self.presets)
        return True

    def maybe_save_screenshot(self, frame):
        if not self._pending_ss: return
        os.makedirs(self._ss_dir, exist_ok=True)
        fn = os.path.join(self._ss_dir, f"vlook_{int(time.time())}.png")
        cv2.imwrite(fn, frame)
        print(f"[V-Look] Screenshot → {fn}")
        self._pending_ss = False

    # ── primitives ────────────────────────────────────────────────
    @staticmethod
    def _rrect(img, x, y, w, h, r, color, alpha=1.0, thick=-1):
        ov = img.copy()
        cv2.rectangle(ov,(x+r,y),(x+w-r,y+h),color,thick)
        cv2.rectangle(ov,(x,y+r),(x+w,y+h-r),color,thick)
        for cx,cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
            cv2.circle(ov,(cx,cy),r,color,thick)
        if alpha < 1.0: cv2.addWeighted(ov,alpha,img,1-alpha,0,img)
        else: img[:]=ov

    @staticmethod
    def _txt(img, s, x, y, sc, color, thick=1):
        cv2.putText(img,s,(x,y),FONT,sc,color,thick,cv2.LINE_AA)

    @staticmethod
    def _bar(img, x, y, w, h, val, maxv, fg):
        cv2.rectangle(img,(x,y),(x+w,y+h),C_BORDER,-1)
        fill = int(w*np.clip(val/max(maxv,1e-6),0,1))
        if fill<=0: return
        glow = tuple(int(c*0.28) for c in fg)
        cv2.rectangle(img,(x,y-2),(x+fill,y+h+2),glow,-1)
        cv2.rectangle(img,(x,y),(x+fill,y+h),fg,-1)
        cv2.line(img,(x+fill,y),(x+fill,y+h),C_TEXT_HI,1)

    @staticmethod
    def _brackets(img, face_found, fn):
        H,W=img.shape[:2]; cx,cy=W//2,H//2; bw,bh=155,115; arm,th=22,2
        color = C_ACCENT if face_found else (C_WARN if (fn//15)%2==0 else None)
        if color is None: return
        for bx,by,sx,sy in [(cx-bw,cy-bh,+1,+1),(cx+bw,cy-bh,-1,+1),
                             (cx-bw,cy+bh,+1,-1),(cx+bw,cy+bh,-1,-1)]:
            cv2.line(img,(bx,by),(bx+sx*arm,by),color,th)
            cv2.line(img,(bx,by),(bx,by+sy*arm),color,th)

    def _tick_fps(self) -> float:
        now = time.perf_counter()
        if self._last_ts:
            self._fps_buf.append(1.0/max(now-self._last_ts,1e-6))
            if len(self._fps_buf)>45: self._fps_buf.pop(0)
        self._last_ts = now
        return float(np.mean(self._fps_buf)) if self._fps_buf else 0.0

    def _sync(self):
        for name,*_ in ALIGN_SLIDERS:
            try: self._tb[name]=cv2.getTrackbarPos(name,self.win_name)
            except: pass

    def _bg(self, frame, px, py, pw, ph, accent):
        roi  = frame[py:py+ph, px:px+pw]
        blur = cv2.GaussianBlur(roi,(25,25),0)
        dark = np.full_like(roi,C_PANEL,dtype=np.uint8)
        frame[py:py+ph,px:px+pw]=cv2.addWeighted(blur,0.50,dark,0.50,0)
        self._rrect(frame,px,py,pw,ph,CORNER_R,accent,thick=1)
        self._rrect(frame,px+1,py+1,pw-2,ph-2,CORNER_R-1,(8,10,16),alpha=0.35,thick=1)

    def _header(self, frame, px, py, pw, fps, accent, label) -> int:
        ty = py+22
        cv2.circle(frame,(px+18,ty-5),5,accent,-1)
        cv2.circle(frame,(px+18,ty-5),7,accent, 1)
        self._txt(frame,label,px+32,ty,0.45,C_TEXT_HI)
        fc = C_ACCENT if fps>=28 else C_CYAN if fps>=18 else C_RED
        mw=30; mf=int(mw*min(fps/60.0,1.0))
        self._bar(frame,px+pw-mw-52,ty-9,mw,4,mf,mw,fc)
        self._txt(frame,f"{fps:4.1f}fps",px+pw-50,ty,0.37,fc)
        dv=ty+10; cv2.line(frame,(px+8,dv),(px+pw-8,dv),C_BORDER,1)
        return dv

    def _real(self, name) -> str:
        v=self._tb.get(name,0)
        if name in ("Offset X","Offset Y","Offset Z"): return f"{v-2000:+d}mm"
        if name=="Mesh Scale": return f"{v}mm"
        return str(v)

    # ── performance panel (bottom-left, always visible) ───────────
    def _draw_perf(self, frame, fps: float):
        H, W = frame.shape[:2]
        has_gpu = self._gpu_util is not None

        # rows: title / render | fps | cpu | ram | [gpu | vram]
        n_rows = 4 + (2 if has_gpu else 0)
        rh     = 17
        pw     = PANEL_W
        ph     = 10 + n_rows * rh + 4
        px     = PANEL_X
        py     = H - ph - 10

        roi  = frame[py:py+ph, px:px+pw]
        blur = cv2.GaussianBlur(roi,(15,15),0)
        dark = np.full_like(roi,C_PANEL,dtype=np.uint8)
        frame[py:py+ph,px:px+pw]=cv2.addWeighted(blur,0.45,dark,0.55,0)
        self._rrect(frame,px,py,pw,ph,8,C_BORDER,thick=1)

        bx    = px+8
        bar_w = pw-95
        ry    = py+12

        # row 0: section label + render time
        self._txt(frame,"PERFORMANCE",bx,ry,0.30,C_TEXT_LO)
        rt_col = C_GREEN if self._render_ms<16 else C_ORANGE if self._render_ms<33 else C_RED
        self._txt(frame,f"render {self._render_ms:5.1f}ms",
                  px+pw-92,ry,0.30,rt_col)
        ry+=rh

        # FPS
        fps_c = C_GREEN if fps>=28 else C_CYAN if fps>=18 else C_RED
        self._txt(frame,"FPS",bx,ry,0.30,C_TEXT_LO)
        self._bar(frame,bx+32,ry-8,bar_w,5,fps,60,fps_c)
        self._txt(frame,f"{fps:5.1f}",bx+32+bar_w+4,ry,0.30,fps_c)
        ry+=rh

        # CPU
        cpu   = min(self._cpu_pct,100.0)
        cpu_c = C_GREEN if cpu<50 else C_ORANGE if cpu<80 else C_RED
        self._txt(frame,"CPU",bx,ry,0.30,C_TEXT_LO)
        self._bar(frame,bx+32,ry-8,bar_w,5,cpu,100,cpu_c)
        self._txt(frame,f"{cpu:5.1f}%",bx+32+bar_w+4,ry,0.30,cpu_c)
        ry+=rh

        # RAM
        ram   = self._ram_mb
        rtot  = max(self._ram_tot_mb,1.0)
        ram_c = C_GREEN if ram<rtot*0.4 else C_ORANGE if ram<rtot*0.75 else C_RED
        self._txt(frame,"RAM",bx,ry,0.30,C_TEXT_LO)
        self._bar(frame,bx+32,ry-8,bar_w,5,ram,rtot,ram_c)
        rs = f"{ram:.0f}MB" if ram<1024 else f"{ram/1024:.1f}GB"
        self._txt(frame,rs,bx+32+bar_w+4,ry,0.30,ram_c)
        ry+=rh

        if has_gpu:
            # GPU util
            gu    = self._gpu_util
            gu_c  = C_GREEN if gu<50 else C_ORANGE if gu<80 else C_RED
            self._txt(frame,"GPU",bx,ry,0.30,C_TEXT_LO)
            self._bar(frame,bx+32,ry-8,bar_w,5,gu,100,gu_c)
            self._txt(frame,f"{gu:5.1f}%",bx+32+bar_w+4,ry,0.30,gu_c)
            ry+=rh

            # VRAM
            gm    = self._gpu_mem or 0
            gt    = self._gpu_tot or max(gm,1)
            gm_c  = C_GREEN if gm<gt*0.5 else C_ORANGE if gm<gt*0.8 else C_RED
            self._txt(frame,"VRAM",bx,ry,0.30,C_TEXT_LO)
            self._bar(frame,bx+32,ry-8,bar_w,5,gm,gt,gm_c)
            gms = f"{gm:.0f}MB" if gm<1024 else f"{gm/1024:.1f}GB"
            self._txt(frame,gms,bx+32+bar_w+4,ry,0.30,gm_c)

        # GPU name / status hint at very bottom
        if has_gpu and self._gpu_name:
            self._txt(frame,self._gpu_name,px+4,py+ph-3,0.24,C_TEXT_LO)
        elif not has_gpu:
            hint = "pip install pynvml  for GPU stats"
            self._txt(frame,hint,px+4,py+ph-3,0.24,C_TEXT_LO)

    # ── main HUD entry point ──────────────────────────────────────
    def draw_hud(self, frame, face_found=False):
        self._frame_no += 1
        fps = self._tick_fps()
        self._sync()
        self._sample_perf()
        H, W = frame.shape[:2]
        self._brackets(frame, face_found, self._frame_no)

        if self._hidden:
            self._txt(frame,"[H] show  [TAB] align",W-170,H-10,0.34,C_TEXT_LO)
            self._draw_perf(frame, fps)
            return frame

        px,py = PANEL_X,PANEL_Y; pw=PANEL_W; rh=26

        if not self._show_align:
            ph = 44+rh+50
            self._bg(frame,px,py,pw,ph,C_ACCENT)
            dv = self._header(frame,px,py,pw,fps,C_ACCENT,"V-LOOK  AR")
            sy = dv+20
            self._txt(frame,"STYLE",px+10,sy,0.35,C_TEXT_LO)
            for i,p in enumerate(self.presets):
                rx,ry2=px+65+i*22,sy-5
                cv2.circle(frame,(rx,ry2),7 if i==self.color_idx else 5,p["frame"],-1)
                if i==self.color_idx: cv2.circle(frame,(rx,ry2),9,C_ACCENT,1)
            self._txt(frame,self.presets[self.color_idx]["name"],px+pw-60,sy,0.37,C_TEXT_HI)
            cy2=sy+rh
            cv2.line(frame,(px+8,cy2-8),(px+pw-8,cy2-8),C_BORDER,1)
            phase=(self._frame_no%30)/30.0
            pr=int(4+2*abs(np.sin(phase*np.pi))) if face_found else 4
            sc=C_ACCENT if face_found else C_WARN
            cv2.circle(frame,(px+18,cy2+2),pr,sc,-1)
            cv2.circle(frame,(px+18,cy2+2),pr+2,sc,1)
            self._txt(frame,"FACE  LOCKED" if face_found else "SCANNING...",
                      px+32,cy2+6,0.40,sc)
            ly=cy2+28
            cv2.line(frame,(px+8,ly-8),(px+pw-8,ly-8),C_BORDER,1)
            for i,(k,d) in enumerate([("[C]","style"),("[R]","reset"),
                                       ("[H]","hide"), ("[TAB]","align")]):
                hx=px+6+i*(pw//4)
                self._txt(frame,k,hx,   ly+6,0.30,C_ACCENT)
                self._txt(frame,d,hx+28,ly+6,0.27,C_TEXT_LO)
        else:
            ph=44+len(ALIGN_SLIDERS)*rh+70
            self._bg(frame,px,py,pw,ph,C_ALIGN)
            dv=self._header(frame,px,py,pw,fps,C_ALIGN,"ALIGNMENT TUNE")
            cy2=dv+20; bx2=px+70; bw2=pw-130
            for name,label,_,maxv,color in ALIGN_SLIDERS:
                val=self._tb[name]
                self._txt(frame,label,px+8,cy2,0.33,C_TEXT_LO)
                self._bar(frame,bx2,cy2-9,bw2,5,val,maxv,color)
                self._txt(frame,self._real(name),px+pw-52,cy2,0.33,C_TEXT_HI)
                cy2+=rh
            cv2.line(frame,(px+8,cy2+2),(px+pw-8,cy2+2),C_BORDER,1)
            cy2+=14
            for line in ["X  + right / - left","Y  + up / - down",
                         "Z  + fwd / - back","MESH = hair width mm"]:
                self._txt(frame,line,px+10,cy2,0.29,C_TEXT_LO); cy2+=14
            cv2.line(frame,(px+8,cy2+2),(px+pw-8,cy2+2),C_BORDER,1)
            cy2+=14
            for i,(k,d) in enumerate([("[R]","reset"),("[TAB]","back"),("[S]","snap")]):
                hx=px+6+i*(pw//3)
                self._txt(frame,k,hx,   cy2,0.30,C_ALIGN)
                self._txt(frame,d,hx+30,cy2,0.27,C_TEXT_LO)

        self._draw_perf(frame, fps)
        self._txt(frame,"[Q] quit",W-68,H-10,0.32,C_TEXT_LO)
        return frame

    # ── accessors ─────────────────────────────────────────────────
    def get_style(self)      -> dict:  return self.presets[self.color_idx]
    def get_offset_x(self)   -> float: return float(self._tb.get("Offset X",2000)-2000)
    def get_offset_y(self)   -> float: return float(self._tb.get("Offset Y",2170)-2000)
    def get_offset_z(self)   -> float: return float(self._tb.get("Offset Z",2000)-2000)
    def get_mesh_scale(self) -> float: return float(max(1,self._tb.get("Mesh Scale",450)))