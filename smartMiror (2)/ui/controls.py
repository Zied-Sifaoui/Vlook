"""
ui/controls.py — WindowManager: title bar, perf panel, alignment sliders, style presets
✅ Fixed: Replaced unreliable arrow keys with A/D for mode switching
=========================================================================================
🎮 KEY BINDINGS:
  Mode Switch:  A / D          (← previous / → next mode)
  Quit:         Q or ESC
  Style Cycle:  C or 1-5       (cycle presets / direct select)
  Reset Sliders:R
  Toggle Panel: H
  Toggle Align: TAB
  Screenshot:   S
  Hair AR Fine Tune (when align panel open):
    Offset X:   ← / → arrows   (or slider)
    Offset Y:   ↑ / ↓ arrows   (or slider)
    Offset Z:   W / S keys     (or slider)
    Mesh Scale: Mouse wheel    (or slider)
=========================================================================================
Mesh Scale: slider 1-1000, value = mm directly (normalize_span=1.0 in renderer).
Offset X/Y/Z: slider 0-4000, real = val - 2000  (±2000 mm range).
"""

import cv2
import numpy as np
import time
import os

# ── optional perf libs ────────────────────────────────────────────────────────
try:
    import psutil as _psutil
    _PROC = _psutil.Process()
    _PROC.cpu_percent(interval=None)
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

_nvml_h      = None
_HAS_NVML    = False
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_h   = pynvml.nvmlDeviceGetHandleByIndex(0)
    _HAS_NVML = True
    print("[HUD] GPU monitor: pynvml OK")
except Exception as e:
    print(f"[HUD] pynvml unavailable ({e}). pip install pynvml for GPU stats.")


def _query_gpu():
    if not _HAS_NVML:
        return None, None, None, None
    try:
        u  = pynvml.nvmlDeviceGetUtilizationRates(_nvml_h)
        m  = pynvml.nvmlDeviceGetMemoryInfo(_nvml_h)
        nm = pynvml.nvmlDeviceGetName(_nvml_h)
        if isinstance(nm, bytes): nm = nm.decode()
        return float(u.gpu), m.used/1024**2, m.total/1024**2, nm
    except Exception:
        return None, None, None, None


# ── palette ───────────────────────────────────────────────────────────────────
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
C_DIM     = (100, 108, 128)  # Added missing definition

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Mode names + accent colours (must match main.py MODES list order)
MODES = ["Hair AR", "Brow Shaper", "Scar Remover", "Mouth Swap"]
MODE_COLORS = [
    (0,  210, 170),   # teal
    (180, 80, 200),   # pink
    (30, 160, 255),   # blue
    (30, 140, 255),   # orange
]

# ── Hair AR alignment sliders ─────────────────────────────────────────────────
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
TITLE_H  = 38     # pixels reserved at top for the title bar


class WindowManager:

    def __init__(self, win_name: str):
        self.win_name   = win_name
        self.presets    = [
            {"name": "Silver", "frame": (200,200,200), "lens": (55, 55, 55)},
            {"name": "Gold",   "frame": ( 30,130,200), "lens": (15, 75,135)},
            {"name": "Black",  "frame": ( 15, 15, 15), "lens": ( 5,  5,  5)},
            {"name": "Rose",   "frame": (110, 85,205), "lens": (55, 35,115)},
            {"name": "Chrome", "frame": (185,215,235), "lens": (85,115,145)},
        ]
        self.color_idx      = 0
        self._fps_buf       = []
        self._last_ts       = 0.0
        self._hidden        = False
        self._show_align    = False
        self._frame_no      = 0
        self._pending_ss    = False
        self._ss_dir        = "screenshots"
        self._tb            = {s[0]: s[2] for s in ALIGN_SLIDERS}

        # perf cache
        self._render_ms  = 0.0
        self._cpu        = 0.0
        self._ram        = 0.0
        self._ram_tot    = 1.0
        self._gpu_util   = None
        self._gpu_mem    = None
        self._gpu_tot    = None
        self._gpu_name   = ""
        self._ptick      = 0
        self._pevery     = 15

    # ── called by main loop ───────────────────────────────────────
    def set_render_ms(self, ms: float):
        self._render_ms = ms

    # ── window setup ─────────────────────────────────────────────
    def create_interface(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, 1280, 720)
        for name, _, default, maximum, _ in ALIGN_SLIDERS:
            cv2.createTrackbar(name, self.win_name, default, maximum,
                               lambda x: None)

    # ── input ─────────────────────────────────────────────────────
    def handle_input(self) -> tuple:
        """
        Returns (running, key, mode_delta).
        mode_delta: -1 / 0 / +1 for ← / none / → mode switch.
        
        ✅ KEY BINDINGS:
          A / ,  → previous mode
          D / .  → next mode
          Q / ESC → quit
        """
        try:
            if cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1:
                return False, -1, 0
        except cv2.error:
            return False, -1, 0

        key = cv2.waitKey(1) & 0xFF
        mode_delta = 0

        # ── Quit ────────────────────────────────────────────────
        if key in (27, ord('q'), ord('Q')):
            return False, key, 0
            
        # ── Mode Switching: A/D keys (reliable) + comma/period ──
        # ✅ Primary: A (prev), D (next) - WASD style
        if key in (ord('a'), ord('A'), ord(',')):
            mode_delta = -1
        if key in (ord('d'), ord('D'), ord('.')):
            mode_delta = 1
            
        # ── Fallback: Arrow keys (platform-dependent, kept for compatibility) ──
        # Arrow key codes: LEFT=81 or 2, RIGHT=83 or 3 (varies by OS/terminal)
        if key in (81, 2, 242):   # LEFT arrow variants
            mode_delta = -1
        if key in (83, 3, 243):  # RIGHT arrow variants
            mode_delta = 1

        # ── Style & UI Controls ─────────────────────────────────
        if key in (ord('c'), ord('C')):
            self.color_idx = (self.color_idx + 1) % len(self.presets)
        if key in (ord('r'), ord('R')):
            for name, _, default, _, _ in ALIGN_SLIDERS:
                cv2.setTrackbarPos(name, self.win_name, default)
        if key in (ord('h'), ord('H')): 
            self._hidden = not self._hidden
        if key in (ord('s'), ord('S')) and not self._show_align:
            self._pending_ss = True
        if key == 9:  # TAB
            self._show_align = not self._show_align
            
        # Direct preset selection: keys 1-5
        for i, code in enumerate(ord(str(n)) for n in range(1, 6)):
            if key == code: 
                self.color_idx = i % len(self.presets)

        return True, key, mode_delta

    def maybe_save_screenshot(self, frame):
        if not self._pending_ss: 
            return
        os.makedirs(self._ss_dir, exist_ok=True)
        fn = os.path.join(self._ss_dir, f"vlook_{int(time.time())}.png")
        cv2.imwrite(fn, frame)
        print(f"[V-Look] Screenshot → {fn}")
        self._pending_ss = False

    # ── perf sampling ─────────────────────────────────────────────
    def _sample_perf(self):
        self._ptick += 1
        if self._ptick % self._pevery != 0:
            return
        if _HAS_PSUTIL:
            try:
                self._cpu     = _PROC.cpu_percent(interval=None)
                self._ram     = _PROC.memory_info().rss / 1024**2
                self._ram_tot = _psutil.virtual_memory().total / 1024**2
            except Exception: 
                pass
        gu, gm, gt, gn = _query_gpu()
        if gu is not None:
            self._gpu_util = gu
            self._gpu_mem  = gm
            self._gpu_tot  = gt
        if gn:
            self._gpu_name = gn[:26]

    # ── primitives ────────────────────────────────────────────────
    @staticmethod
    def _rrect(img, x, y, w, h, r, color, alpha=1.0, thick=-1):
        ov = img.copy()
        cv2.rectangle(ov, (x+r,y),   (x+w-r,y+h),   color, thick)
        cv2.rectangle(ov, (x,  y+r), (x+w,  y+h-r), color, thick)
        for cx, cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
            cv2.circle(ov, (cx,cy), r, color, thick)
        if alpha < 1.0: 
            cv2.addWeighted(ov,alpha,img,1-alpha,0,img)
        else: 
            img[:] = ov

    @staticmethod
    def _txt(img, s, x, y, sc, color, thick=1):
        cv2.putText(img, s, (x,y), FONT, sc, color, thick, cv2.LINE_AA)

    @staticmethod
    def _bar(img, x, y, w, h, val, maxv, fg):
        cv2.rectangle(img,(x,y),(x+w,y+h),C_BORDER,-1)
        fill = int(w * np.clip(val/max(maxv,1e-6),0,1))
        if fill <= 0: 
            return
        glow = tuple(int(c*0.28) for c in fg)
        cv2.rectangle(img,(x,y-2),(x+fill,y+h+2),glow,-1)
        cv2.rectangle(img,(x,y),  (x+fill,y+h),  fg,  -1)
        cv2.line(img,(x+fill,y),(x+fill,y+h),C_TEXT_HI,1)

    @staticmethod
    def _brackets(img, face_found, fn, offset_y=TITLE_H):
        H, W = img.shape[:2]
        cx, cy = W//2, (H+offset_y)//2
        bw, bh = 155, 115; arm, th = 22, 2
        if face_found:                  
            color = C_ACCENT
        elif (fn//15) % 2 == 0:        
            color = C_WARN
        else: 
            return
        for bx, by, sx, sy in [(cx-bw,cy-bh,+1,+1),(cx+bw,cy-bh,-1,+1),
                                (cx-bw,cy+bh,+1,-1),(cx+bw,cy+bh,-1,-1)]:
            cv2.line(img,(bx,by),(bx+sx*arm,by),color,th)
            cv2.line(img,(bx,by),(bx,by+sy*arm),color,th)

    def _tick_fps(self) -> float:
        now = time.perf_counter()
        if self._last_ts:
            self._fps_buf.append(1.0/max(now-self._last_ts,1e-6))
            if len(self._fps_buf) > 45: 
                self._fps_buf.pop(0)
        self._last_ts = now
        return float(np.mean(self._fps_buf)) if self._fps_buf else 0.0

    def _sync(self):
        for name, *_ in ALIGN_SLIDERS:
            try: 
                self._tb[name] = cv2.getTrackbarPos(name, self.win_name)
            except: 
                pass

    def _bg(self, frame, px, py, pw, ph, accent):
        roi  = frame[py:py+ph, px:px+pw]
        blur = cv2.GaussianBlur(roi,(25,25),0)
        dark = np.full_like(roi, C_PANEL, dtype=np.uint8)
        frame[py:py+ph, px:px+pw] = cv2.addWeighted(blur,0.50,dark,0.50,0)
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
        v = self._tb.get(name, 0)
        if name in ("Offset X","Offset Y","Offset Z"): 
            return f"{v-2000:+d}mm"
        if name == "Mesh Scale": 
            return f"{v}mm"
        return str(v)

    # ── title bar (full width, top) ───────────────────────────────
    def _draw_title_bar(self, frame, mode_idx: int, fps: float):
        H, W = frame.shape[:2]
        BAR  = TITLE_H
        accent = MODE_COLORS[mode_idx]

        # frosted background
        roi  = frame[0:BAR, 0:W]
        dark = np.full_like(roi, (14,16,22), np.uint8)
        frame[0:BAR, 0:W] = cv2.addWeighted(roi,0.20,dark,0.80,0)

        # left accent stripe
        cv2.rectangle(frame,(0,0),(4,BAR),accent,-1)

        # mode name
        name = MODES[mode_idx]
        cv2.putText(frame, name, (14,25), FONT, 0.62, accent, 1, cv2.LINE_AA)

        # mode dots
        dot_x = 18 + int(cv2.getTextSize(name,FONT,0.62,1)[0][0]) + 12
        for i, c in enumerate(MODE_COLORS):
            r = 5 if i==mode_idx else 3
            cv2.circle(frame,(dot_x+i*16,19),r,c,-1)
            if i==mode_idx: 
                cv2.circle(frame,(dot_x+i*16,19),r+2,c,1)
        
        # ✅ Updated hint text: show A/D instead of arrows
        hint_x = dot_x + len(MODES)*16 + 8
        self._txt(frame,"A/D mode", hint_x, 23, 0.30, C_DIM)

        # right side: perf stats
        rx = W-320; ry = 5; bw2 = 52; bh2 = 4

        def _stat(label,val,maxv,unit,col,px):
            self._txt(frame,label,px,ry+10,0.27,C_TEXT_LO)
            self._bar(frame,px+22,ry+3,bw2,bh2,val,maxv,col)
            s = f"{val:.0f}{unit}" if val>=10 else f"{val:.1f}{unit}"
            self._txt(frame,s,px+22+bw2+3,ry+10,0.26,col)
            return px+22+bw2+28

        fps_c = C_GREEN if fps>=28 else C_CYAN if fps>=18 else C_RED
        cpu_c = C_GREEN if self._cpu<50 else C_WARN if self._cpu<80 else C_RED
        ram_c = C_GREEN if self._ram<self._ram_tot*0.5 else C_WARN if self._ram<self._ram_tot*0.75 else C_RED

        px2 = rx
        px2 = _stat("FPS", fps,          60,          "", fps_c, px2)
        px2 = _stat("CPU", self._cpu,    100,         "%", cpu_c, px2)
        px2 = _stat("RAM", self._ram,    self._ram_tot,"M", ram_c, px2)
        if self._gpu_util is not None:
            gu_c = C_GREEN if self._gpu_util<50 else C_WARN if self._gpu_util<80 else C_RED
            px2  = _stat("GPU", self._gpu_util, 100, "%", gu_c, px2)
        if self._gpu_mem is not None:
            gm_c = C_GREEN if self._gpu_mem<(self._gpu_tot or 1)*0.5 else C_WARN if self._gpu_mem<(self._gpu_tot or 1)*0.8 else C_RED
            _stat("VRM", self._gpu_mem, self._gpu_tot or self._gpu_mem, "M", gm_c, px2)

        # render ms
        rms_c = C_GREEN if self._render_ms<16 else C_WARN if self._render_ms<33 else C_RED
        self._txt(frame,f"render {self._render_ms:.1f}ms",W-96,BAR-4,0.27,rms_c)

        # separator
        cv2.line(frame,(0,BAR),(W,BAR),tuple(int(c*0.4) for c in accent),1)

    # ── side panel (Hair AR controls) ────────────────────────────
    def _draw_side_panel(self, frame, fps: float, face_found: bool):
        H, W  = frame.shape[:2]
        px    = PANEL_X
        py    = PANEL_Y + TITLE_H
        pw    = PANEL_W
        row_h = 26

        if not self._show_align:
            ph = 44 + row_h + 50
            self._bg(frame,px,py,pw,ph,C_ACCENT)
            dv = self._header(frame,px,py,pw,fps,C_ACCENT,"HAIR AR")
            sy = dv+20
            self._txt(frame,"STYLE",px+10,sy,0.35,C_TEXT_LO)
            for i, p in enumerate(self.presets):
                rx2,ry2=px+65+i*22,sy-5
                cv2.circle(frame,(rx2,ry2),7 if i==self.color_idx else 5,p["frame"],-1)
                if i==self.color_idx: 
                    cv2.circle(frame,(rx2,ry2),9,C_ACCENT,1)
            self._txt(frame,self.presets[self.color_idx]["name"],px+pw-60,sy,0.37,C_TEXT_HI)
            cy2=sy+row_h
            cv2.line(frame,(px+8,cy2-8),(px+pw-8,cy2-8),C_BORDER,1)
            phase=(self._frame_no%30)/30.0
            pr=int(4+2*abs(np.sin(phase*np.pi))) if face_found else 4
            sc=C_ACCENT if face_found else C_WARN
            cv2.circle(frame,(px+18,cy2+2),pr,sc,-1)
            cv2.circle(frame,(px+18,cy2+2),pr+2,sc,1)
            self._txt(frame,"FACE  LOCKED" if face_found else "SCANNING...",px+32,cy2+6,0.40,sc)
            ly=cy2+28; cv2.line(frame,(px+8,ly-8),(px+pw-8,ly-8),C_BORDER,1)
            # ✅ Updated key hints to show A/D
            for i,(k,d) in enumerate([("[A/D]","mode"),("[C]","style"),("[R]","reset"),("[H]","hide"),("[TAB]","align")]):
                hx=px+6+i*(pw//5)
                self._txt(frame,k,hx,   ly+6,0.28,C_ACCENT)
                self._txt(frame,d,hx+28,ly+6,0.25,C_TEXT_LO)
        else:
            ph=44+len(ALIGN_SLIDERS)*row_h+70
            self._bg(frame,px,py,pw,ph,C_ALIGN)
            dv=self._header(frame,px,py,pw,fps,C_ALIGN,"ALIGNMENT TUNE")
            cy2=dv+20; bx2=px+70; bw2=pw-130
            for name,label,_,maxv,color in ALIGN_SLIDERS:
                val=self._tb[name]
                self._txt(frame,label,px+8,cy2,0.33,C_TEXT_LO)
                self._bar(frame,bx2,cy2-9,bw2,5,val,maxv,color)
                self._txt(frame,self._real(name),px+pw-52,cy2,0.33,C_TEXT_HI)
                cy2+=row_h
            cv2.line(frame,(px+8,cy2+2),(px+pw-8,cy2+2),C_BORDER,1)
            cy2+=14
            for line in ["X  + → / - ←","Y  + ↑ / - ↓",
                         "Z  + W / - S","MESH = hair width mm"]:
                self._txt(frame,line,px+10,cy2,0.29,C_TEXT_LO); cy2+=14
            cv2.line(frame,(px+8,cy2+2),(px+pw-8,cy2+2),C_BORDER,1)
            cy2+=14
            for i,(k,d) in enumerate([("[R]","reset"),("[TAB]","back"),("[S]","snap")]):
                hx=px+6+i*(pw//3)
                self._txt(frame,k,hx,   cy2,0.30,C_ALIGN)
                self._txt(frame,d,hx+30,cy2,0.27,C_TEXT_LO)

    # ── main draw_hud ─────────────────────────────────────────────
    def draw_hud(self, frame, mode_idx: int, face_found: bool = False):
        self._frame_no += 1
        fps = self._tick_fps()
        self._sync()
        self._sample_perf()

        H, W = frame.shape[:2]

        # always draw title bar
        self._draw_title_bar(frame, mode_idx, fps)

        # face brackets (shifted below title bar)
        self._brackets(frame, face_found, self._frame_no)

        if self._hidden:
            # ✅ Updated hidden mode hint
            self._txt(frame,"[H]show [A/D]mode [TAB]align",
                      W-220, H-10, 0.32, C_TEXT_LO)
            return frame

        # side panel only for Hair AR mode
        if mode_idx == 0:
            self._draw_side_panel(frame, fps, face_found)

        self._txt(frame,"[Q]quit", W-68, H-10, 0.32, C_TEXT_LO)
        return frame

    # ── accessors (Hair AR) ───────────────────────────────────────
    def get_style(self)      -> dict:  return self.presets[self.color_idx]
    def get_offset_x(self)   -> float: return float(self._tb.get("Offset X",  2000) - 2000)
    def get_offset_y(self)   -> float: return float(self._tb.get("Offset Y",  2170) - 2000)
    def get_offset_z(self)   -> float: return float(self._tb.get("Offset Z",  2000) - 2000)
    def get_mesh_scale(self) -> float: return float(max(1, self._tb.get("Mesh Scale", 450)))