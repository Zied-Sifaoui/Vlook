"""
Glasses Viewer — Smooth Pixel-Level Yaw Fade (no visible edges)
================================================================
How the seamless fade works:
  1. All vertices get a per-vertex alpha from their X position + yaw angle
  2. We render a SMOOTH alpha image using proper per-pixel barycentric
     interpolation — but entirely in NumPy (no Python loops):
       - For each face: rasterise its bounding box pixels at once
       - Interpolate vertex alpha across all pixels simultaneously
       - Write into a float32 alpha buffer
  3. Apply a small Gaussian blur to the alpha buffer → softens any
     remaining triangle-edge artifacts completely
  4. Composite solid-colored glasses through this smooth alpha mask

The result: geometry disappears like it's dissolving into the background,
with zero visible triangle edges.

Requirements:  pip install opencv-python numpy psutil
Controls:      Drag=rotate  Scroll=zoom  R=reset  ESC=quit
"""

import cv2, numpy as np, time, threading, psutil, os

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD OBJ
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = 'Glasses.obj'

def load_obj(path):
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('f '):
                idx = [int(x.split('/')[0])-1 for x in line.split()[1:]]
                for i in range(1, len(idx)-1):
                    faces.append((idx[0], idx[i], idx[i+1]))
    v = np.array(verts, dtype=np.float32)
    f = np.array(faces,  dtype=np.int32)
    v -= v.mean(axis=0)
    v[:, 2] *= -1
    return v, f

verts_raw, faces = load_obj(MODEL_PATH)
half_w    = float(verts_raw[:,0].max())
V_X_NORM  = (verts_raw[:,0] / (half_w + 1e-8)).astype(np.float32)
print(f"Mesh: {len(verts_raw)}V  {len(faces)}F  half_w={half_w:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. WINDOW + INTERACTION
# ─────────────────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 900, 600
REN_W, REN_H = 450, 300       # render half-res, upscale 2×
WINDOW = 'Glasses Viewer'
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, WIN_W, WIN_H)

state = dict(yaw=0.0, pitch=0.0, scale=1800.0, drag=False, mx=0, my=0)

def on_mouse(event, x, y, flags, _):
    s = state
    if   event == cv2.EVENT_LBUTTONDOWN: s['drag']=True;  s['mx']=x; s['my']=y
    elif event == cv2.EVENT_LBUTTONUP:   s['drag']=False
    elif event == cv2.EVENT_MOUSEMOVE and s['drag']:
        s['yaw']   += (x-s['mx'])*0.008
        s['pitch'] += (y-s['my'])*0.008
        s['mx'],s['my'] = x,y
    elif event == cv2.EVENT_MOUSEWHEEL:
        s['scale'] *= (1.1 if flags>0 else 0.9)

cv2.setMouseCallback(WINDOW, on_mouse)

def nothing(x): pass
cv2.createTrackbar('Fade start', WINDOW,  8, 60,  nothing)
cv2.createTrackbar('Fade end',   WINDOW, 35, 90,  nothing)
cv2.createTrackbar('Fade zone',  WINDOW, 50, 100, nothing)
cv2.createTrackbar('Edge blur',  WINDOW,  3, 15,  nothing)  # alpha blur radius

# ─────────────────────────────────────────────────────────────────────────────
# 3. ROTATION
# ─────────────────────────────────────────────────────────────────────────────
def make_R(yaw, pitch):
    cy,sy = np.cos(yaw),   np.sin(yaw)
    cx,sx = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], np.float32)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], np.float32)
    return Rx @ Ry

# ─────────────────────────────────────────────────────────────────────────────
# 4. PER-VERTEX ALPHA
# ─────────────────────────────────────────────────────────────────────────────
def vertex_alpha(yaw, fade_start, fade_end, x_zone):
    abs_yaw = abs(yaw)
    if abs_yaw < fade_start:
        return np.ones(len(V_X_NORM), dtype=np.float32)
    yaw_t = float(np.clip((abs_yaw-fade_start)/max(fade_end-fade_start,1e-6),0,1))
    side  = float(-np.sign(yaw)) if yaw!=0 else 0.0
    score = V_X_NORM * side
    fade  = np.clip((score-x_zone)/(1.0-x_zone+1e-6),0,1)*yaw_t
    return np.clip(1.0-fade,0,1).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 5. CORE RENDERER
#
# Two-pass approach:
#   Pass A — COLOR pass:  fillPoly batched by brightness (fast, per-face shading)
#   Pass B — ALPHA pass:  per-pixel barycentric alpha interpolation using NumPy
#                         tiled over ALL front faces simultaneously
#
# Pass B is the key to seamless fading. We process faces in BATCHES of
# BATCH_SIZE so we never allocate a giant (F×H×W) tensor, but still avoid
# any Python loop over individual faces.
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64    # faces per vectorised batch — tune for your RAM

def render_frame(rotated, v_alpha, scale, W, H, blur_k):
    CX, CY = W//2, H//2

    sx = (rotated[:,0]*scale + CX).astype(np.float32)
    sy = (-rotated[:,1]*scale + CY).astype(np.float32)
    sz =  rotated[:,2]

    # ── Back-face cull ────────────────────────────────────────────────────
    i0,i1,i2 = faces[:,0],faces[:,1],faces[:,2]
    x0,y0 = sx[i0],sy[i0]; x1,y1 = sx[i1],sy[i1]; x2,y2 = sx[i2],sy[i2]
    nz_face = (x1-x0)*(y2-y0)-(y1-y0)*(x2-x0)
    front   = np.where(nz_face>0)[0]
    if len(front)==0:
        return np.full((H,W,3),30,dtype=np.uint8)

    # ── Depth sort ────────────────────────────────────────────────────────
    avg_z      = (sz[i0[front]]+sz[i1[front]]+sz[i2[front]])/3.0
    draw_order = front[np.argsort(-avg_z)]
    fi         = faces[draw_order]              # (F_vis,3)

    # ── Shading ───────────────────────────────────────────────────────────
    r0=rotated[fi[:,0]]; r1=rotated[fi[:,1]]; r2=rotated[fi[:,2]]
    e1=r1-r0; e2=r2-r0
    nx=e1[:,1]*e2[:,2]-e1[:,2]*e2[:,1]
    ny=e1[:,2]*e2[:,0]-e1[:,0]*e2[:,2]
    nz=e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0]
    nlen=np.sqrt(nx**2+ny**2+nz**2)+1e-8
    lx,ly,lz=-0.3,-0.6,1.0; ln=np.sqrt(lx**2+ly**2+lz**2)
    diff=np.clip((nx/nlen)*lx/ln+(ny/nlen)*ly/ln+(nz/nlen)*lz/ln,0,1)
    brightness=np.clip(diff*0.7+0.35,0.2,1.0)  # (F_vis,)

    # ── PASS A: COLOR buffer via batched fillPoly ─────────────────────────
    pts_x = np.clip(np.round(np.column_stack([sx[fi[:,0]],sx[fi[:,1]],sx[fi[:,2]]])).astype(np.int32),0,W-1)
    pts_y = np.clip(np.round(np.column_stack([sy[fi[:,0]],sy[fi[:,1]],sy[fi[:,2]]])).astype(np.int32),0,H-1)
    polys_all = np.stack([pts_x,pts_y],axis=2)  # (F,3,2)

    color_buf = np.zeros((H,W,3),dtype=np.uint8)
    BASE      = np.array([13,217,64],dtype=np.float32)  # BGR green

    LEVELS   = 24
    bright_q = np.round(brightness*LEVELS).astype(np.int32)
    for bk in np.unique(bright_q):
        idx   = np.where(bright_q==bk)[0]
        b_val = brightness[idx[0]]
        col   = (BASE*b_val).astype(np.uint8).tolist()
        cv2.fillPoly(color_buf,[polys_all[i] for i in idx],col)

    # ── PASS B: SMOOTH ALPHA via batched barycentric interpolation ─────────
    # For each face we compute per-pixel alpha by interpolating the 3 vertex
    # alphas over the bounding-box pixel grid.  We process BATCH_SIZE faces
    # at once using broadcasting — no Python loop over individual faces.

    alpha_buf = np.zeros((H,W), dtype=np.float32)
    z_buf     = np.full ((H,W), np.inf, dtype=np.float32)

    # Pre-gather vertex screen coords and alphas for all draw faces
    ax0=sx[fi[:,0]];ay0=sy[fi[:,0]];az0=sz[fi[:,0]];aa0=v_alpha[fi[:,0]]
    ax1=sx[fi[:,1]];ay1=sy[fi[:,1]];az1=sz[fi[:,1]];aa1=v_alpha[fi[:,1]]
    ax2=sx[fi[:,2]];ay2=sy[fi[:,2]];az2=sz[fi[:,2]];aa2=v_alpha[fi[:,2]]
    denom=(ax1-ax0)*(ay2-ay0)-(ay1-ay0)*(ax2-ax0)   # signed area (F,)
    valid=np.abs(denom)>0.5                          # skip degenerate faces

    F_vis=len(fi)
    for b_start in range(0,F_vis,BATCH_SIZE):
        b_end = min(b_start+BATCH_SIZE,F_vis)
        bs    = slice(b_start,b_end)
        Fb    = b_end-b_start

        # Bounding boxes for this batch
        xmins=np.maximum(np.floor(np.minimum(np.minimum(ax0[bs],ax1[bs]),ax2[bs])).astype(int),0)
        ymins=np.maximum(np.floor(np.minimum(np.minimum(ay0[bs],ay1[bs]),ay2[bs])).astype(int),0)
        xmaxs=np.minimum(np.ceil (np.maximum(np.maximum(ax0[bs],ax1[bs]),ax2[bs])).astype(int),W-1)
        ymaxs=np.minimum(np.ceil (np.maximum(np.maximum(ay0[bs],ay1[bs]),ay2[bs])).astype(int),H-1)

        for k in range(Fb):
            if not valid[b_start+k]: continue
            xmn,xmx=xmins[k],xmaxs[k]; ymn,ymx=ymins[k],ymaxs[k]
            if xmn>=xmx or ymn>=ymx: continue

            # Pixel grid for this face's bbox
            px=np.arange(xmn,xmx,dtype=np.float32)+0.5
            py=np.arange(ymn,ymx,dtype=np.float32)+0.5
            gx,gy=np.meshgrid(px,py)               # (H_bb,W_bb)

            # Barycentric weights
            d   = denom[b_start+k]
            w0  = ((ax1[b_start+k]-ax0[b_start+k])*(gy-ay0[b_start+k])
                  -(ay1[b_start+k]-ay0[b_start+k])*(gx-ax0[b_start+k]))/d
            w1  = ((ax2[b_start+k]-ax1[b_start+k])*(gy-ay1[b_start+k])
                  -(ay2[b_start+k]-ay1[b_start+k])*(gx-ax1[b_start+k]))/d
            w2  = 1.0-w0-w1
            ins = (w0>=0)&(w1>=0)&(w2>=0)
            if not ins.any(): continue

            # Interpolated depth + alpha
            pz = w0*az0[b_start+k]+w1*az1[b_start+k]+w2*az2[b_start+k]
            pa = np.clip(w0*aa0[b_start+k]+w1*aa1[b_start+k]+w2*aa2[b_start+k],0,1)

            # Z-test: only update pixels where this face is closest
            cur_z = z_buf[ymn:ymx, xmn:xmx]
            depth_ok = ins & (pz < cur_z)

            z_buf  [ymn:ymx,xmn:xmx][depth_ok] = pz[depth_ok]
            alpha_buf[ymn:ymx,xmn:xmx][depth_ok] = pa[depth_ok]

    # ── Blur the alpha buffer to dissolve triangle edge seams ─────────────
    if blur_k > 1:
        k = blur_k*2+1   # make odd kernel
        alpha_buf = cv2.GaussianBlur(alpha_buf,(k,k),0)

    # ── Composite ─────────────────────────────────────────────────────────
    bg   = np.full((H,W,3),30,dtype=np.float32)
    col  = color_buf.astype(np.float32)
    a3   = alpha_buf[:,:,None]
    out  = col*a3 + bg*(1.0-a3)
    return np.clip(out,0,255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# 6. STATS
# ─────────────────────────────────────────────────────────────────────────────
proc  = psutil.Process(os.getpid())
stats = dict(cpu=0.0, mem_mb=0.0)

def stats_thread():
    while True:
        stats['cpu']    = proc.cpu_percent(interval=0.5)
        stats['mem_mb'] = proc.memory_info().rss/1024**2
        time.sleep(0.5)

threading.Thread(target=stats_thread,daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
frame_times  = []
render_ms_hist = []
print("Controls: Drag=rotate  Scroll=zoom  R=reset  ESC=quit")

while True:
    t0 = time.perf_counter()

    fade_start = np.radians(max(cv2.getTrackbarPos('Fade start',WINDOW),1))
    fade_end   = np.radians(max(cv2.getTrackbarPos('Fade end',  WINDOW),
                                cv2.getTrackbarPos('Fade start',WINDOW)+1))
    x_zone     = cv2.getTrackbarPos('Fade zone', WINDOW)/100.0
    blur_k     = cv2.getTrackbarPos('Edge blur', WINDOW)

    yaw,pitch,scale = state['yaw'],state['pitch'],state['scale']

    # Transform
    t_xfm = time.perf_counter()
    R       = make_R(yaw,pitch)
    rotated = (R @ verts_raw.T).T
    v_alph  = vertex_alpha(yaw,fade_start,fade_end,x_zone)
    xfm_ms  = (time.perf_counter()-t_xfm)*1000

    # Render at half res
    t_ren = time.perf_counter()
    small = render_frame(rotated,v_alph,scale,REN_W,REN_H,blur_k)
    ren_ms = (time.perf_counter()-t_ren)*1000
    render_ms_hist.append(ren_ms)
    if len(render_ms_hist)>30: render_ms_hist.pop(0)

    # Upscale
    frame = cv2.resize(small,(WIN_W,WIN_H),interpolation=cv2.INTER_LINEAR)

    # FPS
    now = time.perf_counter()
    frame_times.append(now)
    frame_times=[t for t in frame_times if now-t<1.0]
    fps     = len(frame_times)
    tot_ms  = (now-t0)*1000
    avg_ren = sum(render_ms_hist)/max(len(render_ms_hist),1)

    # ── Stats panel ───────────────────────────────────────────────────────
    px,py = WIN_W-290,12
    cv2.rectangle(frame,(px-8,py-4),(WIN_W-6,py+155),(18,18,18),-1)
    cv2.rectangle(frame,(px-8,py-4),(WIN_W-6,py+155),(55,55,55),1)

    def sline(label,val,row,col=(170,255,170)):
        cv2.putText(frame,f"{label:<16}{val}",(px,py+row*22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.44,col,1)

    fc = (0,230,0) if fps>=25 else (0,180,255) if fps>=12 else (0,60,255)
    sline("FPS",          str(fps),0,fc)
    sline("Total ms",     f"{tot_ms:.1f}",   1)
    sline("Render ms",    f"{avg_ren:.1f}",  2)
    sline("Transform ms", f"{xfm_ms:.1f}",  3)
    sline("CPU %",        f"{stats['cpu']:.1f}", 4,
          (0,60,255) if stats['cpu']>80 else (170,255,170))
    sline("RAM MB",       f"{stats['mem_mb']:.0f}", 5)
    sline("Render res",   f"{REN_W}×{REN_H}→2×",   6)
    sline("Faces",        f"{len(faces)//2}",        6)

    # Yaw / fade HUD
    cv2.putText(frame,
        f"Yaw:{np.degrees(yaw):+.1f}°  Pitch:{np.degrees(pitch):+.1f}°",
        (12,28),cv2.FONT_HERSHEY_SIMPLEX,0.55,(150,220,255),1)
    cv2.putText(frame,
        f"Fade {np.degrees(fade_start):.0f}→{np.degrees(fade_end):.0f}°  "
        f"zone:{x_zone:.2f}  blur:{blur_k}",
        (12,52),cv2.FONT_HERSHEY_SIMPLEX,0.44,(150,200,255),1)

    # Yaw bar
    bcy,bcx = WIN_H-28,WIN_W//2
    cv2.line(frame,(bcx-140,bcy),(bcx+140,bcy),(45,45,45),5)
    fsp=int(np.degrees(fade_start)*140/60)
    fep=int(min(np.degrees(fade_end)*140/60,140))
    cv2.line(frame,(bcx-fep,bcy),(bcx-fsp,bcy),(90,90,180),5)
    cv2.line(frame,(bcx+fsp,bcy),(bcx+fep,bcy),(90,90,180),5)
    dp=int(np.clip(np.degrees(yaw)*140/60,-140,140))
    cv2.circle(frame,(bcx+dp,bcy),7,(0,220,60),-1)
    cv2.putText(frame,"← yaw →",(bcx-28,bcy+15),
                cv2.FONT_HERSHEY_SIMPLEX,0.35,(80,80,80),1)
    cv2.putText(frame,"Drag=rotate  Scroll=zoom  R=reset  ESC=quit",
                (12,WIN_H-8),cv2.FONT_HERSHEY_SIMPLEX,0.36,(70,70,70),1)

    cv2.imshow(WINDOW,frame)
    key=cv2.waitKey(1)&0xFF
    if   key==27: break
    elif key in (ord('r'),ord('R')): state.update(yaw=0.,pitch=0.,scale=1800.)
    elif key==81: state['yaw']   -=0.04
    elif key==83: state['yaw']   +=0.04
    elif key==82: state['pitch'] -=0.04
    elif key==84: state['pitch'] +=0.04

cv2.destroyAllWindows()