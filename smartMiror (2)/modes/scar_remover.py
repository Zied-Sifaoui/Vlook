"""
modes/scar_remover.py — Adaptive HSV + LAB scar detection with optional U-Net
"""

import cv2
import numpy as np
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

LIPS_IDX  = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,
              14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,
              312,13,82,81,42,183,78]
LEYE_IDX  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
REYE_IDX  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
LBROW_IDX = [70,63,105,66,107,55,65,52,53,46]
RBROW_IDX = [336,296,334,293,300,276,283,282,295,285]
NOSE_IDX  = [1,2,98,327,326,97,99,240,235,64,294,460,370,94,141]


def _exclusion_zone(mask, lms, idx, H, W, expand=10):
    pts = np.array([[int(lms[i].x*W), int(lms[i].y*H)]
                    for i in idx if i < len(lms)], np.int32)
    if len(pts) < 3:
        return
    hull = cv2.convexHull(pts)
    M    = cv2.moments(hull)
    if M["m00"] == 0:
        cv2.fillConvexPoly(mask, hull, 255); return
    cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
    ex = []
    for p in hull:
        px, py = p[0]; dx, dy = px - cx, py - cy
        nm = max(1.0, (dx**2 + dy**2)**0.5)
        ex.append([[px + int(expand*dx/nm), py + int(expand*dy/nm)]])
    cv2.fillConvexPoly(mask, np.array(ex, np.int32), 255)


class _SkinCalibrator:
    def __init__(self):
        self.a_mean = 138.0; self.l_mean = 160.0
        self.a_std  =   6.0; self.l_std  =  12.0
        self._n = 0

    def update(self, frame, fm, ex):
        self._n += 1
        if self._n % 30 != 0: return
        safe = cv2.bitwise_and(fm, cv2.bitwise_not(ex))
        if cv2.countNonZero(safe) < 100: return
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        px  = lab[safe > 0].astype(np.float32)
        self.a_mean = float(np.median(px[:, 1]))
        self.l_mean = float(np.median(px[:, 0]))
        self.a_std  = float(np.std(px[:, 1]))
        self.l_std  = float(np.std(px[:, 0]))

    def scar_mask(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        a   = lab[:, :, 1].astype(np.float32)
        l   = lab[:, :, 0].astype(np.float32)
        at  = self.a_mean + 2.2 * max(self.a_std, 3.0)
        lmn = self.l_mean - 3.0 * max(self.l_std, 5.0)
        lmx = self.l_mean + 2.0 * max(self.l_std, 5.0)
        return ((a > at) & (l > lmn) & (l < lmx)).astype(np.uint8) * 255


class ScarRemoverMode:

    def __init__(self, unet_weights: str = "unet_scar.pth"):
        self._calib   = _SkinCalibrator()
        self._smooth  = None
        self._hist    = deque(maxlen=5)
        self._model   = None
        self._dev     = None

        if _HAS_TORCH:
            try:
                self._model, self._dev = self._load_unet(unet_weights)
                print("[Scar] U-Net weights loaded")
            except Exception as e:
                print(f"[Scar] U-Net not available ({e}) — using HSV+LAB")
        else:
            print("[Scar] torch not installed — using HSV+LAB")

    @staticmethod
    def _load_unet(path):
        class DC(nn.Module):
            def __init__(self, i, o):
                super().__init__()
                self.b = nn.Sequential(
                    nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                    nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
            def forward(self, x): return self.b(x)

        class UNet(nn.Module):
            def __init__(self):
                super().__init__(); f=[64,128,256,512]; ic=3
                self.downs=nn.ModuleList(); self.ups=nn.ModuleList(); self.pool=nn.MaxPool2d(2,2)
                for fi in f: self.downs.append(DC(ic,fi)); ic=fi
                self.bot=DC(f[-1],f[-1]*2)
                for fi in reversed(f): self.ups.append(nn.ConvTranspose2d(fi*2,fi,2,2)); self.ups.append(DC(fi*2,fi))
                self.out=nn.Conv2d(f[0],1,1)
            def forward(self, x):
                sk=[]
                for d in self.downs: x=d(x); sk.append(x); x=self.pool(x)
                x=self.bot(x); sk=sk[::-1]
                for i in range(0,len(self.ups),2):
                    x=self.ups[i](x); s=sk[i//2]
                    if x.shape!=s.shape: x=F.interpolate(x,size=s.shape[2:])
                    x=torch.cat([s,x],1); x=self.ups[i+1](x)
                return self.out(x)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m   = UNet().to(dev)
        m.load_state_dict(torch.load(path, map_location=dev, weights_only=False))
        m.eval()
        return m, dev

    def _unet_mask(self, frame):
        h, w = frame.shape[:2]
        tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t   = tf(pil).unsqueeze(0).to(self._dev)
        with torch.no_grad():
            prob = torch.sigmoid(self._model(t)).squeeze().cpu().numpy()
        return cv2.resize((prob > 0.5).astype(np.uint8)*255,
                          (w, h), interpolation=cv2.INTER_NEAREST)

    def _face_masks(self, frame, lms):
        h2, w2 = frame.shape[:2]
        all_pts = np.array([[int(lm.x*w2), int(lm.y*h2)] for lm in lms], np.int32)
        fm = np.zeros((h2, w2), np.uint8)
        cv2.fillConvexPoly(fm, cv2.convexHull(all_pts), 255)
        ex = np.zeros((h2, w2), np.uint8)
        for idx, exp in [(LIPS_IDX,14),(LEYE_IDX,10),(REYE_IDX,10),
                         (LBROW_IDX,6),(RBROW_IDX,6),(NOSE_IDX,6)]:
            _exclusion_zone(ex, lms, idx, h2, w2, exp)
        return fm, ex

    def process(self, frame, lms) -> np.ndarray:
        if lms is None:
            return frame.copy()

        h2, w2 = frame.shape[:2]
        fm, ex = self._face_masks(frame, lms)
        self._calib.update(frame, fm, ex)

        if self._model is not None:
            raw = self._unet_mask(frame)
            if cv2.countNonZero(ex) > 0:
                raw = cv2.bitwise_and(raw, cv2.bitwise_not(ex))
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            m1  = cv2.inRange(hsv, np.array([0,  50,20]), np.array([12, 255,210]))
            m2  = cv2.inRange(hsv, np.array([163,50,20]), np.array([180,255,210]))
            raw = cv2.bitwise_and(cv2.bitwise_or(m1, m2), self._calib.scar_mask(frame))
            raw = cv2.bitwise_and(raw, fm)
            raw = cv2.bitwise_and(raw, cv2.bitwise_not(ex))
            raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  np.ones((2,2), np.uint8))
            raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
            flt = np.zeros_like(raw)
            cnts, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                a = cv2.contourArea(cnt)
                if 40 < a < 5000:
                    if len(cnt) >= 5:
                        _, ax, _ = cv2.fitEllipse(cnt)
                        if max(ax)/(min(ax)+1e-5) > 2.2:
                            cv2.drawContours(flt, [cnt], -1, 255, -1)
                    elif a > 150:
                        cv2.drawContours(flt, [cnt], -1, 255, -1)
            raw = cv2.dilate(flt, np.ones((4,4), np.uint8), iterations=2)

        k   = np.ones((5,5), np.uint8)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
        raw = cv2.dilate(raw, k)
        rf  = raw.astype(np.float32) / 255.0
        self._smooth = rf if self._smooth is None else 0.4*rf + 0.6*self._smooth
        sb  = (self._smooth > 0.35).astype(np.uint8) * 255
        self._hist.append(sb)

        if len(self._hist) >= 3:
            st = np.stack(list(self._hist), 0).astype(np.float32) / 255.0
            stable = (st.sum(0) >= 3).astype(np.uint8) * 255
        else:
            stable = np.zeros((h2, w2), np.uint8)

        if cv2.countNonZero(stable) > 0:
            return cv2.inpaint(frame, stable, 5, cv2.INPAINT_TELEA)
        return frame.copy()