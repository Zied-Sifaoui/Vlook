import os
import cv2
import numpy as np

MOUTH_IDS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
             324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

_SMOOTH = 0.35


class MouthCorrection:
    def __init__(self):
        self.offset_x = 0
        self.offset_y = 0
        self.scale_delta = 0.0
        self.MOVE_STEP = 3
        self.SCALE_STEP = 0.05
        self.PAD_X = 0.30
        self.PAD_Y = 0.40
        self.MAX_OPACITY = 0.82
        self._mouth_src = None
        self._landmarker = None
        self._smooth_rx = self._smooth_ry = self._smooth_rw = self._smooth_rh = None
        self._frame_count = 0
        self._face_found_cache = False

    def _build_landmarker(self):
        import mediapipe as mp
        from mediapipe.tasks import python as _mp_python
        from mediapipe.tasks.python import vision as _mp_vision
        model_path = "models/face_landmarker.task"
        base_options = _mp_python.BaseOptions(model_asset_path=model_path)
        options = _mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return _mp_vision.FaceLandmarker.create_from_options(options)

    def get_landmarker(self):
        if self._landmarker is None:
            self._landmarker = self._build_landmarker()
        return self._landmarker

    def get_mouth_src(self):
        if self._mouth_src is not None:
            return self._mouth_src
        base = os.path.join(os.path.dirname(__file__), "..", "assets")
        for name in ("mouth.png", "mouth.jpeg", "mouth.jpg"):
            path = os.path.normpath(os.path.join(base, name))
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                self._mouth_src = img
                return self._mouth_src
        print(f"[Mouth] Warning: no mouth image in assets/, using fallback")
        fallback = np.zeros((100, 200, 4), dtype=np.uint8)
        fallback[:, :, :3] = (0, 0, 200)
        fallback[:, :, 3] = 180
        self._mouth_src = fallback
        return self._mouth_src

    def match_color_lab(self, src_bgr, dst_bgr, alpha_mask):
        src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        dst_lab = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        visible = alpha_mask > 0.1
        if not visible.any():
            return src_bgr
        for c in range(3):
            s_vals = src_lab[:, :, c][visible]
            d_vals = dst_lab[:, :, c][visible]
            s_mean, s_std = s_vals.mean(), s_vals.std() + 1e-6
            d_mean, d_std = d_vals.mean(), d_vals.std() + 1e-6
            src_lab[:, :, c] = (src_lab[:, :, c] - s_mean) * (d_std / s_std) + d_mean
        src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)

    def make_ellipse_alpha(self, h, w, feather):
        mask = np.zeros((h, w), dtype=np.float32)
        cx, cy = w // 2, h // 2
        cv2.ellipse(mask, (cx, cy), (max(cx - 4, 1), max(cy - 4, 1)),
                    0, 0, 360, 1.0, -1)
        if feather > 0:
            ksize = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), feather * 0.5)
        return mask

    def alpha_blend(self, background, overlay_bgr, alpha_2d, x, y):
        H, W = background.shape[:2]
        oh, ow = overlay_bgr.shape[:2]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + ow, W), min(y + oh, H)
        if x2 <= x1 or y2 <= y1:
            return background
        sx1, sy1 = x1 - x, y1 - y
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
        roi = background[y1:y2, x1:x2].astype(np.float32)
        ov = overlay_bgr[sy1:sy2, sx1:sx2].astype(np.float32)
        a = alpha_2d[sy1:sy2, sx1:sx2, np.newaxis]
        blended = (ov * a + roi * (1.0 - a)).clip(0, 255).astype(np.uint8)
        out = background.copy()
        out[y1:y2, x1:x2] = blended
        return out

    def _smooth(self, prev, curr):
        if prev is None:
            return curr
        return prev * _SMOOTH + curr * (1.0 - _SMOOTH)

    def process(self, frame):
        import mediapipe as mp
        h, w = frame.shape[:2]
        self._frame_count += 1

        if self._frame_count % 2 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_rgb = cv2.resize(rgb, (320, 240))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)
            detector = self.get_landmarker()
            result = detector.detect(mp_image)
            if not result.face_landmarks:
                self._smooth_rx = self._smooth_ry = self._smooth_rw = self._smooth_rh = None
                self._face_found_cache = False
                return frame, False
            self._face_found_cache = True
            landmarks = result.face_landmarks[0]
            pts = np.array([
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in MOUTH_IDS
            ])
            bx, by, bw, bh = cv2.boundingRect(pts)
            px = int(bw * self.PAD_X)
            py = int(bh * self.PAD_Y)
            rx = max(bx - px, 0)
            ry = max(by - py, 0)
            rw = min(bw + 2 * px, w - rx)
            rh = min(bh + 2 * py, h - ry)
            if rw < 8 or rh < 8:
                self._smooth_rx = self._smooth_ry = self._smooth_rw = self._smooth_rh = None
                self._face_found_cache = False
                return frame, True
            self._smooth_rx = self._smooth(self._smooth_rx, rx)
            self._smooth_ry = self._smooth(self._smooth_ry, ry)
            self._smooth_rw = self._smooth(self._smooth_rw, rw)
            self._smooth_rh = self._smooth(self._smooth_rh, rh)

        if not self._face_found_cache or self._smooth_rx is None:
            return frame, False

        rx = int(round(self._smooth_rx))
        ry = int(round(self._smooth_ry))
        rw = int(round(self._smooth_rw))
        rh = int(round(self._smooth_rh))
        scale = max(0.3, 1.0 + self.scale_delta)
        rw_s = max(int(rw * scale), 8)
        rh_s = max(int(rh * scale), 8)
        rx_s = rx - (rw_s - rw) // 2
        ry_s = ry - (rh_s - rh) // 2
        rx_s += self.offset_x
        ry_s += self.offset_y
        mouth_src = self.get_mouth_src()
        src_h, src_w = mouth_src.shape[:2]
        aspect = src_w / src_h
        if rw_s / rh_s > aspect:
            fit_h = rh_s
            fit_w = int(fit_h * aspect)
        else:
            fit_w = rw_s
            fit_h = int(fit_w / aspect)
        fit_w = max(fit_w, 4)
        fit_h = max(fit_h, 4)
        resized = cv2.resize(mouth_src, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
        fake_bgr = resized[:, :, :3]
        png_alpha = resized[:, :, 3].astype(np.float32) / 255.0
        feather = max(int(min(fit_w, fit_h) * 0.18), 3)
        if feather % 2 == 0:
            feather += 1
        ellipse_alpha = self.make_ellipse_alpha(fit_h, fit_w, feather)
        combined_alpha = ellipse_alpha * png_alpha * self.MAX_OPACITY
        off_x = rx_s + (rw_s - fit_w) // 2
        off_y = ry_s + (rh_s - fit_h) // 2
        roi_region = frame[
            max(off_y, 0):max(off_y, 0) + fit_h,
            max(off_x, 0):max(off_x, 0) + fit_w
        ]
        if roi_region.shape[:2] == (fit_h, fit_w):
            fake_bgr = self.match_color_lab(fake_bgr, roi_region, combined_alpha)
        return self.alpha_blend(frame, fake_bgr, combined_alpha, off_x, off_y), True

    def reset_smooth(self):
        self._smooth_rx = self._smooth_ry = self._smooth_rw = self._smooth_rh = None
        self._frame_count = 0
        self._face_found_cache = False
