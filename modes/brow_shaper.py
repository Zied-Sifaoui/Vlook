"""
modes/brow_shaper.py — Rounded eyebrow overlay + hand gesture + speech recognition
"""

import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import speech_recognition as sr
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

# ── Eyebrow landmark indices ─────────────────────────────────────────────────
LB_UP = [70, 63, 105, 66, 107]
LB_LO = [46, 53,  52, 65,  55]
RB_UP = [300, 293, 334, 296, 336]
RB_LO = [276, 283, 282, 295, 285]

# ── Hand connections for drawing ─────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── Eyebrow helpers ──────────────────────────────────────────────────────────
def _smooth_curve(pts, n=60):
    t  = np.linspace(0, 1, len(pts))
    to = np.linspace(0, 1, n)
    return np.column_stack([np.interp(to, t, pts[:, 0]),
                            np.interp(to, t, pts[:, 1])])


def _sample_brow_color(frame, lms, W, H):
    samples = []
    for idx in [234, 454, 10]:
        lm = lms[idx]
        cx, cy = int(lm.x * W), int(lm.y * H)
        p = frame[max(0, cy-12):cy+12, max(0, cx-12):cx+12]
        if p.size > 0:
            samples.append(np.mean(p, axis=(0, 1)))
    if not samples:
        return np.array([25, 35, 50], np.float32)
    skin = np.mean(samples, axis=0)
    gray = np.dot(skin, [0.114, 0.587, 0.299])
    return np.clip(skin * 0.25 + np.array([gray, gray, gray]) * 0.10,
                   8, 110).astype(np.float32)


def _draw_one_brow(frame, up_idx, lo_idx, lms, W, H, color):
    up = np.array([(lms[i].x*W, lms[i].y*H) for i in up_idx], np.float32)
    lo = np.array([(lms[i].x*W, lms[i].y*H) for i in lo_idx], np.float32)
    up = up[np.argsort(up[:, 0])]
    lo = lo[np.argsort(lo[:, 0])]

    n  = 60
    uc = _smooth_curve(up, n)
    lc = _smooth_curve(lo, n)

    t    = np.linspace(0, 1, n)
    lift = (np.exp(-((t - 0.42)**2) / (2 * 0.18**2))
            * np.mean(np.abs(uc[:, 1] - lc[:, 1])) * 0.55)
    au   = uc.copy(); au[:, 1] -= lift

    tap = 0.6 + 0.4 * np.sin(np.linspace(0, np.pi, n)) ** 0.4
    mid = (au[:, 1] + lc[:, 1]) / 2
    half= (lc[:, 1] - au[:, 1]) / 2
    tu  = np.column_stack([uc[:, 0], mid - half * tap])
    tl  = np.column_stack([lc[:, 0], mid + half * tap])
    cont= np.vstack([tu, tl[::-1]]).astype(np.int32)

    h2, w2 = frame.shape[:2]
    mask = np.zeros((h2, w2), np.float32)
    cv2.fillPoly(mask, [cont], 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=6, sigmaY=5)

    core = np.zeros((h2, w2), np.float32)
    sh   = (cont - cont.mean(0)) * 0.68 + cont.mean(0)
    cv2.fillPoly(core, [sh.astype(np.int32)], 1.0)
    core = cv2.GaussianBlur(core, (0, 0), sigmaX=3, sigmaY=2)

    comb = np.clip(mask * 0.50 + core * 0.72, 0, 1)
    a3   = np.stack([comb * 0.70] * 3, axis=-1)
    cl   = np.full_like(frame, color.astype(np.uint8))
    frame[:] = np.clip(frame * (1 - a3) + cl * a3, 0, 255).astype(np.uint8)


# ── Hand helpers ─────────────────────────────────────────────────────────────
def _get_finger_states(landmarks):
    fingers = []
    fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
    for tip, dip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(1 if landmarks[tip].y < landmarks[dip].y else 0)
    return fingers


def _recognize_gesture(fingers):
    thumb, index, middle, ring, pinky = fingers
    if   thumb==1 and index==1 and middle==1 and ring==1 and pinky==1:
        return "HELLO"
    elif thumb==0 and index==0 and middle==0 and ring==0 and pinky==0:
        return "NO"
    elif thumb==0 and index==1 and middle==0 and ring==0 and pinky==0:
        return "YES"
    elif thumb==1 and index==0 and middle==0 and ring==0 and pinky==1:
        return "THANKS"
    elif index==1 and middle==1 and ring==0:
        return "SMALLER"
    elif thumb==1 and index==1 and pinky==1:
        return "BIGGER"
    elif thumb==1 and index==0 and middle==0 and ring==0 and pinky==0:
        return "I DON'T LIKE IT"
    elif pinky==1 and index==0 and middle==0:
        return "I LIKE IT"
    elif thumb==0 and index==1 and middle==1 and ring==1 and pinky==1:
        return "BYE"
    return ""


def _draw_hand(frame, landmarks):
    h, w, _ = frame.shape
    points = []
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)


def _wrap_text(text, max_chars=55):
    words  = text.split()
    lines  = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current += (" " if current else "") + word
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [""]


# ── Speech recognition worker ────────────────────────────────────────────────
def _speech_worker(state):
    recognizer = sr.Recognizer()

    # Try PyAudio first, fall back to sounddevice if available
    _use_sd = False
    try:
        import sounddevice as sd
        _use_sd = True
        sd.check_input_settings()
    except Exception:
        _use_sd = False

    if _use_sd:
        import sounddevice as sd
        RATE = 16000
        BLOCK = int(RATE * 2)  # 2-second chunks
        audio_buffer = []
        recording = [False]
        stop_recording = [False]

        def callback(indata, frames, time_info, status):
            if status:
                pass
            audio_buffer.append(indata.copy())
            if not recording[0]:
                recording[0] = True

        try:
            with sd.InputStream(samplerate=RATE, channels=1, callback=callback,
                                blocksize=BLOCK, dtype='int16'):
                with state["lock"]:
                    state["status"] = "Listening..."
                while not stop_recording[0]:
                    sd.sleep(500)  # 500ms polling
                    if len(audio_buffer) >= 3:  # ~6 seconds accumulated
                        raw = np.concatenate(audio_buffer, axis=0).flatten()
                        audio_buffer.clear()
                        recording[0] = False
                        with state["lock"]:
                            state["status"] = "Processing..."
                        try:
                            audio_data = sr.AudioData(
                                raw.tobytes(), RATE, 2)  # 16-bit = 2 bytes
                            text = recognizer.recognize_google(audio_data)
                            with state["lock"]:
                                state["text"] = text
                                state["status"] = "Got it!"
                        except sr.UnknownValueError:
                            with state["lock"]:
                                state["status"] = "Couldn't understand"
                        except sr.RequestError:
                            with state["lock"]:
                                state["status"] = "API error"
                        except Exception:
                            with state["lock"]:
                                state["status"] = "Retrying..."
        except Exception as e:
            with state["lock"]:
                state["status"] = f"SD error: {e}"
        return

    # Fallback: try sr.Microphone() with PyAudio
    try:
        mic = sr.Microphone()
    except Exception as e:
        with state["lock"]:
            state["status"] = f"Mic unavailable: {e}"
        return

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    while True:
        try:
            with mic as source:
                with state["lock"]:
                    state["status"] = "Listening..."
                audio = recognizer.listen(source, phrase_time_limit=5)

            with state["lock"]:
                state["status"] = "Processing..."

            text = recognizer.recognize_google(audio)

            with state["lock"]:
                state["text"]   = text
                state["status"] = "Got it!"

        except sr.UnknownValueError:
            with state["lock"]:
                state["status"] = "Couldn't understand"
        except sr.RequestError:
            with state["lock"]:
                state["status"] = "API error"
        except Exception:
            pass


# ── Main mode class ──────────────────────────────────────────────────────────
class BrowShaperMode:

    def __init__(self):
        # Face landmarker
        base_f = _mp_python.BaseOptions(model_asset_path="models/face_landmarker.task")
        self._face_opts = _mp_vision.FaceLandmarkerOptions(
            base_options=base_f,
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._face_lm = None

        # Hand landmarker
        base_h = _mp_python.BaseOptions(model_asset_path="models/hand_landmarker.task")
        self._hand_opts = _mp_vision.HandLandmarkerOptions(
            base_options=base_h,
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._hand_lm = None

        # Speech state (shared with worker thread)
        self._speech_state = {"text": "", "status": "Idle", "lock": threading.Lock()}
        self._speech_thread = None

        self.timestamp = 0
        self._frame_count = 0
        self._cached_face_lms = None
        self._cached_hand_lms = None

    def _ensure_models(self):
        if self._face_lm is None:
            self._face_lm = _mp_vision.FaceLandmarker.create_from_options(self._face_opts)
        if self._hand_lm is None:
            self._hand_lm = _mp_vision.HandLandmarker.create_from_options(self._hand_opts)

    def start_speech(self):
        if self._speech_thread is not None and self._speech_thread.is_alive():
            return
        self._speech_state["text"] = ""
        self._speech_state["status"] = "Starting..."
        self._speech_thread = threading.Thread(
            target=_speech_worker, args=(self._speech_state,), daemon=True)
        self._speech_thread.start()

    def _init_models(self):
        self._ensure_models()

    def process(self, frame):
        self._ensure_models()
        h, w = frame.shape[:2]
        self.timestamp += 33
        self._frame_count += 1

        gesture = ""
        run_ml = (self._frame_count % 2 == 0)

        if run_ml:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_rgb = cv2.resize(rgb, (320, 240))
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)

            # ── Face / eyebrow ────────────────────────────────────────────────
            if self._face_lm is not None:
                try:
                    face_result = self._face_lm.detect_for_video(mp_img, self.timestamp)
                    if face_result.face_landmarks:
                        self._cached_face_lms = face_result.face_landmarks
                    else:
                        self._cached_face_lms = None
                except Exception as e:
                    print(f"[Brow] Face detection error: {e}")

            # ── Hand / gesture ────────────────────────────────────────────────
            if self._hand_lm is not None:
                try:
                    hand_result = self._hand_lm.detect_for_video(mp_img, self.timestamp)
                    if hand_result.hand_landmarks:
                        self._cached_hand_lms = hand_result.hand_landmarks
                    else:
                        self._cached_hand_lms = None
                except Exception as e:
                    print(f"[Brow] Hand detection error: {e}")

        # ── Render brows from cache ───────────────────────────────────────────
        if self._cached_face_lms is not None:
            for face_landmarks in self._cached_face_lms:
                color = _sample_brow_color(frame, face_landmarks, w, h)
                _draw_one_brow(frame, LB_UP, LB_LO, face_landmarks, w, h, color)
                _draw_one_brow(frame, RB_UP, RB_LO, face_landmarks, w, h, color)

        # ── Render hands from cache ───────────────────────────────────────────
        if self._cached_hand_lms is not None:
            for hand_landmarks in self._cached_hand_lms:
                _draw_hand(frame, hand_landmarks)
                fingers = _get_finger_states(hand_landmarks)
                gesture = _recognize_gesture(fingers)

        # ── Speech ────────────────────────────────────────────────────────
        with self._speech_state["lock"]:
            speech_text   = self._speech_state["text"]
            speech_status = self._speech_state["status"]

        # ── Subtitle bar ──────────────────────────────────────────────────
        SUBTITLE_H = 90
        cam_h, cam_w = frame.shape[:2]
        bar = np.zeros((SUBTITLE_H, cam_w, 3), dtype=np.uint8)
        bar[:] = (20, 20, 20)

        dot_color = (0, 220, 0) if speech_status == "Listening..." else (
            (0, 200, 255) if "Processing" in speech_status else (80, 80, 80))
        cv2.circle(bar, (18, SUBTITLE_H // 2), 7, dot_color, -1)
        cv2.putText(bar, speech_status, (32, SUBTITLE_H // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        lines = _wrap_text(speech_text, max_chars=70)
        lines = lines[-2:]
        y_offsets = [28, 58] if len(lines) == 2 else [45]
        for line, yo in zip(lines, y_offsets):
            cv2.putText(bar, line, (120, yo),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 180), 2, cv2.LINE_AA)

        alpha = 0.82
        frame[cam_h - SUBTITLE_H:cam_h, :] = cv2.addWeighted(
            frame[cam_h - SUBTITLE_H:cam_h, :], 1 - alpha,
            bar, alpha, 0)

        return frame, gesture, speech_text, speech_status

    def close(self):
        self._face_lm = None
        self._hand_lm = None
