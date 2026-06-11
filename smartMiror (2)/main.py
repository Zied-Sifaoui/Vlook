import cv2
import time
import numpy as np
import threading
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modes.hair_ar import HairARMode
from modes.brow_shaper import BrowShaperMode
from modes.scar_remover import ScarRemoverMode
from modes.mouth_swap import MouthSwapMode
from ui.controls import WindowManager, MODES
from core.api_server import APIServer
#from core.remote import RemoteController
from core.geometry import GeometryEngine
from core.face_mask import build_face_depth_map
from core.renderer import ARRenderer
from core.processor import FaceProcessor, SyncProcessor

import firebase_admin
from firebase_admin import credentials, firestore

ESP32_IP = "10.182.242.79"
MODEL_PATH = "models/face_landmarker.task"
HAIR_OBJ = "assets/hair.obj"
MOUTH_IMG = "assets/images/mouth.jpg"
UNET_WEIGHTS = "unet_scar.pth"
WIN_NAME = "V-Look Beauty Suite"

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


class VLookApp:
    def __init__(self):
        self.mode = 0
        self.running = True
        self.win_name = WIN_NAME

        # ── Firebase listener (method is defined below inside the class) ──
        self._firebase_watch = (
            db.collection('jetson_control')
              .document('active_filter')
              .on_snapshot(self.on_filter_change)
        )

        if os.name == 'nt':
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera 0")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.K = GeometryEngine.get_camera_matrix(self.W, self.H)

        self.ui = WindowManager(WIN_NAME)
        self.hair = HairARMode(MODEL_PATH, HAIR_OBJ, self.W, self.H, self.K.copy())
        self.brow = BrowShaperMode()
        self.scar = ScarRemoverMode(unet_weights=UNET_WEIGHTS)
        self.mouth = MouthSwapMode(MOUTH_IMG)
        self.sync_proc = SyncProcessor(MODEL_PATH)

        self.remote = APIServer(host="0.0.0.0", port=8000)
        self.remote.start()
        print(f"[V-Look] {self.W}x{self.H} | A/D mode | C/1-5 style | Q/ESC quit")

    # ── Firebase callback — must be inside the class ──────────────────────
    def on_filter_change(self, doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            data = doc.to_dict()
            category = data.get('category')
            sub = data.get('subCategory')

            mode_map = {
                ('hair', 'hair'): 0,
                ('hair', 'eyebrows'): 1,
                ('face_surgery', 'scar'): 2,
                ('face_surgery', 'mouth'): 3,
                ('face_surgery', 'nose'): 4,
            }
            new_mode = mode_map.get((category, sub))
            if new_mode is not None:
                self.mode = new_mode
                print(f"[Firebase] Mode changed to {new_mode} ({category}/{sub})")

    def _sync_remote_to_ui(self):
        try:
            r_data = self.remote.data
        except (AttributeError, Exception):
            return
        if not r_data:
            return

        if "mode" in r_data:
            self.mode = int(r_data["mode"]) % len(MODES)
        if "color" in r_data:
            self.ui.color_idx = int(r_data["color"]) % len(self.ui.presets)

        slider_map = {
            "x": ("Offset X", 2000),
            "y": ("Offset Y", 2170),
            "z": ("Offset Z", 2000),
            "scale": ("Mesh Scale", 450),
        }
        for key, (name, _) in slider_map.items():
            if key in r_data:
                try:
                    val = int(r_data[key])
                    cv2.setTrackbarPos(name, WIN_NAME, val)
                except (ValueError, KeyError, cv2.error):
                    pass

    def run(self):
        self.ui.create_interface()
        cv2.waitKey(100)
        print("[V-Look] Running")

        while self.running:
            self._sync_remote_to_ui()

            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            face_found = False
            t0 = time.perf_counter()

            if self.mode == 0:
                ts_ms = int((time.perf_counter() - self.hair._t0) * 1000)
                self.hair.processor.process_frame(frame, ts_ms)
                with self.hair._lock:
                    if self.hair._fd["found"]:
                        face_found = True
                        rvec_f, _ = cv2.Rodrigues(self.hair._fd["R_mat"])
                        tvec_f = self.hair._fd["tvec"]
                        lms_f = self.hair._fd["landmarks"]
                        face_depth = None
                        if lms_f is not None:
                            face_depth = build_face_depth_map(lms_f, self.W, self.H, rvec_f, tvec_f, self.K)
                        t0 = time.perf_counter()
                        frame = self.hair.renderer.render(
                            frame, rvec_f, tvec_f, self.K,
                            style=self.ui.get_style(),
                            offset_x=self.ui.get_offset_x(),
                            offset_y=self.ui.get_offset_y(),
                            offset_z=self.ui.get_offset_z(),
                            mesh_scale=self.ui.get_mesh_scale(),
                            face_depth_map=face_depth,
                            occ_margin=15.0,
                        )
                        # ── DEBUG: Draw Coordinate Axes ──────────────────────────
                        if self.mode == 0 and face_found:
                            face_origin = tvec_f.reshape(3)
                            axis_len = 80.0
                            axes_3d = np.float32([
                                [axis_len, 0, 0],
                                [0, axis_len, 0],
                                [0, 0, axis_len],
                            ]).reshape(-1, 3)
                            axes_2d, _ = cv2.projectPoints(
                                axes_3d, rvec_f, face_origin.reshape(3, 1), self.K, np.zeros((4, 1))
                            )
                            axes_2d = axes_2d.reshape(-1, 2).astype(int)
                            center_2d, _ = cv2.projectPoints(
                                np.float32([0, 0, 0]).reshape(-1, 3),
                                rvec_f, face_origin.reshape(3, 1), self.K, np.zeros((4, 1))
                            )
                            center_2d = tuple(center_2d[0].astype(int).ravel())

                            if all(0 <= p < self.W or 0 <= p < self.H for p in center_2d):
                                cv2.line(frame, center_2d, tuple(axes_2d[0]), (0, 0, 255), 3)
                                cv2.line(frame, center_2d, tuple(axes_2d[1]), (0, 255, 0), 3)
                                cv2.line(frame, center_2d, tuple(axes_2d[2]), (255, 0, 0), 3)
                                cv2.putText(frame, "FACE", center_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                            mesh_offset = self.ui.get_offset_y()
                            mesh_anchor = face_origin + np.array([0, mesh_offset, 0])
                            mesh_axes_2d, _ = cv2.projectPoints(
                                axes_3d, rvec_f, mesh_anchor.reshape(3, 1), self.K, np.zeros((4, 1))
                            )
                            mesh_axes_2d = mesh_axes_2d.reshape(-1, 2).astype(int)
                            mesh_center_2d, _ = cv2.projectPoints(
                                np.float32([0, 0, 0]).reshape(-1, 3),
                                rvec_f, mesh_anchor.reshape(3, 1), self.K, np.zeros((4, 1))
                            )
                            mesh_center_2d = tuple(mesh_center_2d[0].astype(int).ravel())

                            if all(0 <= p < self.W or 0 <= p < self.H for p in mesh_center_2d):
                                cv2.line(frame, mesh_center_2d, tuple(mesh_axes_2d[0]), (100, 100, 255), 2)
                                cv2.line(frame, mesh_center_2d, tuple(mesh_axes_2d[1]), (100, 255, 100), 2)
                                cv2.line(frame, mesh_center_2d, tuple(mesh_axes_2d[2]), (255, 100, 100), 2)
                                cv2.putText(frame, "MESH", mesh_center_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        # ── END DEBUG ────────────────────────────────────────────
                        self.ui.set_render_ms((time.perf_counter() - t0) * 1000)
            else:
                lms = self.sync_proc.detect(frame)
                if self.mode == 1:
                    frame = self.brow.process(frame, lms, self.W, self.H)
                elif self.mode == 2:
                    frame = self.scar.process(frame, lms)
                elif self.mode == 3:
                    frame = self.mouth.process(frame, lms)
                    if hasattr(self.mouth, 'draw_controls'):
                        self.mouth.draw_controls(frame)
                face_found = False

            frame = self.ui.draw_hud(frame, self.mode, face_found=face_found)
            self.ui.maybe_save_screenshot(frame)
            cv2.imshow(self.win_name, frame)

            running, key, mode_delta = self.ui.handle_input()
            if not running:
                self.running = False
                break
            if mode_delta != 0:
                self.mode = (self.mode + mode_delta) % len(MODES)

        print("[V-Look] Shutting down...")
        self.hair.close()
        if self.remote:
            self.remote.stop()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        VLookApp().run()
    except KeyboardInterrupt:
        print("\n[V-Look] Interrupted")
    except Exception as e:
        print(f"[FATAL] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()