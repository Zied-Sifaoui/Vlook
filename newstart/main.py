import cv2
import time
import numpy as np
import threading

from core.processor import FaceProcessor
from core.renderer  import ARRenderer
from core.geometry  import GeometryEngine
from ui.controls    import WindowManager
from core.face_mask import build_face_depth_map


class VLookApp:

    def __init__(self):
        self.win_name = "V-Look System"
        self.running  = True

        self.geo       = GeometryEngine()
        self.ui        = WindowManager(self.win_name)
        self.renderer  = ARRenderer("assets/hair.obj", normalize_span=4.0)
        self.processor = FaceProcessor(
            "models/face_landmarker.task", self.on_result)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera 0.")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.K = self.geo.get_camera_matrix(self.W, self.H)
        print(f"[V-Look] {self.W}x{self.H}")

        self.lock      = threading.Lock()
        self.face_data = {"found": False, "R_mat": None, "tvec": None, "landmarks": None}
        self._t0       = time.perf_counter()

    def on_result(self, result, image, ts_ms):
        if not result.face_landmarks:
            with self.lock:
                self.face_data["found"] = False
            return

        dist  = np.zeros((4, 1), np.float32)
        lms   = result.face_landmarks[0]
        pts2d = np.array([[lms[i].x * self.W, lms[i].y * self.H]
                          for i in self.geo.SOLVE_IDX], dtype=np.float32)

        with self.lock:
            prev_r = None
            prev_t = self.face_data["tvec"]
            if self.face_data["R_mat"] is not None:
                prev_r, _ = cv2.Rodrigues(self.face_data["R_mat"])

        ok, rvec, tvec = self.geo.solve_pose(
            pts2d, self.K, dist, prev_rvec=prev_r, prev_tvec=prev_t)

        if not ok:
            return

        #tvec[0] = -tvec[0]
        #rvec[1] = -rvec[1]
        #rvec[2] = -rvec[2]

        R_new, _ = cv2.Rodrigues(rvec)

        with self.lock:
            self.face_data["R_mat"] = self.geo.smooth_rotation_matrix(
                self.face_data["R_mat"], R_new, alpha=0.3)
            prev_t = self.face_data["tvec"]
            self.face_data["tvec"]      = tvec if prev_t is None else (0.3 * prev_t + 0.7 * tvec)
            self.face_data["landmarks"] = lms
            self.face_data["found"]     = True

    def run(self):
        self.ui.create_interface()
        cv2.waitKey(100)
        print("[V-Look] Running — Q/ESC quit | TAB align panel")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            ts_ms = int((time.perf_counter() - self._t0) * 1000)
            self.processor.process_frame(frame, ts_ms)

            face_found = False
            with self.lock:
                if self.face_data["found"]:
                    face_found = True
                    rvec_f, _ = cv2.Rodrigues(self.face_data["R_mat"])
                    tvec_f    = self.face_data["tvec"]
                    lms_f     = self.face_data["landmarks"]

                    face_depth = None
                    if lms_f is not None:
                        face_depth = build_face_depth_map(
                            lms_f, self.W, self.H,
                            rvec_f, tvec_f, self.K)

                    t0 = time.perf_counter()
                    frame = self.renderer.render(
                        frame, rvec_f, tvec_f, self.K,
                        style          = self.ui.get_style(),
                        offset_x       = self.ui.get_offset_x(),
                        offset_y       = self.ui.get_offset_y(),
                        offset_z       = self.ui.get_offset_z(),
                        mesh_scale     = self.ui.get_mesh_scale(),
                        face_depth_map = face_depth,
                        occ_margin     = 15.0,
                    )
                    self.ui.set_render_ms((time.perf_counter() - t0) * 1000)

            frame = self.ui.draw_hud(frame, face_found=face_found)
            self.ui.maybe_save_screenshot(frame)
            cv2.imshow(self.win_name, frame)

            if not self.ui.handle_input():
                self.running = False

        print("[V-Look] Shutting down...")
        self.processor.close()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = VLookApp()
    app.run()