# core/api_server.py
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

class APIServer:
    def __init__(self, host="0.0.0.0", port=8000):
        self.app = FastAPI()
        self.host = host
        self.port = port
        
        # Initial State (Matches your ESP32 defaults)
        self._data = {
            "mode": 0, "color": 0,
            "x": 2000, "y": 2170, "z": 2000, "scale": 450
        }
        
        self._setup_routes()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        @self.app.get("/status")
        def get_status():
            return self._data

        @self.app.get("/set")
        def set_val(mode: int = None, color: int = None, x: int = None, 
                    y: int = None, z: int = None, scale: int = None):
            if mode is not None: self._data["mode"] = mode
            if color is not None: self._data["color"] = color
            if x is not None: self._data["x"] = x
            if y is not None: self._data["y"] = y
            if z is not None: self._data["z"] = z
            if scale is not None: self._data["scale"] = scale
            return {"status": "ok"}

        @self.app.get("/reset")
        def reset():
            self._data.update({"x": 2000, "y": 2170, "z": 2000, "scale": 450})
            return {"status": "reset"}

    def start(self):
        """Starts the server in a background thread so it doesn't block OpenCV"""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"[API] Server running at http://{self.host}:{self.port}")

    def _run(self):
        # We use log_level error to keep the console clean for your AR prints
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="error")

    @property
    def data(self):
        return self._data
    
    def stop(self):
        pass # Uvicorn handles cleanup on exit