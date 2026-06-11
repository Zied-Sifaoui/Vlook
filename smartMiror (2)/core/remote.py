"""
core/remote.py — ESP32 HTTP Polling Controller
✅ Thread-safe state sync for V-Look Beauty Suite
"""

import threading
import time
import requests
from typing import Dict, Optional


class RemoteController:
    """
    Background thread that polls the ESP32 web server for state updates.
    Thread-safe: use `.data` property to read latest synced values.
    """
    
    def __init__(self, ip: str, interval: float = 0.1):
        """
        Args:
            ip: ESP32 IP address (e.g., "10.182.242.79") — no http:// prefix
            interval: Polling interval in seconds (default: 0.1s = 10Hz)
        """
        # ✅ FIX: Build URL correctly — ip should NOT include http://
        if ip.startswith("http"):
            self.base_url = ip.rstrip("/")
        else:
            self.base_url = f"http://{ip}"
            
        self.interval = interval
        self._data: Dict[str, int] = {}  # ✅ FIXED: was "self._ Dict"
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"[Remote] Started polling {self.base_url}/status")
        
    def stop(self):
        """Stop the polling thread gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[Remote] Stopped")
        
    @property
    def data(self) -> Dict[str, int]:
        """
        Thread-safe access to latest ESP32 state.
        Returns a copy to prevent external mutation.
        """
        with self._lock:
            return self._data.copy()
            
    def _poll_loop(self):
        """Internal: fetch /status endpoint in a loop."""
        url = f"{self.base_url}/status"
        
        while self._running:
            try:
                resp = requests.get(url, timeout=1.0)
                if resp.status_code == 200:
                    new_data = resp.json()
                    
                    # Only update if values actually changed (reduce UI thrash)
                    with self._lock:
                        if new_data != self._data:  # ✅ FIXED: was "self._"
                            self._data = new_data
                            
            except requests.RequestException as e:
                # ESP32 may be rebooting, out of range, or network glitch
                # Silent fail — will retry on next poll
                pass
            except Exception as e:
                # Log unexpected errors but don't crash the thread
                print(f"[Remote WARN] Poll error: {e}")
                
            time.sleep(self.interval)