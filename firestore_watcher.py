import json
import os
import threading
import time

import firebase_admin
from firebase_admin import credentials, firestore


class FirestoreWatcher:
    def __init__(self, cred_path, collection, document, field):
        self._collection = collection
        self._document = document
        self._field = field
        self._current_value = None
        self._listeners = []
        self._running = False
        self._thread = None

        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        self._db = firestore.client()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        print(f"[Firestore] Watching {self._collection}/{self._document}.{self._field}")

    def _poll(self):
        doc_ref = self._db.collection(self._collection).document(self._document)
        backoff = 1.0
        while self._running:
            try:
                doc = doc_ref.get()
                backoff = 1.0
                if doc.exists:
                    data = doc.to_dict() or {}
                    val = data.get(self._field)
                    if val != self._current_value:
                        old = self._current_value
                        self._current_value = val
                        print(f"[Firestore] {self._field} changed: {old!r} -> {val!r}")
                        for cb in self._listeners:
                            cb(val)
                else:
                    print(f"[Firestore] Document {self._collection}/{self._document} does not exist")
            except Exception as e:
                print(f"[Firestore] Poll error: {e} — retrying in {backoff:.0f}s")
                backoff = min(backoff * 2, 30.0)
            time.sleep(backoff)

    def on_change(self, callback):
        self._listeners.append(callback)

    def get(self):
        return self._current_value

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[Firestore] Watcher stopped")
