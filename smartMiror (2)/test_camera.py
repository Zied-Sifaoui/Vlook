# test_camera.py — Minimal camera + face detection test
import cv2
import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.processor import SyncProcessor

def test():
    print("🎥 Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # ✅ Force DirectShow backend on Windows
    
    if not cap.isOpened():
        print("❌ Cannot open camera. Try:")
        print("   • Close other apps using camera (Zoom, Teams, etc.)")
        print("   • Check Windows privacy settings: Settings > Privacy > Camera")
        return
    
    # Set common Windows-friendly resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Camera ready: {W}x{H}")
    
    # Load processor
    try:
        processor = SyncProcessor("models/face_landmarker.task")
        print("✅ MediaPipe model loaded")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        cap.release()
        return
    
    print("\n👉 Point your face at the camera. Press 'q' to quit.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame = cv2.flip(frame, 1)  # Mirror for selfie view
        
        # Detect
        landmarks = processor.detect(frame)
        
        # Visual feedback
        status = "✅ FACE DETECTED" if landmarks else "❌ No face"
        color = (0, 255, 0) if landmarks else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw landmarks if found
        if landmarks:
            for lm in landmarks:
                x, y = int(lm.x * W), int(lm.y * H)
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        
        cv2.imshow("Debug: Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    processor.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Test complete")

if __name__ == "__main__":
    test()