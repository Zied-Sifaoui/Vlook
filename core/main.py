import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp  # Add this import
import time
import os

_BASE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_BASE, '..', 'models')
_ASSETS = os.path.join(_BASE, '..', 'assets', 'images')

def main():
    # Path to your Face Landmarker task model
    model_path = os.path.join(_MODELS, "face_landmarker.task")
    
    try:
        # Initialize FaceLandmarker with proper options
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        
        # Create the FaceLandmarker
        landmarker = vision.FaceLandmarker.create_from_options(options)
        print("✅ Face Landmarker initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing Face Landmarker: {e}")
        print(f"Please make sure '{model_path}' file exists in the current directory")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("🎥 Webcam opened successfully")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    
    frame_count = 0
    fps = 0
    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Calculate FPS every second
        if current_time - prev_time >= 1.0:
            fps = frame_count
            frame_count = 0
            prev_time = current_time
        
        # The model expects RGB images
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image - USE mp.Image, not vision.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Calculate timestamp in milliseconds
        timestamp_ms = int(current_time * 1000)
        
        try:
            # Detect landmarks
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Draw landmarks if any face detected
            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    # Draw all landmarks
                    for landmark in face_landmarks:
                        # Convert normalized coordinates to pixel coordinates
                        x_px = int(landmark.x * frame.shape[1])
                        y_px = int(landmark.y * frame.shape[0])
                        
                        # Draw landmark point
                        if 0 <= x_px < frame.shape[1] and 0 <= y_px < frame.shape[0]:
                            cv2.circle(frame, (x_px, y_px), 2, (0, 255, 0), -1)
            
            # Draw face bounding box if available
            if hasattr(results, 'face_detections') and results.face_detections:
                for detection in results.face_detections:
                    bbox = detection.bounding_box
                    cv2.rectangle(frame,
                                 (bbox.origin_x, bbox.origin_y),
                                 (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                                 (0, 255, 255), 2)
        
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display number of faces detected
        if hasattr(results, 'face_landmarks'):
            cv2.putText(frame, f"Faces: {len(results.face_landmarks)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show webcam frame
        cv2.imshow("Face Landmarks - Press 'q' to quit", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"face_landmark_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved frame as {filename}")
    
    # Cleanup
    print("🔄 Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    
    # Close the landmarker properly
    if hasattr(landmarker, 'close'):
        landmarker.close()
    
    print("✅ Program terminated successfully")

if __name__ == "__main__":
    # Check for required model file
    model_file = os.path.join(_MODELS, "face_landmarker.task")
    if not os.path.exists(model_file):
        print(f"❌ Model file '{model_file}' not found!")
        print("Please download it from MediaPipe's GitHub releases:")
        print("https://github.com/google-ai-edge/mediapipe/releases")
        print("\nOr let MediaPipe download it automatically by running this code:")
        print("""
        # This will automatically download the model on first run
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(
            model_asset_buffer=b''  # Empty buffer triggers automatic download
        )
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        
        # This will download the model
        landmarker = vision.FaceLandmarker.create_from_options(options)
        """)
    else:
        main()