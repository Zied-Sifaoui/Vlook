import mediapipe as mp

print("MediaPipe version:", mp.__version__)

try:
    face_detection = mp.solutions.face_detection
    print("FaceDetection loaded: True")
except Exception as e:
    print("FaceDetection loaded: False")
    print(e)

