import cv2
import numpy as np
import speech_recognition as sr

# -----------------------------
# Speech recognizer
# -----------------------------
recognizer = sr.Recognizer()
mic = sr.Microphone()
print("System started... Speak something")

text_display = ""

while True:

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        # Convert speech to text
        text = recognizer.recognize_google(audio)
        text_display = text
        print("You said:", text)

    except sr.UnknownValueError:
        text_display = "Could not understand"
        print("Could not understand audio")

    except sr.RequestError as e:
        text_display = "API Error"
        print(f"API error: {e}")

    # -----------------------------
    # Create black screen
    # -----------------------------
    frame = np.zeros((300, 900, 3), dtype=np.uint8)  # ✅ Fixed: proper black image

    # Handle long text by wrapping or truncating
    display = text_display[:45] + "..." if len(text_display) > 45 else text_display

    # Show text
    cv2.putText(frame,
                display,
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3)

    cv2.imshow("Speech To Text (For Deaf)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cv2.destroyAllWindows()