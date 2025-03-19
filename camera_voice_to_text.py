import cv2
import speech_recognition as sr
import threading  # To handle speech recognition separately

# Initialize webcam
cap = cv2.VideoCapture(0)  # Open webcam

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Function for speech recognition
def recognize_speech():
    with sr.Microphone() as source:
        print("Adjusting for background noise... Please wait")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Noise reduction
        print("Listening... Speak now.")

        try:
            audio = recognizer.listen(source, timeout=5)  # Add timeout to avoid waiting forever
            text = recognizer.recognize_google(audio)  # Convert to text
            print("You said:", text)
        except sr.UnknownValueError:
            print("❌ Could not understand the audio.")
        except sr.RequestError:
            print("❌ API request error")

# Start speech recognition in a separate thread
speech_thread = threading.Thread(target=recognize_speech, daemon=True)
speech_thread.start()

while True:
    # Capture video frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Display video frame
    cv2.imshow("Camera", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
