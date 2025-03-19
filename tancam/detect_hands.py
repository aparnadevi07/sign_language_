import cv2
import mediapipe as mp
from ultralytics import YOLO

# Load YOLO model (custom-trained model for sign language)
model = YOLO("yolov8n.pt")  # Replace with your custom-trained model if needed

# Initialize MediaPipe Hands for detecting hand landmarks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load class names from the YOLO model (from data.yaml or model itself)
model_names = model.names  # This gives the gesture class names like 'hello', 'thanks', etc.

# Open the webcam
cap = cv2.VideoCapture(0)  # Change to a video path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the frame for gesture detection
    results = model.predict(frame, stream=True)

    # Iterate over each result (detected object)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)  # Get the bounding box coordinates

            class_id = int(result.boxes.cls[0])  # Get the class ID of the detected object
            confidence = result.boxes.conf[0]  # Confidence score of the detection

            # Get the gesture name if the class ID corresponds to a trained class
            if class_id < len(model_names):
                gesture_name = model_names[class_id]  # Trained gesture name
            else:
                gesture_name = "unknown sign"  # Label for unknown signs

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

            # Label the bounding box with gesture name and confidence score
            label = f"{gesture_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert the BGR frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands to detect hand landmarks
    hand_results = hands.process(rgb_frame)

    # Draw landmarks for the detected hand (if any)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the final output frame with both gesture detection and hand landmarks
    cv2.imshow("Hand Gesture Detection - YOLO + MediaPipe", frame)

    # Press 'q' to exit the video loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
