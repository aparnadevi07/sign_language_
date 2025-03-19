import cv2
from ultralytics import YOLO
import mediapipe as mp

# Load the custom-trained YOLOv8 model
model = YOLO('C:/thanu/tancam/detect/train9/weights/best.pt')

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB (MediaPipe needs RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results_mediapipe = hands.process(rgb_frame)

        # Draw hand landmarks if detected
        if results_mediapipe.multi_hand_landmarks:
            for hand_landmarks in results_mediapipe.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Perform YOLOv8 prediction
        results_yolo = model(frame)

        # Process and display results (bounding boxes, labels, etc.)
        for result in results_yolo:
            boxes = result.boxes
            classes = result.names

            for i, box in enumerate(boxes):
                label = classes[box.cls]
                confidence = box.conf
                cv2.putText(frame, f'{label}: {confidence:.2f}', (int(box.x1), int(box.y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the output
        cv2.imshow('Hand Tracking with Gesture Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
