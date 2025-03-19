import cv2
import mediapipe as mp
from ultralytics import YOLO

# Load YOLO model
model = YOLO("C:/thanu/tancam/runs/detect/train9/weights/best.pt")

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Mapping class IDs to ASL meanings
asl_labels = {0: "Hello", 1: "Thank You", 2: "Yes", 3: "No", 4: "Love"}  # Update this based on your trained classes

# Open video
cap = cv2.VideoCapture('WhatsApp Video 2025-02-28 at 11.28.41 AM.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(frame, stream=True)

    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)  # Bounding box
            class_id = int(cls)  # Get class ID
            confidence = float(conf)  # Confidence score

            # Get label (default to "Unknown" if not in dictionary)
            label = asl_labels.get(class_id, "Unknown")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display label & confidence
            text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show output
    cv2.imshow("ASL Recognition - YOLO + MediaPipe", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
