import cv2
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 Model (Update the path)
model_path = "C:/thanu/tancam/runs/detect/train2/weights/best.pt"
model = YOLO(model_path)

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Class Names Mapping (Matches `data.yaml`)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster YOLO inference (Adjustable)
    resized_frame = cv2.resize(frame, (640, 480))

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face & hand landmarks using MediaPipe
    holistic_results = holistic.process(rgb_frame)

    # Draw face mesh
    if holistic_results.face_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

    # Draw hand landmarks
    if holistic_results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if holistic_results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # YOLOv8 Inference
    results = model(resized_frame, conf=0.5)  # Adjust confidence threshold if needed

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class index

            # ✅ Ignore low-confidence detections
            if conf < 0.5:
                continue

            # ✅ Get class name
            class_name = class_names[cls]

            # ✅ Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            print(f"Detected: {class_name}, Confidence: {conf:.2f}")

    # Show the output frame
    cv2.imshow("Sign Language Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
