import cv2
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 Model
model_path = "runs/detect/train9/weights/best.pt"  # Update this to your trained model path
model = YOLO(model_path)

# Initialize MediaPipe Holistic (Google's full-body tracker)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define class labels (update these based on your trained YOLO model)
asl_labels = {
    0: "Hello",
    1: "Yes",
    2: "No",
    3: "Thank You",
    4: "Please",
    5: "I Love You",
    6: "Sorry",
    # Add more labels based on your trained YOLO model
}

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect full body landmarks (face, hands, pose) using MediaPipe Holistic
    holistic_results = holistic.process(rgb_frame)

    # Draw cleaner face mesh (only contours, no full mesh)
    if holistic_results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            holistic_results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,  # Less clutter, only key contours
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),  # Outline
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)  # Dots
        )

    # Draw hand landmarks (left & right hand)
    if holistic_results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if holistic_results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Draw body pose landmarks
    if holistic_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # YOLOv8 ASL Sign Detection
    results = model(frame)  # Run YOLO detection
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class index

            # Get label from dictionary (default to "Unknown" if not found)
            label = asl_labels.get(cls, "Unknown")

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("ASL Detection (YOLOv8 + Google MediaPipe)", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
