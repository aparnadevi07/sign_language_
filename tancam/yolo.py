from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load YOLO model

# Run YOLOv8 on webcam (0 is the default camera)
model.predict(source=0, show=True)
