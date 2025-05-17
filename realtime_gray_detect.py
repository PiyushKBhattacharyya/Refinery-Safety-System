import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model (update the path if needed)
model = YOLO('runs/detect/train/weights/best.pt')

# Open webcam (use 0 for default camera, or provide a video file path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert grayscale back to 3 channels for YOLO
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Run YOLOv8 detection
    results = model(gray_3ch)
    # Plot results on the frame
    annotated = results[0].plot()
    cv2.imshow('YOLOv8 Grayscale Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 