from ultralytics import YOLO

# Train YOLOv8 on the custom dataset
def main():
    model = YOLO('yolov8n.pt')  # You can change to yolov8s.pt, yolov8m.pt, etc.
    model.train(data='dataset/dataset/data.yaml', epochs=50, imgsz=640)

if __name__ == '__main__':
    main()