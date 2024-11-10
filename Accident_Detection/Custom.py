from ultralytics import YOLO

# Load YOLOv10n model from scratch
model = YOLO("D:/CAT_python/YOLOv10/yolov10n.pt")

# Train the model
model.train(data="D:/CAT_python/customdata.yaml", epochs=100, imgsz=640)