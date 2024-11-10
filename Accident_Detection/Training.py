# FOR TRAINING
from ultralytics import YOLO
model = YOLO("D:/CAT_python/YOLOv10/yolov10n.pt")
model.train(data="D:/CAT_python/customdata.yaml", epochs=100, imgsz=640)
