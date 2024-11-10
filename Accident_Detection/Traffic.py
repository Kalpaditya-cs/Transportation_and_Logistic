import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
model_path = 'D:/CAT_python/YOLOv10/yolov10n.pt'

model = YOLO(model_path) 

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = model(rgb_frame) 
    
    predictions = results[0].boxes 
    vehicle_classes = [2, 3, 5, 7]  
    vehicle_count = 0

    for result in predictions:
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        conf = result.conf[0].item()
        cls = int(result.cls[0].item())
        
        if cls in vehicle_classes:
            vehicle_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{cls} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame, vehicle_count

max_green_time = 60  # Maximum green light time in seconds
min_green_time = 10  # Minimum green light time in seconds
vehicle_threshold = 10  # Threshold number of vehicles to change light duration
default_green_time = 30  # Default green light time in seconds

#cap = cv2.VideoCapture('D:/CAT_python/YOLOv10/videoplayback (2).mp4') 
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame, vehicle_count = process_frame(frame)
    
    if vehicle_count > vehicle_threshold:
        green_light_time = min(max_green_time, default_green_time + (vehicle_count - vehicle_threshold) * 2)
    else:
        green_light_time = max(min_green_time, default_green_time - (vehicle_threshold - vehicle_count) * 2)
    
    print(f"Vehicle Count: {vehicle_count}, Green Light Time: {green_light_time} seconds")
    
    cv2.putText(frame, f'Traffic Light: {green_light_time} sec Green', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Traffic Monitoring', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 time.sleep(1) 
cap.release()
cv2.destroyAllWindows()
