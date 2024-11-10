import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO

# Define the path to your YOLOv10n model file
model_path = 'D:/CAT_python/YOLOv10/yolov10n.pt'

# Load the YOLOv10 model
model = YOLO(model_path)  # Load YOLOv10 model from the specified path

def process_frame(frame):
    # Convert frame to RGB (YOLO models expect RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform detection with YOLOv10n
    results = model(rgb_frame)  # Perform detection
    
    # Extract predictions
    predictions = results[0].boxes  # Adjust based on your model's output format
    vehicle_classes = [2, 3, 5, 7]  # COCO classes: Car, Motorcycle, Bus, Truck
    vehicle_count = 0

    # Process the predictions and annotate the frame
    for result in predictions:
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        conf = result.conf[0].item()
        cls = int(result.cls[0].item())
        
        if cls in vehicle_classes:
            vehicle_count += 1
            # Draw bounding box on detected vehicles
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{cls} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame, vehicle_count

# Traffic Light Control Parameters
max_green_time = 60  # Maximum green light time in seconds
min_green_time = 10  # Minimum green light time in seconds
vehicle_threshold = 10  # Threshold number of vehicles to change light duration
default_green_time = 30  # Default green light time in seconds

# Initialize video capture with a video file
#cap = cv2.VideoCapture('D:/CAT_python/YOLOv10/videoplayback (2).mp4')  # Replace with your video file path
# Initialize video capture from the web camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the current frame with YOLOv10n
    frame, vehicle_count = process_frame(frame)
    
    # Decision logic for traffic light control
    if vehicle_count > vehicle_threshold:
        green_light_time = min(max_green_time, default_green_time + (vehicle_count - vehicle_threshold) * 2)
    else:
        green_light_time = max(min_green_time, default_green_time - (vehicle_threshold - vehicle_count) * 2)
    
    # Simulate sending the green light time to a traffic light controller
    print(f"Vehicle Count: {vehicle_count}, Green Light Time: {green_light_time} seconds")
    
    # Display the frame with vehicle detections and traffic light status
    cv2.putText(frame, f'Traffic Light: {green_light_time} sec Green', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Traffic Monitoring', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Simulate the passage of time for the traffic light
    time.sleep(1)  # Adjust time.sleep() as needed for real-time simulation

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
