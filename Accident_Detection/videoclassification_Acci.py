from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('C:/Users/arink/AppData/Local/Microsoft/Windows/INetCache/IE/RJHMWM4Q/best[1].pt') 

video_path = "D:/Teens Involved in Car Crash That Was Like a Movie Scene.mp4"  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)  

    boxes = results[0].boxes.xyxy
    confidences = results[0].boxes.conf
    class_ids = results[0].boxes.cls

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].tolist()
        conf = confidences[i].item()
        class_id = int(class_ids[i].item())

        class_name = model.names[class_id] 

        if conf > 0.5:
            accident_detected = True
            label = f'Accident Detected! {class_name} {conf:.2f}'
            color = (0, 0, 255)
        else:
            accident_detected = False
            label = f'{class_name} {conf:.2f}'
            color = (0, 255, 0)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    out.write(frame)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
