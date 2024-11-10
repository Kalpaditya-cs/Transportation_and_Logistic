# Transportation_and_Logistic
# AI Accident_Detection

https://github.com/user-attachments/assets/0a547a0d-7fc9-452e-8689-5c1bfa225706

# AI Traffic Management

![VID_20241110_015524](https://github.com/user-attachments/assets/d8312d1d-f797-4d91-9e62-326dc292ac0d)

# About the Project
Accident Detection and Traffic Management System
The Accident Detection and Traffic Management System is an intelligent solution that integrates accident detection and traffic management using computer vision and deep learning techniques. The system utilizes advanced object detection algorithms, such as YOLO (You Only Look Once), to monitor and analyze traffic in real-time. By processing video feeds, the system can detect accidents, monitor traffic flow, and optimize signal timings for improved road safety and traffic management.

This project was trained with a custom dataset of 1,500 images, which includes various scenarios like vehicle collisions, pedestrian crossings, and abnormal traffic behaviors. The system helps to identify accidents or potential hazards on the road as they occur and can also support dynamic traffic control, improving the efficiency and safety of urban transportation systems

# Requirements
* ultralytics==8.0.0
* opencv-python==4.7.0.72
* torch==2.1.0
* torchvision==0.15.0
* numpy==1.23.0
* matplotlib==3.6.0
* pandas==1.5.0
* tensorboard==2.12.0
* scikit-learn==1.0.2

# Overview

This project implements accident detection in video footage using YOLOv10, a powerful deep learning-based object detection model. The system processes video frames to detect accidents or accident-related activities (such as vehicle collisions, sudden stops, or vehicle behavior anomalies) in real-time. The model uses object detection to identify and track vehicles, people, and objects involved in accidents

# Features:

Real-time detection of accident-related events in videos. Bounding boxes around detected objects (vehicles, pedestrians, etc.).
Confidence scores for each detected object. Utilizes YOLOv10 for high-speed object detection and real-time video analysis.
Outputs a processed video with bounding boxes and labels for detections.

# Getting Started
Prerequisites
Before you begin, ensure you have met the following requirements:

Python 3.x installed.
pip or conda for installing dependencies.
A compatible GPU (optional but recommended for faster inference).

Installation
Install Dependencies
Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'

Install the required dependencies:
pip install -r requirements.txt


The requirements.txt should include:
ultralytics
opencv-python
numpy
torch

Usage

1. Load Your YOLOv10 Model
   Ensure you have a YOLOv10 model trained on accident-related data.

   To load your custom-trained YOLOv10 model:

   from ultralytics import YOLO

   model = YOLO('C:/Users/arink/AppData/Local/Microsoft/Windows/INetCache/IE/RJHMWM4Q/best[1].pt')  # Replace with your model path

2. Detect Accidents in a Video
   Once the model is loaded, you can run the provided script to process videos and detect accidents. The script will draw bounding 
   boxes around detected objects and display confidence scores.

   Run the detection script:

   from ultralytics import YOLO

   python detect_accidents.py --input_video path_to_video.mp4 --output_video output_video.mp4

3. View Results
   The processed video will be saved with bounding boxes around detected objects. The video will include labels with the class name and 
   confidence score. If an accident is detected, it will be highlighted based on the model's predictions.

4. Adjust Parameters (Optional)
   You can tweak the script to: Set the detection confidence threshold to ignore low-confidence detections.
   Modify post-processing logic to track objects across frames or detect accident-related behaviors

# Training Your Own YOLOv10 Model
To train your own YOLOv10 model for accident detection, follow these steps:

1. Prepare Your Dataset
  Collect a dataset of accident-related images or video frames. Annotate the images with bounding boxes and labels, specifying the 
  objects you want the model to detect (e.g., vehicles, pedestrians, and accident-related scenes). Organize your dataset into a format 
  compatible with YOLOv10 (e.g., image files and corresponding .txt label files).

2. Train the Mode
   Once your dataset is ready, you can train the YOLOv10 model. Here’s a general command to train it:
   python train.py --img 640 --batch 16 --epochs 50 --data accident_data.yaml --weights yolov10.pt --device 0
   accident_data.yaml: This YAML file should contain information about the dataset (e.g., paths to training and validation data and 
   class names).
   yolov10.pt: A pre-trained YOLOv10 model to fine-tune on your dataset

3. Evaluate and Test the Model
   After training, you can evaluate the model's performance on a validation set or a test video. Use the following command to test the 
   model
   python val.py --weights runs/train/exp/weights/best.pt --data accident_data.yaml


# Contributing
  We welcome contributions to improve accident detection models. If you have suggestions, bug fixes, or new features to contribute, 
  feel free to open an issue or submit a pull request.

# Acknowledgments
  YOLOv10: For powerful and efficient object detection.
  OpenCV: For video processing and visualization.
  PyTorch: For the deep learning framework used in YOLO.

# Conclusion:
  This README provides a structured guide for setting up accident detection using YOLOv10 on video footage. It includes installation 
  instructions, usage examples, and steps for training your own custom YOLOv10 model for specific accident detection tasks.

