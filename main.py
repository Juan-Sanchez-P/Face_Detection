import eel
import cv2
import numpy as np
import base64
import os
from ultralytics import YOLO

# Initialize Eel
eel.init("Gui")

# Load Pretrained YOLOv8 Model
model = YOLO("yolov8n.pt")

@eel.expose

def process_video(video_data):
    try:
        # Decode base64 video data
        video_bytes = base64.b64decode(video_data.split(",")[1])
        video_path = "temp_video.mp4"
        
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # Open video file
        cap = cv2.VideoCapture(video_path)
        detected_objects = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = model(frame)

            for result in results:
                for box in result.boxes:
                    label = model.names[int(box.cls)]
                    confidence = float(box.conf[0]) * 100
                    if label in ["dog", "cat"]:
                        detected_objects.append({"label": label, "confidence": round(confidence, 2)})

            # Break early for demo purposes (remove for full video processing)
            break

        cap.release()
        os.remove(video_path)

        return detected_objects

    except Exception as e:
        return [{"label": "Error", "confidence": str(e)}]

# Start the app
eel.start("index.html", size=(800, 600))
