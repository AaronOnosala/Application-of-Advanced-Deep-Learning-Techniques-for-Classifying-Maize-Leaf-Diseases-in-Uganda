# importing libraries
from ultralytics import YOLO

# loading the model
model = YOLO('yolov8l-cls.pt')
model.train(data='/Users/aarononosala/Documents/Makerere/Classification_maize',
            epochs=100, imgsz=640)
