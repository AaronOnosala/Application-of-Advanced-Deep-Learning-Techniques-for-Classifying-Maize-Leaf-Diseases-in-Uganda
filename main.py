# Import the YOLO library from the ultralytics module
from ultralytics import YOLO

# Initialize the YOLO model with a pre-trained classification model
model = YOLO('yolov8l-cls.pt')

# Train the YOLO model with the specified parameters
# - data: Path to the directory containing the maize classification dataset
# - epochs: Number of training epochs to run (100 in this case)
# - imgsz: Size of the input images for training (640x640 pixels)
model.train(data='/Users/aarononosala/Documents/Makerere/Classification_maize',
            epochs=100, imgsz=640)
