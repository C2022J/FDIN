from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x-cls.pt")

# Perform object detection on an image
results = model("image/image.JPG")
results[0].show()
