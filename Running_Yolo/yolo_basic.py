from ultralytics import YOLO
import cv2

model=YOLO('yolov10l.pt')
results=model("Images/car.jpg", show=True)
cv2.waitKey(0) # 0 means unless the user input, don't do anything