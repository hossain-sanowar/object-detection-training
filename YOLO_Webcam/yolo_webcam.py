# Import necessary libraries
from ultralytics import YOLO  # YOLO model for object detection
import cv2  # OpenCV for video capture and image processing
import cvzone  # Simplifies some OpenCV tasks, like drawing rectangles or text
import math  # Used for rounding numbers
import time  # Used to calculate frame timing for FPS

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Capture video from the external camera (0); webcam (1)
#cap = cv2.VideoCapture("../videos/cars.mp4")  # Capture video from the recorded videos
cap.set(3, 1280)  # Set the width of the video capture to 1280 pixels
cap.set(4, 720)  # Set the height of the video capture to 720 pixels


# Load the YOLO model with the specified pre-trained weights
model = YOLO("../Yolo-Weights/yolov10l.pt")  # Adjust path to your weight file

# List of class names YOLO can detect, corresponding to model outputs
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Initialize variables to calculate FPS (Frames Per Second)
prev_frame_time = 0
new_frame_time = 0

# Main loop to process video frames
while True:
    new_frame_time = time.time()  # Record the current time for FPS calculation
    success, img = cap.read()  # Capture a frame from the webcam

    if not success:  # Check if frame was captured successfully
        break  # If not, exit the loop

    # Run YOLO model on the captured frame
    results = model(img, stream=True)

    # Process each detection result
    for r in results:
        boxes = r.boxes  # Get the bounding boxes of detected objects
        for box in boxes:
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0] #box.xywh[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers

            # Calculate width and height of the bounding box
            w, h = x2 - x1, y2 - y1

            # Draw a rectangle around the detected object
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Extract and round the confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Extract the class index and get the corresponding class name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Put the class name and confidence score on the image
            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculate FPS and print it to the console
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    # Display the processed frame
    cv2.imshow("Image", img)

    # Wait for 1 millisecond before processing the next frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
