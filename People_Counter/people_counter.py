import numpy as np
from ultralytics import YOLO  # YOLO model for object detection
import cv2  # OpenCV for video processing
import cvzone  # Helper library for drawing and text on images
import math  # Used for rounding numbers
from sort import *  # SORT algorithm for object tracking

# Initialize the video capture object to read the input video file
cap = cv2.VideoCapture("../Videos/people.mp4")

# Load the YOLO model with pre-trained weights
model = YOLO("../Yolo-Weights/yolov10l.pt")

# List of class names that YOLO can detect, corresponding to model outputs
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

# Load the mask image, which defines the region of interest in the video
mask = cv2.imread("mask.png")

# Initialize the SORT tracker with parameters
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define the coordinates of the lines for counting people crossing in two directions
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

# Initialize lists to keep track of IDs of people who have crossed the lines
totalCountUp = []
totalCountDown = []

# Main loop to process each frame of the video
while True:
    success, img = cap.read()  # Read a frame from the video
    if not success:  # If the frame was not read successfully, break the loop
        break

    # Apply the mask to the frame to focus on the region of interest
    imgRegion = cv2.bitwise_and(img, mask)

    # Overlay additional graphics on the frame
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))

    # Perform object detection on the masked region
    results = model(imgRegion, stream=True)

    # Initialize an empty array to store detections
    detections = np.empty((0, 5))

    # Process each detection result
    for r in results:
        boxes = r.boxes  # Get the bounding boxes of detected objects
        for box in boxes:
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1  # Calculate width and height of the bounding box

            # Extract and round the confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Extract the class index and get the corresponding class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Filter detections to only consider people with a confidence > 0.3
            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update the tracker with the current frame's detections
    resultsTracker = tracker.update(detections)

    # Draw the counting lines on the frame
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    # Process each tracked object
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Draw the bounding box and the ID on the frame
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        # Calculate the center point of the bounding box
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if the person crosses the upper counting line
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        # Check if the person crosses the lower counting line
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    # Display the count on the frame
    cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    # Show the processed frame
    cv2.imshow("Image", img)
    # cv2.waitKey(1)  # Wait for 1 ms before processing the next frame
    # Wait for 1 millisecond before processing the next frame, and check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop, release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
