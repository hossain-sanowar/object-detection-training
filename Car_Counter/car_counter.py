# Import necessary libraries
import numpy as np  # For numerical operations and handling arrays
from ultralytics import YOLO  # YOLO model for object detection
import cv2  # OpenCV for video capture and image processing
import cvzone  # Simplifies OpenCV tasks like drawing rectangles or adding text
import math  # Used for rounding confidence scores
from sort import *  # SORT algorithm for object tracking

# Initialize video capture object
cap = cv2.VideoCapture("../videos/cars.mp4")  # Capture video from file

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

# Create Mask
# Load a mask image to define the region of interest (ROI) in the frame
# https://www.canva.com/design/DAGNGvzsIuU/FN0U4T-YT7zhsQHHjklFIQ/edit
mask = cv2.imread("mask.png")  # Mask image

# Initialize SORT tracker with specified parameters
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define the line limits for counting objects
limits = [400, 297, 673, 297]  # Coordinates for the counting line
totalCount = []  # List to keep track of counted object IDs

# Main loop to process video frames
while True:
    success, img = cap.read()  # Capture a frame from the video

    if not success:  # Exit the loop if the frame is not captured
        break

    # Apply the mask to the frame to focus on the region of interest
    print("Image shape:", img.shape)
    print("Mask shape:", mask.shape)

# Resize the mask to match the dimensions of img
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

# Now apply the bitwise_and operation with the resized mask
    imgRegion = cv2.bitwise_and(img, mask_resized)


    # Overlay graphics on the image (like logos or other fixed elements)
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)  # Graphics image with transparency
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))  # Overlay the graphics onto the frame

    # Run YOLO model on the masked frame
    results = model(imgRegion, stream=True)

    # Prepare an empty array to store detection results
    detections = np.empty((0, 5))

    # Process each detection result
    for r in results:
        boxes = r.boxes  # Get the bounding boxes of detected objects
        for box in boxes:
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
            w, h = x2 - x1, y2 - y1  # Calculate width and height of the bounding box

            # Extract and round the confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Extract the class index and get the corresponding class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Filter detections by class and confidence score
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                # Add the detection to the array for tracking
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update the tracker with the current frame's detections
    resultsTracker = tracker.update(detections)

    # Draw the counting line on the image
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # Process each tracking result
    for result in resultsTracker:
        x1, y1, x2, y2, id = result  # Unpack the tracking result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
        w, h = x2 - x1, y2 - y1  # Calculate width and height of the bounding box

        # Draw a rectangle around the tracked object
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        # Display the ID of the tracked object
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        # Calculate the center of the bounding box
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # Draw the center point

        # Check if the object crosses the counting line
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if id not in totalCount:  # Count the object only once
                totalCount.append(id)  # Add the object ID to the total count
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)  # Change line color

    # Display the total count on the image
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # Show the processed frame in a window
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)  # Uncomment to show the region of interest separately

    # Wait for 1 millisecond before processing the next frame, and check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
