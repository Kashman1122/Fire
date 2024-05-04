from ultralytics import YOLO
import cvzone
import cv2
import math

# Running real time from webcam
# cap = cv2.VideoCapture("file4.mp4")
cap = cv2.VideoCapture("file6.mp4")

model = YOLO('best.pt')

# Reading the classes
classnames = ['smoke', 'fire']  # Add 'smoke' class

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    # Getting bbox, confidence, and class names information to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])

            # Adjust confidence thresholds for smoke and fire
            smoke_threshold = 30  # Adjust as needed
            fire_threshold = 50  # Adjust as needed

            if Class == 0 and confidence > smoke_threshold:  # Smoke class
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)  # Blue for smoke
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

            elif Class == 1 and confidence > fire_threshold:  # Fire class
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Red for fire
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


#IF YOU ARE WORKING WITH IOT DEVICE LIKE ESP32 CAM THEN U CAN JUST REPLACE THE IP ADDRESS OF THAT MODULE WITH "FILE.MP4" 

#url = 'http://192.168.22.XX/cam-hi.jpg'(use your ip address)
#cap = cv2.VideoCapture(url)



