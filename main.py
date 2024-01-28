import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read() # get frame camera

    if success:
        results = model(frame) # predict yolo model

        annotated_frame = results[0].plot() # yolo detec img

        cv2.imshow("YOLOv8 Inference", annotated_frame) # show detec win

        if cv2.waitKey(1) & 0xFF == ord("q"): # q key -> quit
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()