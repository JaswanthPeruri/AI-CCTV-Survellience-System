# Example using OpenCV and YOLOv8
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Nano version for real-time
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{model.names[int(box.cls)]} {box.conf:.2f}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    cv2.imshow('Real-Time Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()