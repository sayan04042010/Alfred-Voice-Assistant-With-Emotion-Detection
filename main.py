import cv2
from ultralytics import YOLO

# 1. Load your custom model
# Ensure 'best.pt' is in the same folder as this script
model = YOLO("last.pt") 

# 2. Initialize the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run Inference
    # imgsz should match what you used in Kaggle (likely 640)
    results = model(frame, conf=0.5, stream=True)

    for r in results:
        # Get annotated frame (boxes and labels)
        annotated_frame = r.plot() 
        
        # Access specific emotions detected
        for box in r.boxes:
            class_id = int(box.cls[0])
            emotion_name = r.names[class_id]
            print(f"Detected Emotion: {emotion_name}")

    # 4. Display the results
    cv2.imshow("YOLOv11 Emotion Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()