import cv2
from ultralytics import YOLO

# Load pretrained vehicle detection model (COCO)
vehicle_model = YOLO("yolov8n.pt")  # or yolov8m.pt for better accuracy

# Load your trained license plate detection model
plate_model = YOLO("best.pt")

# COCO vehicle classes IDs (commonly used ones)
vehicle_classes = [2, 3, 5, 7]  # 2=car, 3=motorbike, 5=bus, 7=truck

# Map COCO class IDs to names (you can customize this)
coco_class_names = vehicle_model.names

cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam. Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Detect vehicles in the frame
    vehicle_results = vehicle_model(frame)

    for v_result in vehicle_results:
        for box in v_result.boxes:
            cls = int(box.cls[0])
            if cls in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw vehicle bounding box and label
                label = coco_class_names[cls]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # Step 2: Crop vehicle region for plate detection
                vehicle_crop = frame[y1:y2, x1:x2]

                # Step 3: Detect license plates inside the vehicle crop
                plate_results = plate_model(vehicle_crop)

                for p_result in plate_results:
                    for p_box in p_result.boxes:
                        px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                        # Adjust plate box coordinates relative to original frame
                        abs_x1, abs_y1 = x1 + px1, y1 + py1
                        abs_x2, abs_y2 = x1 + px2, y1 + py2

                        # Draw plate bounding box and label
                        plate_label = "plate"
                        cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                        cv2.putText(frame, plate_label, (abs_x1, abs_y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Vehicle + Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
