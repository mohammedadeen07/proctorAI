import cv2
import mediapipe as mp
from ultralytics import YOLO
import os

# 1. Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = YOLO("yolov8n.pt")  # This will download a small 6MB file on first run
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    
    # --- PART A: YOLO OBJECT DETECTION ---
    # We tell YOLO to only look for 'cell phone' (class 67) and 'person' (class 0)
    # classes=[0, 67] filters the results to just those two
    yolo_results = model(frame, stream=True, conf=0.5, classes=[0, 67], verbose=False)
    
    phone_detected = False
    person_count = 0

    for r in yolo_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            
            if label == "cell phone":
                phone_detected = True
            if label == "person":
                person_count += 1
            
            # Draw a bounding box around the detected object
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # --- PART B: FACE MESH (From previous days) ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)

    # --- PART C: FINAL LOGIC & ALERTS ---
    alert_text = "STATUS: OK"
    alert_color = (0, 255, 0)

    if phone_detected:
        alert_text = "WARNING: PHONE DETECTED!"
        alert_color = (0, 0, 255)
    elif person_count > 1:
        alert_text = f"WARNING: {person_count} PEOPLE FOUND!"
        alert_color = (0, 0, 255)

    cv2.putText(frame, alert_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 3)

    cv2.imshow('AI Proctoring - Day 6: Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()