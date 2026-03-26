import cv2
import os
import csv
import time
import mediapipe as mp
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# 1. SETUP LOGGING & FOLDERS
log_file = "proctoring_log.csv"
violation_dir = "violations"
MAX_STRIKES = 3
strike_count = 0
last_strike_time = 0  # To track cooldown
COOLDOWN_SECONDS = 5  # Wait 5 seconds between strikes

if not os.path.exists(violation_dir):
    os.makedirs(violation_dir)

if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Violation_Type", "Screenshot_Path", "Strike_Number"])

# 2. SETUP MODELS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
yolo_model = YOLO("yolov8n.pt") 
model_path = 'face_landmarker.task' 

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print(f"--- AI Proctoring Suite Day 9: Stable Strike System ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- PART A: YOLO SCAN ---
    yolo_results = yolo_model(frame, stream=True, conf=0.5, classes=[0, 67], verbose=False)
    phone_found = False
    people_count = 0

    for r in yolo_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 67: phone_found = True
            if cls == 0: people_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # --- PART B: BIOMETRIC SCAN ---
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    # --- PART C: VIOLATION LOGIC ---
    status = "SYSTEM OK"
    color = (0, 255, 0)
    violation_detected = False
    current_violation = ""

    if phone_found:
        status = "ALARM: PHONE DETECTED!"
        color = (0, 0, 255)
        violation_detected = True
        current_violation = "Phone_Usage"
    elif people_count > 1:
        status = f"ALARM: {people_count} PEOPLE!"
        color = (0, 0, 255)
        violation_detected = True
        current_violation = "Multiple_People"

    # --- STRIKE LOGIC WITH COOLDOWN ---
    current_time = time.time()
    
    # Only count a strike if a violation is seen AND the cooldown has passed
    if violation_detected and (current_time - last_strike_time > COOLDOWN_SECONDS):
        strike_count += 1
        last_strike_time = current_time # Reset the timer
        
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        log_time = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save Screenshot
        img_name = f"Strike_{strike_count}_{timestamp_str}.jpg"
        img_path = os.path.join(violation_dir, img_name)
        cv2.imwrite(img_path, clean_frame)
        
        # Log to CSV
        try:
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([log_time, current_violation, img_path, strike_count])
        except PermissionError:
            print("⚠️ Close the CSV file!")

        # --- TERMINATION CHECK ---
        if strike_count >= MAX_STRIKES:
            print(f"❌ EXAM TERMINATED: {MAX_STRIKES} Strikes Reached.")
            break

    # UI Dashboard
    cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.putText(frame, f"Strikes: {strike_count}/{MAX_STRIKES}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show Cooldown Warning
    if current_time - last_strike_time < COOLDOWN_SECONDS and strike_count > 0:
        cv2.putText(frame, "COOLDOWN ACTIVE", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Day 9 - Stable Strike Proctoring', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()