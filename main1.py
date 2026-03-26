import cv2
import mediapipe as mp
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
                               for p in face_landmarks.landmark])

        # --- 1. HEAD POSE (Day 3) ---
        nose = face_landmarks.landmark[1]
        l_eye_c = face_landmarks.landmark[33]
        r_eye_c = face_landmarks.landmark[263]
        h_ratio = abs(nose.x - l_eye_c.x) / abs(nose.x - r_eye_c.x)
        
        head_status = "Forward"
        if h_ratio < 0.6: head_status = "Looking Left"
        elif h_ratio > 1.6: head_status = "Looking Right"

        # --- 2. EYE GAZE (Day 4) ---
        (l_cx, l_cy), _ = cv2.minEnclosingCircle(mesh_points[468:473])
        (r_cx, r_cy), _ = cv2.minEnclosingCircle(mesh_points[473:478])
        l_eye_w = mesh_points[133][0] - mesh_points[33][0]
        r_eye_w = mesh_points[263][0] - mesh_points[362][0]
        avg_gaze = ((l_cx - mesh_points[33][0])/l_eye_w + (r_cx - mesh_points[362][0])/r_eye_w) / 2
        
        gaze_status = "Centered"
        if avg_gaze < 0.40: gaze_status = "Gaze Left"
        elif avg_gaze > 0.60: gaze_status = "Gaze Right"

        # --- 3. MOUTH DETECTION (Day 5) ---
        # Landmark 13 is Upper Lip (inner), 14 is Lower Lip (inner)
        upper_lip = mesh_points[13]
        lower_lip = mesh_points[14]
        # Calculate vertical distance
        mouth_distance = np.linalg.norm(upper_lip - lower_lip)
        
        mouth_status = "Closed"
        # If distance is more than 10 pixels (adjust based on your camera distance)
        if mouth_distance > 15:
            mouth_status = "Talking/Open"

        # --- VISUALIZATION ---
        h_color = (0, 255, 0) if head_status == "Forward" else (0, 0, 255)
        g_color = (0, 255, 0) if gaze_status == "Centered" else (0, 0, 255)
        m_color = (0, 255, 0) if mouth_status == "Closed" else (0, 0, 255)

        cv2.putText(frame, f"Head: {head_status}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, h_color, 2)
        cv2.putText(frame, f"Gaze: {gaze_status}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, g_color, 2)
        cv2.putText(frame, f"Mouth: {mouth_status}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, m_color, 2)

    cv2.imshow('AI Proctoring Suite', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()