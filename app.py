import streamlit as st
import cv2
import os
import time
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. SETUP & DIRECTORIES ---
st.set_page_config(page_title="ProctorAI Pro", layout="wide")

# Create a folder to store the "Proof" screenshots
if not os.path.exists("evidence"):
    os.makedirs("evidence")

# --- 2. SESSION STATE ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'exam_active' not in st.session_state: st.session_state.exam_active = False
if 'strike_count' not in st.session_state: st.session_state.strike_count = 0
if 'last_strike_time' not in st.session_state: st.session_state.last_strike_time = 0
if 'violation_log' not in st.session_state: st.session_state.violation_log = []

# --- 3. MODELS (Cached) ---
@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    return yolo, detector

yolo_model, detector = load_models()

# --- 4. ENTRANCE GATE ---
if not st.session_state.logged_in:
    st.title("🛡️ Secure Exam Entrance")
    col_l, col_c, col_r = st.columns([1,2,1])
    with col_c:
        with st.container(border=True):
            name = st.text_input("Candidate Name")
            reg = st.text_input("Registration ID")
            if st.button("Enter Exam Hall", use_container_width=True):
                if name and reg:
                    st.session_state.user_name, st.session_state.user_id = name, reg
                    st.session_state.logged_in = True
                    st.rerun()

# --- 5. THE EXAM HALL ---
else:
    # --- SIDEBAR MONITOR ---
    st.sidebar.markdown(f"### 📷 Active Monitoring")
    frame_placeholder = st.sidebar.empty()
    alert_placeholder = st.sidebar.empty() # For the RED warning text
    st.sidebar.divider()
    st.sidebar.metric("Strikes", f"{st.session_state.strike_count} / 3")
    
    # --- MAIN QUIZ AREA ---
    if not st.session_state.exam_active:
        st.header(f"Welcome, {st.session_state.user_name}")
        st.info("The exam will begin once the camera is initialized.")
        if st.button("🚀 START EXAM", use_container_width=True):
            st.session_state.exam_active = True
            st.rerun()
    else:
        st.header("📝 BCA Final Year Quiz")
        # Quiz Questions
        q1 = st.radio("What is the time complexity of Binary Search?", ["O(n)", "O(log n)", "O(n^2)"], index=None)
        
        if st.button("✅ SUBMIT EXAM", use_container_width=True):
            st.session_state.exam_active = False
            st.session_state.logged_in = False
            st.balloons()
            st.success("Submitted successfully!")
            time.sleep(2)
            st.rerun()

    # --- 6. THE AI ENGINE (With Screenshot Logic) ---
    if st.session_state.exam_active:
        cap = cv2.VideoCapture(0)
        f_count = 0
        
        while st.session_state.exam_active:
            ret, frame = cap.read()
            if not ret: break
            f_count += 1
            frame = cv2.flip(frame, 1)

            # Optimization: Process every 6th frame for speed
            if f_count % 6 == 0:
                small = cv2.resize(frame, (160, 90)) # Fast processing
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                # AI Scanning
                results = yolo_model(small, conf=0.5, classes=[0, 67], verbose=False)
                phone, p_count = False, 0
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == 67: phone = True
                        if int(box.cls[0]) == 0: p_count += 1
                
                face = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small))

                # --- VIOLATION LOGIC ---
                v_type = ""
                if phone: v_type = "Phone Detected"
                elif p_count > 1: v_type = "Multiple People"
                elif not face.face_landmarks: v_type = "Face Missing"

                now = time.time()
                if v_type and (now - st.session_state.last_strike_time > 5):
                    st.session_state.strike_count += 1
                    st.session_state.last_strike_time = now
                    
                    # 📸 SAVE SCREENSHOT AS EVIDENCE
                    img_name = f"evidence/{st.session_state.user_id}_strike_{st.session_state.strike_count}.jpg"
                    cv2.imwrite(img_name, frame) # Saves the full-quality frame
                    
                    # Log it
                    st.session_state.violation_log.append({"Time": datetime.now().strftime("%H:%M:%S"), "Type": v_type})
                    
                    # Termination check
                    if st.session_state.strike_count >= 3:
                        st.session_state.exam_active = False
                        st.error("🚨 EXAM VOIDED: Too many violations.")
                        cap.release()
                        st.rerun()

                # Display Feedback to Student
                if v_type:
                    alert_placeholder.error(f"⚠️ WARNING: {v_type}")
                else:
                    alert_placeholder.success("✅ System Status: OK")

                # Update Sidebar Camera
                frame_placeholder.image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            time.sleep(0.01)
        cap.release()