# 🛡️ ProctorAI: AI-Powered Exam Proctoring System
**Final Year BCA Project | Specialized in AI/ML & IoT**

An intelligent, web-based proctoring suite built with **Streamlit**, **YOLOv8**, and **MediaPipe**.

## 🚀 Features
- **Biometric Monitoring:** Real-time face presence detection using MediaPipe.
- **Object Detection:** Detects mobile phones and unauthorized persons using YOLOv8.
- **Secure Exam Hall:** Questions are hidden until the AI camera is initialized.
- **Automated Evidence:** Saves screenshots of violations to an `evidence/` folder.
- **Modern UI:** Professional dashboard with real-time strike alerts.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **AI Models:** YOLOv8n, MediaPipe Face Landmarker
- **Frontend:** Streamlit
- **Computer Vision:** OpenCV

## 📦 Installation
1. Clone the repo: `git clone https://github.com/your-username/proctor-ai.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`