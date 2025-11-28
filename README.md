🚀 NetricaAI – Intelligent CCTV Monitoring & Smart Attendance System
<p align="center"> <img src="https://img.shields.io/badge/AI%20Powered-Computer%20Vision-blue?style=for-the-badge" /> <img src="https://img.shields.io/badge/Technology-Flask%20%7C%20FastAPI%20%7C%20Python-green?style=for-the-badge" /> <img src="https://img.shields.io/badge/Models-ArcFace%20%7C%20YOLOv8-orange?style=for-the-badge" /> </p>
📌 Overview

NetricaAI is an advanced AI-powered CCTV Monitoring & Smart Attendance System designed to automate workforce attendance, enhance workplace security, and monitor environments in real time.

The platform combines Computer Vision, Deep Learning, FastAPI/Flask services, and SQL Server, enabling:

✔ Real-time facial recognition
✔ Automated Entry/Exit attendance
✔ Crowd monitoring
✔ Posture detection
✔ Live CCTV streaming
✔ Attendance & event dashboards
✔ Employee face registration via Streamlit

This system is ideal for corporate offices, universities, factories, and high-security environments.

✨ Key Features
🔍 1. Real-Time Face Recognition

YOLOv8 for face detection

ArcFace ONNX for high-accuracy embedding

Liveness & alignment using MediaPipe

⏱ 2. Smart Attendance Automation

Entry/Exit detection based on camera configuration

No biometric machines required

Accurate logs stored in SQL Server

🧍 3. Posture Analysis

Standing / Sitting classification

Useful for monitoring staff behavior

👥 4. Crowd Detection

Detect groups (3+ people) inside ROI

Automatic snapshot & DB logging

Crowd duration tracking

📊 5. Dashboards & Logs

Attendance logs

Employee-specific history

Crowd detection dashboard

Live camera monitoring

📸 6. Streamlit Registration App

Register employees via webcam or photo upload

Automatically generate embeddings

🏗️ System Architecture
RTSP CCTV Cameras ──▶ FFmpeg Stream Pulling
        │
        ▼
YOLOv8 Face Detection ──▶ MediaPipe (Alignment)
        │
        ▼
ArcFace Embedding ──▶ Identity Matching
        │
        ▼
Attendance Logic (Entry/Exit)
        │
        ├──▶ SQL Server (Employees, AttendanceLogs, CrowdLogs)
        └──▶ Live Stream Overlay (Flask/FastAPI)

📂 Project Structure
/NetricaAI
│
├── cctv_app.py                # Main Flask backend
├── face_register.py           # Streamlit employee registration
├── process_employee.py        # Bulk employee upload
│
├── models/                    # ArcFace & YOLO models
├── templates/                 # HTML dashboards
├── static/                    # CSS & JS
├── utils/                     # Embedding + DB utils
├── captured_faces/            # Saved recognized faces
├── captured_crowds/           # Crowd snapshots
├── output_logs/               # CSV logs
│
├── camera_locations.json      # Camera config
├── requirements.txt
└── .env

⚙️ Installation Guide
1️⃣ Clone the Repo
git clone https://github.com/YourRepo/NetricaAI.git
cd NetricaAI

2️⃣ Setup Virtual Environment
python -m venv env
env\Scripts\activate

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Setup Environment Variables

Create a .env file:

DB_DRIVER=ODBC Driver 18 for SQL Server
DB_SERVER=xxx.xxx.xxx.xxx
DB_NAME=NetricaAI
DB_USERNAME=xxxx
DB_PASSWORD=xxxx

RTSP_USER=DataMonitor
RTSP_PASSWORD=D@taMon1tor

5️⃣ Install FFmpeg

Required for RTSP stream decoding:
https://ffmpeg.org/download.html

6️⃣ Run Application
python cctv_app.py


➡️ Local dashboard:
http://127.0.0.1:5004/

🔌 Important API Endpoints
🎥 Camera Streaming
Endpoint	Description
/api/video_feed/<camera_id>	Live stream with overlays
/api/start_all	Start all cameras
/api/stop_all	Stop all cameras
📒 Logs
Endpoint	Purpose
/api/logs	Attendance logs
/crowd-detection	Crowd events
/attendance-summary	First/Last entry per employee
🖼️ Flow of project

https://github.com/PythonMLClub/NetricaAI_CCTV-Monitoring-and-Smart-Attendance-System-/blob/main/Netrica_flow_diagram.svg

