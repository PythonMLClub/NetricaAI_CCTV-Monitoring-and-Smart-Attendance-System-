🚀 NetricaAI — Real-Time CCTV Monitoring & Smart Attendance System

AI-powered facial recognition, crowd monitoring, posture detection, and automated attendance logging.

This project provides an intelligent surveillance solution using real-time video analytics, deep learning, and computer vision models. NetricaAI integrates CCTV/RTSP camera streams with AI modules to deliver:

Live facial recognition

Liveness & posture analysis

Automated Entry/Exit attendance

Crowd detection & alerts

Historical logs and dashboards

Streamlit tools for employee face registration

🧠 Why NetricaAI is an AI Project

The system performs real-time intelligent understanding of CCTV footage using:

YOLOv8 – Face detection

ArcFace – Face embeddings & recognition

MediaPipe – Pose & face landmark alignment

Crowd behavior analysis

Anomaly & pattern tracking

Cosine similarity for identity matching

This makes the system truly AI-driven, not just a video monitoring tool.

🎯 Business Objectives

Automate attendance without physical biometric devices

Provide real-time security insights

Detect crowds or unusual movement

Centralize surveillance and logging

Enable HR/Admin to track attendance & crowd events

👥 Stakeholders

HR – Attendance reports & workforce analytics

Admin/Security – Real-time monitoring

IT/Infra – Network/CCTV management

Data/Analytics Team – Insights & trends

🛠️ Tech Stack
Backend

Python 3.10+

Flask / FastAPI

OpenCV, MediaPipe, FFmpeg

YOLOv8 (Ultralytics)

ArcFace (ONNX Runtime)

SQL Server (pyodbc)

ThreadPoolExecutor

Frontend

HTML / CSS / JS (Jinja templates)

Streamlit (Employee Registration)

Tools

Docker

Git / Git Bash

SSMS

VLC, Postman

📦 Folder Structure
/ (repo root)
│
├── models/                     # ArcFace, YOLO weights
├── captured_faces/             # Captured face snapshots
├── captured_crowds/            # Crowd snapshots
├── output_logs/                # Log CSVs per camera
├── templates/                  # HTML dashboards
├── static/                     # CSS, JS
├── utils/
│   ├── arcface_embedder.py
│   ├── db_handler.py
│
├── cctv_app.py                 # Main Flask application
├── face_register.py            # Streamlit registration
├── process_employee.py         # Bulk upload script
├── camera_locations.json       # Camera metadata
├── requirements.txt
└── .env                        # Environment configuration

🔄 End-to-End System Workflow

Employee Registration

Streamlit captures image

Detect face → ArcFace embedding

Save to SQL Server

Live CCTV Streaming

FFmpeg pulls RTSP frames

YOLOv8 detects faces

MediaPipe aligns face

ArcFace embedding → Recognition

Attendance Logic

Identify Entry / Exit camera

Infer event & insert logs

Save face snapshots

Crowd Detection

ROI selection

Detect groups ≥ 3

Save snapshots, push logs

Dashboards

Live view

Attendance logs

Crowd detection logs

Attendance summary

✨ Key Features
🔹 Real-Time Face Recognition

ArcFace embedding

Cosine similarity identity matching

Multi-camera support

🔹 Smart Attendance Automation

Entry/Exit inference

No manual biometric device needed

Fast SQL logging

Avoids false exits

🔹 Crowd Detection & Alerts

Proximity-based grouping

Static vs. moving crowd classification

Snapshots & DB logs

🔹 Posture Detection

Standing / Sitting detection using MediaPipe Pose

🔹 Live Video Streaming

/api/video_feed/<camera_id>

MJPEG format

Overlays for face, FPS, posture, ROI

🔹 Streamlit Employee Registration

Webcam capture

Image upload

Live face embedding

🔧 Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/YourUsername/NetricaAI.git
cd NetricaAI

2️⃣ Create Virtual Environment
python -m venv env
env\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Install FFmpeg (required for RTSP)

Download: https://ffmpeg.org/download.html

Add to PATH → verify:

ffmpeg -version

5️⃣ Configure .env

Example:

DB_DRIVER=ODBC Driver 18 for SQL Server
DB_SERVER=103.117.172.65
DB_NAME=Netrica
DB_USERNAME=sa
DB_PASSWORD=*****
RTSP_USER=DataMonitor
RTSP_PASSWORD=D@taMon1tor

6️⃣ Run the Flask App
python cctv_app.py


Local URL:
http://127.0.0.1:5004/

🧪 API Endpoints
🎥 Camera Control
Endpoint	Description
POST /api/start/<camera_id>	Start a camera stream
POST /api/stop/<camera_id>	Stop a camera
POST /api/start_all	Start all cameras
POST /api/stop_all	Stop all cameras
GET /api/status	Camera health
GET /api/video_feed/<id>	Live MJPEG feed
📌 Logs & Attendance
Endpoint	Description
GET /api/logs	Paginated attendance logs
GET /attendance-summary	First Entry / Last Exit per employee
GET /crowd-detection	All crowd events
🎯 ROI Management
Endpoint	Description
POST /api/set_roi/<camera_id>	Set ROI for crowd detection
POST /api/reset_roi/<camera_id>	Clear ROI
🖼️ System Architecture Diagram

(Include your SVG here)

Netrica_flow_diagram.svg

📌 Future Enhancements

Guard availability detection

Mobile usage detection

Meal monitoring

Virtual geofencing

ID card compliance

Worker-hour analytics

Auto-grouping improvements
