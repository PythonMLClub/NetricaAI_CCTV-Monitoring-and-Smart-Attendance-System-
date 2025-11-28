## 🚀 NetricaAI – Intelligent CCTV Monitoring & Smart Attendance System
## <p align="center"> <img src="https://img.shields.io/badge/AI%20Powered-Computer%20Vision-blue?style=for-the-badge" />  <img src="https://img.shields.io/badge/Framework-Flask%20%7C%20FastAPI-green?style=for-the-badge" /> <img src="https://img.shields.io/badge/Models-ArcFace%20%7C%20YOLOv8-orange?style=for-the-badge" />  <img src="https://img.shields.io/badge/Database-SQL%20Server-red?style=for-the-badge" /> </p>

## NetricaAI is an advanced AI-driven CCTV Monitoring & Smart Attendance System built for enterprises, universities, and high-security environments It automates.

## 📌 Overview

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

The system uses Deep Learning + Computer Vision for intelligent, real-time understanding of CCTV feeds.

## 🌟 1. Key Features

### 🔍 1.1 Real-Time Face Recognition

YOLOv8 for precise face detection

ArcFace ONNX for high-accuracy embedding

MediaPipe for liveness & alignment

Cosine similarity matching

### ⏱ 1.2 Smart Attendance Automation

Entry/Exit detection based on camera type

Zero manual intervention

SQL Server logging

Prevents duplicate/false entries

### 🧍 1.3 Posture Detection

Standing / Sitting detection

Useful for monitoring staff behavior

### 👥 1.4 Crowd Detection

Detect groups ≥ 3 inside ROI

Track duration & behavior

Auto-capture & DB log snapshots

### 📊 1.5 Dashboards & Reporting

Attendance logs

Employee-based search

Crowd analytics

Live camera feeds

Summary page (Entry/Exit)

### 📸 1.6 Streamlit Employee Registration

Register via webcam or photo upload

Auto-generate ArcFace embeddings

Stores embedding + face image in SQL Server

## 🏗️ 2. System Architecture

RTSP CCTV Cameras
        │
        ▼
 FFmpeg Stream Pulling
        │
        ▼
YOLOv8 Face Detection
        │
        ▼
MediaPipe Alignment
        │
        ▼
ArcFace Embedding
        │
        ▼
Identity Matching
        │
        ├──▶ SQL Server (Employees, AttendanceLogs, CrowdLogs)
        └──▶ Live Stream Rendering (Flask)

## 📂 3. Project Structure

/NetricaAI
│
├── cctv_app.py                # Main backend (Flask)
├── face_register.py           # Streamlit registration UI
├── process_employee.py        # Bulk employee import
│
├── models/                    # ArcFace & YOLO models
├── templates/                 # HTML/Jinja dashboards
├── static/                    # JS, CSS, assets
├── utils/                     # Embedding + DB helpers
│
├── captured_faces/            # Saved face snapshots
├── captured_crowds/           # Crowd snapshots
├── output_logs/               # Log CSV files
│
├── camera_locations.json      # Camera configuration
├── requirements.txt
└── .env

## ⚙️ 4. Installation Guide

### 🟦 4.1 Clone the Repository

git clone https://github.com/YourRepo/NetricaAI.git

cd NetricaAI

### 🟦 4.2 Create Virtual Environment

python -m venv env

env\Scripts\activate

### 🟦 4.3 Install Dependencies

pip install -r requirements.txt

### 🟦 4.4 Configure .env

DB_DRIVER=ODBC Driver 18 for SQL Server

DB_SERVER=xxx.xxx.xxx.xxx

DB_NAME=NetricaAI

DB_USERNAME=xxxx

DB_PASSWORD=xxxx

RTSP_USER=DataMonitor

RTSP_PASSWORD=D@taMon1tor

### 🟦 4.5 Install FFmpeg

Download: https://ffmpeg.org/download.html

Verify: ffmpeg -version

### 🟦 4.6 Run the Application

python cctv_app.py


➡️ Access UI at: http://127.0.0.1:5004/

## 🔌 5. API Endpoints

### 🎥 Camera Operations

POST - /api/start/<camera_id> - Start camera stream

POST - /api/stop/<camera_id> - Stop camera stream

POST - /api/start_all - Start all cameras

POST - /api/stop_all - Stop all cameras

GET - /api/status - Camera health

### 📸 Live Video Streaming

| GET | /api/video_feed/<camera_id> | Live MJPEG stream |

### 📒 Logs & Attendance

| GET | /api/logs | Attendance logs |

| GET | /crowd-detection | Crowd events |

| GET | /attendance-summary | Daily entry–exit summary |


## 🖼️ 6. Screenshot Previews

📍 Dashboard

🎥 Live Stream

👥 Crowd Detection

🧍 Posture Detection

🧑‍💼 Employee Registration

## 🚀 7. Future Enhancements

ID Card Compliance Monitoring

Guard Availability Tracking

Mobile Phone Usage Detection

Meal/Sleep Monitoring

Virtual Geofencing

Enhanced Analytics Dashboard

## Project Flow

https://github.com/PythonMLClub/NetricaAI_CCTV-Monitoring-and-Smart-Attendance-System-/blob/main/Netrica_flow_diagram.svg
