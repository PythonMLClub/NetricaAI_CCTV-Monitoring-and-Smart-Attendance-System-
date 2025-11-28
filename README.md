# 🚀 NetricaAI: AI-Powered CCTV Monitoring & Smart Attendance System

<p align="center">
  <img src="https://img.shields.io/badge/AI%20Powered-Computer%20Vision-blue?style=for-the-badge&logo=robot" alt="AI Powered" />
  <img src="https://img.shields.io/badge/Framework-Flask%20%7C%20FastAPI-green?style=for-the-badge&logo=flask" alt="Framework" />
  <img src="https://img.shields.io/badge/Models-ArcFace%20%7C%20YOLOv8-orange?style=for-the-badge&logo=tensor-flow" alt="Models" />
  <img src="https://img.shields.io/badge/Database-SQL%20Server-red?style=for-the-badge&logo=microsoft-sql-server" alt="Database" />
  <img src="https://img.shields.io/github/stars/PythonMLClub/NetricaAI_CCTV-Monitoring-and-Smart-Attendance-System-?style=for-the-badge&logo=github" alt="GitHub Stars" />
  <img src="https://img.shields.io/github/forks/PythonMLClub/NetricaAI_CCTV-Monitoring-and-Smart-Attendance-System-?style=for-the-badge&logo=github" alt="GitHub Forks" />
</p>


NetricaAI is an advanced **AI-driven CCTV Monitoring & Smart Attendance System** built for enterprises, universities, and high-security environments. It automates employee attendance, enhances security through real-time surveillance, and provides intelligent insights using live video feeds from IP/CCTV/RTSP cameras.

This system combines **facial recognition, head-pose and liveness checks, object & crowd detection** to ensure accurate authentication and proactive monitoring. It supports real-time admin supervision, interactive dashboards, and historical log analysis. Built with Python, it leverages **Flask/FastAPI** for backend services, **OpenCV, MediaPipe, YOLO, Face Recognition** for AI processing, and **SQL Server** for secure data storage. A **Streamlit** interface handles employee registration and face embedding management.

The project aims to deliver a scalable, intelligent surveillance platform for automated attendance, safety monitoring, and centralized insights—making security and workforce management more efficient and reliable.

## 📌 Why NetricaAI? (Project Overview)
This qualifies as an **AI-based system** because it integrates machine learning and computer vision models for real-time perception, decision-making, and behavior analysis. Unlike traditional CCTV, it *understands*, *analyzes*, and *reacts* to human activities using advanced algorithms.

Key AI Capabilities:
- **Face Detection & Recognition**: Using deep learning models like ArcFace/FaceNet.
- **Liveness & Spoof Detection**: Differentiates real faces from photos/videos.
- **Behavior Tracking**: Monitors head pose, eye blinking, mouth movement, and object proximity.
- **Anomaly Detection**: Identifies suspicious audio activity and anomalies.
- **Object/Person Detection**: YOLO-based on live streams.
- **Event Logging & Alerts**: Generates intelligent real-time security alerts.

In essence, NetricaAI uses **Computer Vision + Deep Learning + Real-Time Streaming** to mimic human-like monitoring, transforming it into a true AI-driven security and attendance platform.

### Business Objectives
- Automate time & attendance using CCTV feeds (no physical biometric touchpoints).
- Improve security (real-time "who" and "where" tracking).
- Detect and record crowding events for EHS/operations.
- Provide searchable logs and summaries for HR & Admin.

### Key Stakeholders
- **HR**: Attendance and shift adherence.
- **Admin/Security**: Access & crowd monitoring.
- **IT/Infra**: Camera network, server, certificates.
- **Data/Analytics**: Trend reports.

### High-Level Scope
- Employee registration via webcam/photo (Streamlit) and bulk CSV+image ingestion.
- Real-time face detection & recognition from RTSP streams.
- Attendance logging with Entry/Exit logic, ROI/crowd zones, posture inference.
- Dashboards + APIs for logs, summaries, and crowd events.

## 🌟 Key Features
### 🔍 Real-Time Face Recognition
- **YOLOv8** for precise face detection.
- **ArcFace ONNX** for high-accuracy embedding.
- **MediaPipe** for liveness, alignment, and posture landmarks.
- **Cosine Similarity Matching** against stored embeddings.

### ⏱ Smart Attendance Automation
- Automatic Entry/Exit detection based on camera type (from `camera_locations.json`).
- Zero manual intervention with safeguards against duplicates/false entries.
- Logs to SQL Server with confidence scores and timestamps.
- Caching for last events and employee details.

### 🧍 Posture Detection
- Classifies as **Standing / Sitting / Unknown** using MediaPipe landmarks.
- Enhances behavior monitoring in crowds or attendance scenarios.

### 👥 Crowd Detection
- Detects groups of ≥3 people within configurable ROI zones.
- Tracks duration, posture summaries, and persistence (e.g., ≥180s triggers special logs).
- Auto-captures snapshots and logs to DB with JPEG embeddings.

### 📊 Dashboards & Reporting
- Interactive views for attendance logs, employee searches, crowd analytics.
- Live camera feeds with annotations (boxes, labels, FPS).
- Summary pages showing first Entry/last Exit per employee/day.
- Paginated APIs for filtered data (e.g., by date, camera, employee).

### 📸 Streamlit Employee Registration
- User-friendly UI for single registrations via webcam or photo upload.
- Auto-generates ArcFace embeddings and stores in SQL Server.
- Bulk processing from CSV + image folders with failure logging.

## 🏗️ System Architecture
```
RTSP CCTV Cameras
        │
        ▼
FFmpeg Stream Pulling (Auto-Reconnect)
        │
        ▼
YOLOv8 Face Detection
        │
        ▼
MediaPipe Alignment & Pose Detection
        │
        ▼
ArcFace Embedding & Cosine Matching
        │
        ├──▶ SQL Server (Employees, AttendanceLogs, CrowdDetection)
        └──▶ Live Stream Rendering (Flask MJPEG)
```

For detailed flow, see the [Project Flow Diagram](https://github.com/PythonMLClub/NetricaAI_CCTV-Monitoring-and-Smart-Attendance-System-/blob/main/Netrica_flow_diagram.svg).

## 📂 Project Structure
```
NetricaAI/
├── captured_crowds/          # Crowd snapshots (per camera)
├── captured_faces/           # Recognized face crops (per camera)
├── envface/models/           # ONNX & YOLO weights (arcface.onnx, yolov8m-face-lindevs.pt)
├── output_logs/              # Rotating CSVs & per-camera logs
├── static/                   # CSS/JS/assets
├── templates/                # Jinja2 HTML pages (e.g., dashboard.html, employee_logs.html)
├── utils/                    # Helpers (arcface_embedder.py, db_handler.py)
├── Employee_images/Employesbase/ # Bulk employee images
├── camera_locations.json     # Camera metadata (Location, Type: Entry/Exit)
├── cctv_app.py               # Main Flask app
├── face_register.py          # Streamlit registration
├── process_employee.py       # Bulk CSV+image ingestion
├── requirements.txt          # Dependencies
├── .env                      # Env vars (DB, RTSP creds)
└── Netrica_flow_diagram.svg  # Project flow SVG
```

## ⚙️ Installation Guide
### 1. Clone the Repository
```bash
git clone https://github.com/PythonMLClub/NetricaAI_CCTV-Monitoring-and-Smart-Attendance-System-.git
cd NetricaAI_CCTV-Monitoring-and-Smart-Attendance-System-
```

### 2. Create Virtual Environment
```bash
python -m venv envface
envface\Scripts\activate  # On Windows
source envface/bin/activate  # On Linux/macOS
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: Upgrade OpenCV if facing buffering issues: `pip install --upgrade opencv-python`.

### 4. Configure `.env`
```
DB_DRIVER=ODBC Driver 18 for SQL Server
DB_SERVER=****
DB_NAME=NetricaAI
DB_USERNAME=**
DB_PASSWORD=***
RTSP_USER=***
RTSP_PASSWORD=***
```

### 5. Install FFmpeg
- Download from [ffmpeg.org/download.html](https://ffmpeg.org/download.html).
- Add to PATH (e.g., `C:\ffmpeg\bin` on Windows).
- Verify: `ffmpeg -version` (should show version 7.1.1 or similar).

### 6. Run the Application
```bash
python cctv_app.py
```
- Access the UI at: [http://127.0.0.1:5004/](http://127.0.0.1:5004/)
- For Streamlit registration: `streamlit run face_register.py`
- Bulk processing: `python process_employee.py`

**Local URLs for Testing**:
- Logs: [http://127.0.0.1:5004/api/logs](http://127.0.0.1:5004/api/logs)
- Summary: [http://127.0.0.1:5004/attendance-summary](http://127.0.0.1:5004/attendance-summary)
- Crowd: [http://127.0.0.1:5004/crowd-detection](http://127.0.0.1:5004/crowd-detection)

## 🔌 API Endpoints
### 🎥 Camera Operations
| Method | Endpoint              | Description                  |
|--------|-----------------------|------------------------------|
| POST   | /api/start/<camera_id>| Start camera stream         |
| POST   | /api/stop/<camera_id> | Stop camera stream          |
| POST   | /api/start_all        | Start all cameras           |
| POST   | /api/stop_all         | Stop all cameras            |
| GET    | /api/status           | Camera health status        |
| POST   | /api/set_roi/<camera_id> | Set ROI for crowd detection |
| POST   | /api/reset_roi/<camera_id> | Reset ROI               |

### 📸 Live Video Streaming
| Method | Endpoint                  | Description             |
|--------|---------------------------|-------------------------|
| GET    | /api/video_feed/<camera_id>| Live MJPEG stream      |

### 📒 Logs & Attendance
| Method | Endpoint             | Description                  |
|--------|----------------------|------------------------------|
| GET    | /api/logs            | Paginated attendance logs   |
| GET    | /crowd-detection     | Crowd events                |
| GET    | /attendance-summary  | Daily entry-exit summary    |


## 🚀 Future Enhancements
- Auto Grouping & Crusher Operation.
- Guard Availability Tracking.
- ID Card Compliance Monitoring.
- Meal/Sleep & Mobile Usage Monitoring.
- Virtual Geofencing with Face Recognition.
- Trained Image Dashboard.
- Settings Page Configuration Backend.
- Enhanced Analytics & Trends.

## 🔧 Troubleshooting
- **FFmpeg Issues**: Ensure it's in PATH; reinstall if expired IE Tab (contact IT).
- **Stream Buffering**: Upgrade OpenCV.
- **DB Connection**: Verify `.env` creds; use SSMS for testing (Server: 103.117.172.65, User: sa, Pass: Tiger@1234*).
- **RTSP Testing**: Use VLC with provided credentials (e.g., rtsp://DataMonitor:D@taMon1tor@190.108.202.102:8554/...).
- **GPU Access**: Remote Desktop to 13.126.39.90:7379 (User: inc3098, Pass: Inc#$#3098).

## 📄 Database Schema
- **Employees**: EmployeeID (PK), FullName, Department, EmbeddingVector (VARBINARY), FaceImage (VARBINARY).
- **AttendanceLogs**: EmployeeID, Timestamp, CameraID, Location, ConfidenceScore, Status (Entry/Exit/-), Description.
- **CrowdDetection**: CrowdDetectionID (PK), CameraID, PeopleCount, DetectionTime, Duration, Posture, ImageEmbedding (VARBINARY).


## 🔗 Resources
- [Project Flow SVG](https://github.com/PythonMLClub/NetricaAI_CCTV-Monitoring-and-Smart-Attendance-System-/blob/main/Netrica_flow_diagram.svg)
- [GitHub Repo](https://github.com/PythonMLClub/NetricaAI_CCTV-Monitoring-and-Smart-Attendance-System-)

