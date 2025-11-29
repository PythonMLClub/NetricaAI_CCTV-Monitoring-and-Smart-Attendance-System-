from flask import Flask, render_template, Response, jsonify, request, send_from_directory, redirect, url_for, session
import pyodbc
import bcrypt
from flask_cors import CORS
import cv2 as cv
import numpy as np
import threading
import queue
import os
import logging
import sys
import pyodbc
from datetime import datetime, timedelta, date
import json
import uuid
import base64
import absl.logging
import multiprocessing
from ultralytics import YOLO
import onnxruntime as ort
import csv
import mediapipe as mp
import subprocess
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv


# ==================== GLOBAL APP LOGGER ====================
import logging
from logging.handlers import RotatingFileHandler

# Suppress MediaPipe warnings
absl.logging.set_verbosity(absl.logging.ERROR)

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Global app logger
app_logger = logging.getLogger("NetricaAI")
app_logger.setLevel(logging.INFO)

# Rotating file handler: 5 MB per file, keep 5 backups
handler = RotatingFileHandler(
    "logs/netrica.log",
    maxBytes=5_000_000,
    backupCount=5,
    encoding='utf-8'
)
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)

# Avoid duplicate handlers
if not app_logger.handlers:
    app_logger.handlers.clear()
    app_logger.addHandler(handler)

# Optional: Also log to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
app_logger.addHandler(console_handler)

app_logger.info("NetricaAI system initialized")
# ==========================================================

# Load environment variables
load_dotenv()

# Force UTF-8 for console output
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Initialize Flask app
app = Flask(__name__)

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-prod!')  # Secure fallback for dev; use env var in prod

CORS(app, resources={r"/*": {"origins": "*"}})  # Replace "*" with specific origins in production

# Create necessary directories
os.makedirs("output_logs", exist_ok=True)
os.makedirs("captured_faces", exist_ok=True)
os.makedirs("captured_crowds", exist_ok=True)

# Global caches with TTL
employee_details_cache = {}
last_attendance_cache = {}
last_event_cache = {}
CACHE_TTL = 60
EVENT_CACHE_TTL = 3600

# Global camera-to-location mapping
camera_location_map = {}

# Thread pool for async tasks
# 2 cores - 1 thread
# number of worker threads = half of your available CPU cores
executor = ThreadPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() // 2))

# Table names as constants
Employees_table = "Employees"
AttendanceLogs_table = "AttendanceLogs"
CrowdDetection_table = "CrowdDetection"

# Environment variables
rtsp_user = os.getenv('RTSP_USER')
rtsp_password = os.getenv('RTSP_PASSWORD')
rtsp_ip_m1 = os.getenv('RTSP_IP_M1')
rtsp_ip_m3 = os.getenv('RTSP_IP_M3')

# Global processors dictionary
processors = {}

# Standard ArcFace alignment template points
# Eyes aligned
# Nose centered
# Mouth level
# Same scale
# Same rotation
target_points = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

# Load known embeddings
known_embeddings = {}

def load_camera_locations(json_path="camera_locations.json"):
    global camera_location_map, entry_cameras, exit_cameras
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            camera_location_map = json.load(f)
        app_logger.info(f"Loaded {len(camera_location_map)} cameras from {json_path}")

        entry_cameras = [cam_id for cam_id, info in camera_location_map.items() if info.get("Type") == "Entry"]
        exit_cameras = [cam_id for cam_id, info in camera_location_map.items() if info.get("Type") == "Exit"]
        app_logger.info(f"Entry cameras: {len(entry_cameras)}, Exit cameras: {len(exit_cameras)}")
    except FileNotFoundError:
        app_logger.error(f"camera_locations.json not found at {json_path}")
        camera_location_map = {}
        entry_cameras = exit_cameras = []
    except Exception as e:
        app_logger.error(f"Failed to load camera locations: {e}")
        camera_location_map = {}
        entry_cameras = exit_cameras = []

# Database functions (same as before, omitted for brevity)
def get_db_connection():
    missing_vars = [v for v in ['DB_DRIVER', 'DB_SERVER', 'DB_NAME', 'DB_USERNAME', 'DB_PASSWORD'] if not os.getenv(v)]
    if missing_vars:
        app_logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    conn_str = (
        f"Driver={{{os.getenv('DB_DRIVER')}}};"
        f"Server={os.getenv('DB_SERVER')};"
        f"Database={os.getenv('DB_NAME')};"
        f"UID={os.getenv('DB_USERNAME')};"
        f"PWD={os.getenv('DB_PASSWORD')};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=30;"
    )

    app_logger.info("Attempting database connection...")
    app_logger.debug(f"Connection string: {conn_str.replace(os.getenv('DB_PASSWORD'), '***')}")

    try:
        conn = pyodbc.connect(conn_str)
        app_logger.info("Database connected successfully")
        return conn
    except Exception as e:
        app_logger.error(f"Database connection failed: {e}")
        raisee

def get_all_embeddings():
    logger = logging.getLogger(__name__)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT EmployeeID, EmbeddingVector FROM {Employees_table} WHERE EmbeddingVector IS NOT NULL")
        records = cursor.fetchall()
        logger.info(f"[OK] Fetched {len(records)} embeddings from database")
        return [(row[0], row[1]) for row in records]
    except Exception as e:
        logger.error(f"[ERROR] Database error: {e}")
        return []
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

def get_employee_details(emp_id):
    if emp_id in employee_details_cache:
        return employee_details_cache[emp_id]
    logger = logging.getLogger(__name__)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = f"SELECT FullName, Department FROM {Employees_table} WHERE EmployeeID = ?"
        cursor.execute(query, (emp_id,))
        result = cursor.fetchone()
        if result:
            logger.info(f"[OK] Fetched details for EmployeeID: {emp_id}")
            employee_details_cache[emp_id] = (result[0], result[1])
            return result[0], result[1]
        logger.warning(f"[WARNING] No details found for EmployeeID: {emp_id}")
        employee_details_cache[emp_id] = (None, None)
        return None, None
    except Exception as e:
        logger.error(f"[ERROR] Error fetching details for {emp_id}: {e}")
        return None, None
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()


def get_last_event_type(emp_id, camera_id):
    logger = logging.getLogger(__name__)
    cache_key = (emp_id, camera_id)
    current_time = time.time()
    if cache_key in last_event_cache and current_time - last_event_cache[cache_key][1] < EVENT_CACHE_TTL:
        return last_event_cache[cache_key][0]
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = f"""
            SELECT TOP 1 Status
            FROM {AttendanceLogs_table}
            WHERE EmployeeID = ? AND CameraID = ?
            ORDER BY Timestamp DESC
        """
        cursor.execute(query, (emp_id, camera_id))
        result = cursor.fetchone()
        event_type = result[0] if result else None
        last_event_cache[cache_key] = (event_type, current_time)
        logger.info(f"[OK] Fetched last event type for EmployeeID: {emp_id}, CameraID: {camera_id} - {event_type}")
        return event_type
    except Exception as e:
        logger.error(f"[ERROR] Error fetching last event type for {emp_id} on {camera_id}: {e}")
        return None
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

def log_attendance(emp_id, timestamp, camera_id, confidence_score, event_type="Entry"):
    logger = logging.getLogger(__name__)
    if event_type not in ["Entry", "Exit", "-"]:
        logger.error(f"[ERROR] Invalid event_type: {event_type}. Must be 'Entry', 'Exit', or '-'.")
        return False, f"Invalid event_type: {event_type}"
    
    # Fetch location and description from camera_location_map
    camera_info = camera_location_map.get(camera_id, {"Location": "TIDEL_PARK", "Description": "Unknown"})
    location = camera_info.get("Location")
    description = camera_info.get("Description")
    
    # Dynamically create entry and exit camera lists from camera_location_map
    entry_cameras = [cam_id for cam_id, info in camera_location_map.items() if info.get("Type") == "Entry"]
    exit_cameras = [cam_id for cam_id, info in camera_location_map.items() if info.get("Type") == "Exit"]

    # Dynamically create camera pairs (assuming Entry and Exit cameras share a common prefix or pattern)
    camera_pairs = {}
    for entry_cam in entry_cameras:
        # Find corresponding exit camera by replacing "Entry" with "Exit" in the camera ID
        exit_cam = entry_cam.replace("Entry", "Exit")
        if exit_cam in exit_cameras:
            camera_pairs[entry_cam] = exit_cam
            camera_pairs[exit_cam] = entry_cam
    
    if camera_id in exit_cameras and event_type == "Exit":
        corresponding_entry_camera = camera_pairs[camera_id]
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            query = f"""
                SELECT TOP 1 Status
                FROM {AttendanceLogs_table}
                WHERE EmployeeID = ? AND CameraID = ? AND Status = 'Entry'
                ORDER BY Timestamp DESC
            """
            cursor.execute(query, (emp_id, corresponding_entry_camera))
            result = cursor.fetchone()
            if not result:
                logger.info(f"[INFO] No corresponding entry found for EmployeeID: {emp_id} on {corresponding_entry_camera}. Logging as '-'.")
                event_type = "-"
        except Exception as e:
            logger.error(f"[ERROR] Error checking entry for {emp_id} on {corresponding_entry_camera}: {e}")
            event_type = "-"
        finally:
            if 'cursor' in locals(): cursor.close()
            if 'conn' in locals(): conn.close()
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        logger.info(f"[OK] Attempting to log attendance for EmployeeID: {emp_id}, Event: {event_type}, Location: {location}, Description: {description}")
        cursor.execute(f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{AttendanceLogs_table}'")
        if cursor.fetchone()[0] == 0:
            logger.error(f"[ERROR] {AttendanceLogs_table} table does not exist in the database")
            return False, f"{AttendanceLogs_table} table does not exist in the database"
        status = event_type
        created_at = datetime.now()
        query = f"""
            INSERT INTO {AttendanceLogs_table} (EmployeeID, Timestamp, CameraID, Location, ConfidenceScore, Status, CreatedAt, Description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, emp_id, timestamp, camera_id, location, confidence_score, status, created_at, description)
        conn.commit()
        logger.info(f"[OK] Successfully logged attendance for EmployeeID: {emp_id} as {event_type}")
        last_event_cache[(emp_id, camera_id)] = (event_type, time.time())
        return True, "Attendance logged successfully"
    except Exception as e:
        logger.error(f"[ERROR] Database error while logging attendance: {e}")
        return False, f"Database error: {e}"
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()


# Set up logging for a specific camera
def setup_logging(camera_id):
    log_filename = os.path.join("output_logs", f"face_detection_log_{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger(f"Camera_{camera_id}")
    logger.setLevel(logging.DEBUG)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(f"{camera_id}: %(asctime)s - %(levelname)s - %(message)s"))
    
    logger.handlers = [file_handler, console_handler]
    return logger

# Posture detection (same as before, omitted for brevity)
def detect_posture(landmarks, frame_height, threshold=0.15):
    logger = logging.getLogger(__name__)
    try:
        mp_pose = mp.solutions.pose
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        hip_y = (left_hip.y + right_hip.y) / 2 * frame_height
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * frame_height
        shoulder_to_hip = hip_y - shoulder_y
        knee_visible = (left_knee.visibility > 0.5 and right_knee.visibility > 0.5 and
                        left_knee.x != 0 and right_knee.x != 0 and
                        left_knee.y != 0 and right_knee.y != 0)
        if not knee_visible:
            if hip_y > 0.6 * frame_height and abs(shoulder_to_hip) < 0.6 * frame_height:
                logger.info("[OK] Posture detected: Sitting (inferred)")
                return "Sitting"
            return "Unknown"
        knee_y = (left_knee.y + right_knee.y) / 2 * frame_height
        ankle_y = (left_ankle.y + right_ankle.y) / 2 * frame_height
        def calculate_angle(p1, p2, p3):
            v1 = np.array([p1.x * frame_height - p2.x * frame_height, p1.y * frame_height - p2.y * frame_height])
            v2 = np.array([p3.x * frame_height - p2.x * frame_height, p3.y * frame_height - p2.y * frame_height])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
            return angle
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        knee_angle = (left_knee_angle + right_knee_angle) / 2
        if knee_y > hip_y and knee_angle < 150 and abs(shoulder_to_hip) < 0.6 * frame_height:
            logger.info("[OK] Posture detected: Sitting")
            return "Sitting"
        elif shoulder_to_hip > 0.5 * frame_height:
            logger.info("[OK] Posture detected: Standing")
            return "Standing"
        return "Unknown"
    except Exception as e:
        logger.warning(f"[WARNING] Error detecting posture: {e}")
        return "Unknown"

# Face alignment and preprocessing (same as before, omitted for brevity)
def align_and_preprocess_face(face_crop, face_mesh, logger):
    if face_crop is None or face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
        logger.warning("[WARNING] Invalid face crop; skipping alignment and resizing")
        if face_crop is not None and face_crop.size > 0:
            return cv.resize(face_crop, (112, 112))
        return None
    try:
        results = face_mesh.process(cv.cvtColor(face_crop, cv.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            src_points = np.array([(lm.x * face_crop.shape[1], lm.y * face_crop.shape[0])
                                  for lm in landmarks.landmark
                                  if lm.x > 0 and lm.y > 0 and lm.visibility > 0.5])
            if len(src_points) >= 3:
                tform = cv.getAffineTransform(src_points[:3], target_points[:3])
                aligned_img = cv.warpAffine(face_crop, tform, (112, 112))
                return aligned_img
            else:
                logger.warning(f"[WARNING] Insufficient valid landmarks ({len(src_points)} < 3); using direct resize")
        else:
            logger.warning("[WARNING] No face landmarks detected; using direct resize")
        return cv.resize(face_crop, (112, 112))
    except Exception as e:
        logger.error(f"[ERROR] MediaPipe FaceMesh processing failed: {e}")
        return cv.resize(face_crop, (112, 112))

# Get embedding (same as before, omitted for brevity)
def get_embedding(face_crop, rec_session, rec_input_name, rec_output_name, face_mesh, logger):
    # Calls your earlier align_and_preprocess_face() to correct rotation and resize the face to 112×112.
    # If alignment fails (e.g., face too small, no landmarks) → returns a zero vector (512 zeros).
    # This avoids breaking later code that expects a numeric embedding.
    aligned_img = align_and_preprocess_face(face_crop, face_mesh, logger)
    if aligned_img is None:
        logger.error("[ERROR] Failed to align face; returning zero embedding")
        return np.zeros(512, dtype=np.float32)
    
    # Normalize pixel values
    aligned_img = (aligned_img - 127.5) / 127.5
    # Reorder axes to match ArcFace format
    aligned_img = np.transpose(aligned_img, (2, 0, 1))
    # Add batch dimension & ensure correct dtype
    # Model input shape = (1, 3, 112, 112) → 1 = batch size.
    # Converts to 32-bit floats for ONNX Runtime.
    aligned_img = np.expand_dims(aligned_img, axis=0).astype(np.float32)

    # Output = the unique numerical signature of that face.
    embedding = rec_session.run([rec_output_name], {rec_input_name: aligned_img})[0]

    # Normalize embedding (L2 normalization)
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    # Removes the extra batch dimension → shape (512,).
    # Easier to store in DB or compare with other embeddings.
    return embedding.flatten()

# Cosine similarity (same as before, omitted for brevity)
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2)

# Frame capture with FFmpeg (same as before, omitted for brevity)
def capture_frames(video_source, frame_queue, stop_event, width, height, camera_id):
    logger = logging.getLogger(f"Camera_{camera_id}")
    logger.info(f"Starting FFmpeg capture thread for source: {video_source} at {width}x{height}")
    while not stop_event.is_set():
        cmd = [
            "ffmpeg",  # CORRECT on Linux/Ubuntu
            "-re",
            "-rtsp_transport", "tcp",
            "-i", video_source,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"scale={width}:{height}",
            "-r", "12",
            "-reconnect", "1",
            "-reconnect_at_eof", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "10",
            "-timeout", "30000000",
            "-"
        ]
        pipe = None
        attempt = 0
        max_attempts = 5
        while not stop_event.is_set() and attempt < max_attempts:
            try:
                pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=width * height * 3 * 2)
                expected_size = width * height * 3
                last_frame_time = time.time()
                while not stop_event.is_set():
                    raw = pipe.stdout.read(expected_size)
                    if len(raw) != expected_size:
                        logger.warning(f"[WARNING] Incomplete frame data, size {len(raw)} vs expected {expected_size}")
                        break
                    frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
                    if frame is None or frame.size == 0:
                        logger.warning("[WARNING] Invalid frame received from FFmpeg")
                        continue
                    try:
                        frame_queue.put_nowait(frame)
                        last_frame_time = time.time()
                    except queue.Full:
                        try:
                            frame_queue.get_nowait()
                            frame_queue.put_nowait(frame)
                        except queue.Empty:
                            pass
                    if time.time() - last_frame_time > 10:
                        logger.warning("[WARNING] No frames received for 10 seconds, restarting FFmpeg")
                        break
                attempt += 1
                if attempt < max_attempts:
                    logger.info(f"Restarting FFmpeg capture (attempt {attempt}/{max_attempts})")
                    time.sleep(2)
            except Exception as e:
                logger.error(f"[ERROR] Capture thread crashed: {e}")
                attempt += 1
                if attempt < max_attempts:
                    logger.info(f"Restarting FFmpeg capture (attempt {attempt}/{max_attempts}) after crash")
                    time.sleep(2)
            finally:
                if pipe:
                    pipe.terminate()
                    try:
                        pipe.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pipe.kill()
        if attempt >= max_attempts:
            logger.error(f"[ERROR] Failed to restart FFmpeg after {max_attempts} attempts, stopping thread")

class CameraProcessor:
    def __init__(self, url, camera_id):
        self.url = url
        self.camera_id = camera_id
        self.location = camera_location_map.get(camera_id, {"Location": "TIDEL_PARK", "Description": "Unknown"})["Location"]
        self.frame_queue = queue.Queue(maxsize=100)
        self.stop_thread = threading.Event()
        self.roi_defined = False
        self.roi_start = None
        self.roi_end = None
        self.roi = None
        self.drawing = False
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_pos = None
        self.width = 1280
        self.height = 720
        self.csv_logs = []
        self.logger = setup_logging(camera_id)
        self.out = None
        self.capture_thread = None
        self.frame_skip = 1
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.individuals = {}
        self.pose_frame_skip = 2
        self.pose_frame_count = 0
        self.zones = {
            'Zone1': {
                'roi': None,
                'max_count': 10,
                'current_count': 0,
                'last_count': 0,
                'standing_start': {},
                'crowd_groups': {},
                'crowd_lines': []
            }
        }
        self.alert_triggered = False
        self.crowd_start_time = None
        self.standing_duration_threshold = 60
        self.current_frame = None
        self.current_frame_lock = threading.Lock()
        
        # Initialize MediaPipe instances PER CAMERA
        self.logger.info(f"[OK] Initializing MediaPipe for {camera_id}")
        mp_face_mesh = mp.solutions.face_mesh

        # self.face_mesh – for face landmarks (used during alignment).
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5
        )
        
        # self.pose – for body pose & posture detection (Standing / Sitting).
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.logger.info(f"[OK] MediaPipe initialized for {camera_id}")
        
        self.logger.info(f"[OK] Loading YOLOv8 face detection model for {camera_id}")
        try:
            # Load YOLOv8 face detector
            self.det_model = YOLO("models/yolov8m-face-lindevs.pt")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.det_model.to(device)
            self.logger.info(f"[OK] YOLOv8 face detection model loaded on {device}")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load YOLOv8 face detection model: {e}")
            raise
        
        self.logger.info(f"[OK] Loading ArcFace ONNX model for {camera_id}")
        try:
            # Load ArcFace ONNX model (face recognition)
            rec_model_path = "models/arcface.onnx"
            providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.rec_session = ort.InferenceSession(rec_model_path, providers=providers)
            self.rec_input_name = self.rec_session.get_inputs()[0].name
            self.rec_output_name = self.rec_session.get_outputs()[0].name
            self.logger.info(f"[OK] ArcFace ONNX model loaded on {providers[0]}")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load ArcFace ONNX model: {e}")
            raise

    def initialize_stream(self):
        self.logger.info(f"Initializing FFmpeg stream for: {self.url}")
        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            try:
                cap = cv.VideoCapture(self.url)
                if not cap.isOpened():
                    self.logger.error("[ERROR] Cannot open video source with OpenCV, using default 1280x720")
                    self.width, self.height = 1280, 720
                    fps = 12
                else:
                    self.width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) or 1280
                    self.height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) or 720
                    fps = int(cap.get(cv.CAP_PROP_FPS)) or 12
                    cap.release()
                    self.logger.info(f"Detected resolution: {self.width}x{self.height}, FPS: {fps}")
                self.capture_thread = threading.Thread(target=capture_frames, args=(self.url, self.frame_queue, self.stop_thread, self.width, self.height, self.camera_id))
                self.capture_thread.daemon = True
                self.capture_thread.start()
                self.logger.info(f"[OK] FFmpeg capture thread started at {fps} fps")
                return fps, True
            except Exception as e:
                attempt += 1
                self.logger.error(f"[ERROR] Failed to initialize stream (attempt {attempt}/{max_attempts}): {e}")
                if attempt == max_attempts:
                    return None, False
                time.sleep(5)

    def async_save_face_capture(self, face_crop, emp_id, timestamp, log_entry):
        def wrapper():
            success = self.save_face_capture(face_crop, emp_id, timestamp)
            if success:
                log_entry["FaceCaptured"] = True
                log_entry["CapturePath"] = f"captured_faces/{self.camera_id}/{emp_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            else:
                log_entry["FaceCaptured"] = False
        executor.submit(wrapper)

    def save_face_capture(self, face_crop, emp_id, timestamp):
        capture_dir = os.path.join("captured_faces", self.camera_id)
        os.makedirs(capture_dir, exist_ok=True)
        filename = os.path.join(capture_dir, f"{emp_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg")
        try:
            cv.imwrite(filename, face_crop)
            self.logger.info(f"[OK] Saved full-size face capture for {emp_id} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to save full-size face capture: {e}")
            return False

    def save_crowd_capture(self, frame, timestamp, people_count, zone_id, group_id=""):
        capture_dir = os.path.join("captured_crowds", self.camera_id)
        os.makedirs(capture_dir, exist_ok=True)
        filename = os.path.join(capture_dir, f"crowd_{self.camera_id}_{zone_id}_{group_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{people_count}.jpg")
        try:
            cv.imwrite(filename, frame)
            self.logger.info(f"[OK] Saved crowd capture to {filename}")
            return True, frame
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to save crowd capture: {e}")
            return False, None


    def calculate_proximity_and_groups(self, boxes, original_indices, postures):
        """
        FIXED: Distance-based crowd detection with proper grouping
        - Removed strict posture matching (optional)
        - Increased distance threshold to 200px
        - Added comprehensive logging
        - Fixed index handling
        """
        # Minimum 3 people for crowd
        if len(boxes) < 3:
            self.logger.debug(f"[DEBUG] Only {len(boxes)} people detected - need 3+ for crowd")
            return False, []
        
        # Validate input lengths
        if len(boxes) != len(postures) or len(boxes) != len(original_indices):
            self.logger.warning(
                f"[WARNING] Length mismatch: boxes={len(boxes)}, "
                f"postures={len(postures)}, indices={len(original_indices)}"
            )
            return False, []

        # Calculate centers with validation
        centers = []
        for i, ((x1, y1, x2, y2), posture) in enumerate(zip(boxes, postures)):
            if x1 < x2 and y1 < y2:  # Valid box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centers.append((cx, cy, posture))
                self.logger.debug(f"[DEBUG] Person {i}: center=({cx},{cy}), posture={posture}")
            else:
                self.logger.warning(f"[WARNING] Invalid box {i}: ({x1},{y1},{x2},{y2})")
        
        if len(centers) < 3:
            self.logger.debug(f"[DEBUG] Only {len(centers)} valid centers - need 3+")
            return False, []

        # Build adjacency graph based on distance
        adjacency = [set() for _ in range(len(centers))]
        proximity_threshold = 200  # pixels - INCREASED from 150
        
        self.logger.debug(f"[DEBUG] Building adjacency graph with threshold={proximity_threshold}px")
        
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                x1, y1, posture1 = centers[i]
                x2, y2, posture2 = centers[j]
                
                # Calculate Euclidean distance
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                
                # OPTION 1: Posture-agnostic (RECOMMENDED for better detection)
                if distance < proximity_threshold:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
                    self.logger.debug(
                        f"[DEBUG] Connected person {i}<->person {j}: "
                        f"distance={distance:.1f}px, postures={posture1}/{posture2}"
                    )
                

        # Count connections
        total_connections = sum(len(adj) for adj in adjacency)
        self.logger.debug(f"[DEBUG] Total connections formed: {total_connections}")

        # Find connected components using DFS (Depth-First Search)

        # Depth-First Search (DFS) is used because it is memory-efficient, simple to implement, and effective for problems where a solution might be found deep in a graph
        visited = set()
        groups = []
        
        def dfs(node, group):
            visited.add(node)
            group.append(original_indices[node])
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    dfs(neighbor, group)

        # Find all groups
        for i in range(len(centers)):
            if i not in visited and len(adjacency[i]) > 0:
                group = []
                dfs(i, group)
                if len(group) >= 3:  # Minimum 3 people for crowd
                    groups.append(group)
                    self.logger.info(
                        f"[OK] Found crowd group with {len(group)} people: {group}"
                    )

        has_crowd = len(groups) > 0
        
        if has_crowd:
            self.logger.info(f"[CROWD DETECTED] {len(groups)} group(s) found")
        else:
            self.logger.debug("[DEBUG] No crowds detected (no groups with 3+ people)")
        
        return has_crowd, groups

    def draw_crowd_lines(self, frame, boxes, group_indices, color=(0, 0, 255), is_static=False):
        """
        FIXED: Draw connecting lines between crowd members
        - Added bounds checking
        - Different colors for static vs dynamic crowds
        - Shows distances on lines
        """
        if not group_indices or len(group_indices) < 3:
            return
        
        # Validate member indices (direcation)
        valid_indices = [i for i in group_indices if 0 <= i < len(boxes)]
        if len(valid_indices) < 3:
            self.logger.warning(
                f"[WARNING] Insufficient valid indices: {len(valid_indices)}"
            )
            return
        
        try:
            # Compute centers for each face
            centers = []
            for i in valid_indices:
                box = boxes[i]
                # cx = midpoint (left + right) / 2
                # cy = midpoint (top + bottom) / 2
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                centers.append((cx, cy))
            
            # Choose color: Purple for static, Orange for dynamic
            line_color = (255, 0, 255) if is_static else (0, 165, 255)
            
            # Draw lines between all pairs
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    # Draw line
                    # his creates fully connected graph:
                    # If 4 members A, B, C, D:
                    # You get lines:
                    # A-B
                    # A-C
                    # A-D
                    # B-C
                    # B-D
                    # C-D
                    # This makes it visually clear that these people are part of a crowd cluster.

                    cv.line(frame, centers[i], centers[j], line_color, 2)
                    
                    # Calculate distance
                    # This uses Euclidean distance formula.
                    distance = np.sqrt(
                        (centers[i][0] - centers[j][0]) ** 2 + 
                        (centers[i][1] - centers[j][1]) ** 2
                    )
                    
                    # Draw distance label at midpoint
                    mid_x = (centers[i][0] + centers[j][0]) // 2
                    mid_y = (centers[i][1] + centers[j][1]) // 2
                    
                    cv.putText(
                        frame,
                        f"{int(distance)}px",
                        (mid_x, mid_y),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        line_color,
                        1
                    )
        
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to draw crowd lines: {e}")

    def process_face(self, face_crop, box, conf, frame_counter, i, posture, full_frame):
        try:
            x1, y1, x2, y2 = map(int, box)
            full_region = full_frame[y1:y2, x1:x2]
            if full_region.size == 0:
                self.logger.error(f"[ERROR] Invalid full region: shape={full_region.shape if full_region is not None else None}, box={box}")
                return None
            self.logger.debug(f"[DEBUG] Processing full region: shape={full_region.shape}, box={box}, conf={conf}")

            embedding = get_embedding(face_crop, self.rec_session, self.rec_input_name,
                                    self.rec_output_name, self.face_mesh, self.logger)
            name = "Unknown"
            max_sim = 0.4
            emp_id_match = None
            for emp_id, known_emb in known_embeddings.items():
                sim = cosine_similarity(embedding, known_emb)
                if sim > max_sim:
                    max_sim = sim
                    name = emp_id
                    emp_id_match = emp_id
            self.logger.info(f"Max similarity for face: {max_sim}")

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            full_name, department = get_employee_details(emp_id_match) if emp_id_match else ("Unknown", "Unknown")
            if not full_name and emp_id_match:
                full_name = emp_id_match
            log_entry = {
                "Frame": frame_counter,
                "TrackID": i,
                "EmployeeID": emp_id_match if emp_id_match else "Unknown",
                "Name": full_name,
                "Department": department,
                "CenterX": center_x,
                "CenterY": center_y,
                "FaceDetected": True,
                "FaceMatched": emp_id_match is not None,
                "MaxSimilarity": max_sim,
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "EventType": None,
                "FaceCaptured": False,
                "CapturePath": None,
                "CrowdDetected": False,
                "CrowdSize": 0,
                "StandingDuration": 0,
                "Posture": posture
            }
            self.csv_logs.append(log_entry)
            if emp_id_match:
                timestamp = datetime.now()
                conf_float = float(conf)
                # Initialize entry and exit cameras dynamically
                entry_cameras = [cam_id for cam_id, info in camera_location_map.items() if info.get("Type") == "Entry"]
                exit_cameras = [cam_id for cam_id, info in camera_location_map.items() if info.get("Type") == "Exit"]

                if self.camera_id in entry_cameras:
                    event_type = "Entry"
                    self.logger.info(f"Person detected on {self.camera_id}, logging as Entry")
                elif self.camera_id in exit_cameras:
                    event_type = "Exit"
                    self.logger.info(f"Person detected on {self.camera_id}, checking for corresponding Entry")
                else:
                    def attendance_callback(last_event):
                        if last_event == "Entry":
                            event_type = "Exit"
                        else:
                            event_type = "Entry"
                        log_entry["EventType"] = event_type
                        self.logger.info(f"Person detected on {self.camera_id}, logging as {event_type} (last event: {last_event})")
                        def log_callback(result):
                            success, msg = result
                            if success:
                                self.logger.info(f"[OK] Attendance logged for {full_name} as {event_type}")
                                self.async_save_face_capture(full_region, emp_id_match, timestamp, log_entry)
                            else:
                                self.logger.warning(f"[WARNING] Attendance log failed for {full_name}: {msg}")
                                log_entry["EventType"] = "-"
                                self.csv_logs[-1]["EventType"] = "-"
                                self.logger.info(f"[OK] Recorded failed attendance attempt for {full_name} as '-'")
                        executor.submit(log_attendance, emp_id_match, timestamp, self.camera_id, conf_float, event_type).add_done_callback(lambda fut: log_callback(fut.result()))
                    executor.submit(get_last_event_type, emp_id_match, self.camera_id).add_done_callback(lambda fut: attendance_callback(fut.result()))
                    return x1, y1, x2, y2, full_name, conf, posture
                log_entry["EventType"] = event_type
                def log_callback(result):
                    success, msg = result
                    if success:
                        self.logger.info(f"[OK] Attendance logged for {full_name} as {event_type}")
                        self.async_save_face_capture(full_region, emp_id_match, timestamp, log_entry)
                    else:
                        self.logger.warning(f"[WARNING] Attendance log failed for {full_name}: {msg}")
                        log_entry["EventType"] = "-"
                        self.csv_logs[-1]["EventType"] = "-"
                        self.logger.info(f"[OK] Recorded failed attendance attempt for {full_name} as '-'")
                executor.submit(log_attendance, emp_id_match, timestamp, self.camera_id, conf_float, event_type).add_done_callback(lambda fut: log_callback(fut.result()))
            return x1, y1, x2, y2, full_name, conf, posture
        except Exception as e:
            self.logger.error(f"[ERROR] Error processing face: {e}")
            return None

    def log_crowd_event(self, people_count, detection_time, duration, posture_counts, frame):
        logger = self.logger
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            camera_info = camera_location_map.get(self.camera_id, {})
            location = camera_info.get("Location")
            description = camera_info.get("Description")
            
            if location is None or description is None:
                logger.warning(f"[WARNING] No location or description found for CameraID: {self.camera_id}")
            
            detection_time_str = detection_time.strftime('%Y-%m-%d %H:%M:%S')
            detection_date_str = detection_time.date().strftime('%Y-%m-%d')
            
            posture_summary = f"Standing: {posture_counts.get('Standing', 0)}, Sitting: {posture_counts.get('Sitting', 0)}"
            
            success, buffer = cv.imencode('.jpg', frame)
            
            if success:
                image_binary = buffer.tobytes()
                query = """
                    INSERT INTO [dbo].[CrowdDetection] (CameraID, Location, Description, PeopleCount, DetectionTime, Duration, DetectionDate, Posture, ImageEmbedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(query, (
                    self.camera_id,
                    location,
                    description,
                    people_count,
                    detection_time_str,
                    duration,
                    detection_date_str,
                    posture_summary,
                    image_binary
                ))
            else:
                logger.error(f"[ERROR] Failed to encode image for database")
                query = """
                    INSERT INTO [dbo].[CrowdDetection] (CameraID, Location, Description, PeopleCount, DetectionTime, Duration, DetectionDate, Posture)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(query, (
                    self.camera_id,
                    location,
                    description,
                    people_count,
                    detection_time_str,
                    duration,
                    detection_date_str,
                    posture_summary
                ))
            
            conn.commit()
            logger.info(f"[OK] Logged crowd event for CameraID: {self.camera_id}, People: {people_count}, Duration: {duration}s")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to log crowd event: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            if 'cursor' in locals(): cursor.close()
            if 'conn' in locals(): conn.close()

    def get_frame(self):
        with self.current_frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None


    def process(self):
        """
        FIXED: Main processing loop with corrected crowd detection logic
        """
        self.logger.info(f"Starting video processing for {self.camera_id}")
        input_fps, success = self.initialize_stream()
        if not success or input_fps is None:
            self.logger.error("[ERROR] Stream initialization failed")
            return
        
        input_fps = min(input_fps, 30) if input_fps else 30
        
        log_csv_path = os.path.join(
            "output_logs",
            f"face_detection_log_{self.camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        
        frame_counter = 0
        faces_detected_count = 0
        crowd_detected_count = 0
        start_time = datetime.now()
        
        # This runs forever until you call /api/stop/<camera_id>.
        while not self.stop_thread.is_set():
            # Get frame from queue
            try:
                if self.frame_queue.empty():
                    time.sleep(0.1)
                    continue
                # Avoids blocking the camera , The processing thread does not read RTSP directly.
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"[ERROR] Error retrieving frame: {e}")
                continue
            
            # Frame skipping
            self.frame_count += 1
            # If frame_skip = 3, you process 1 frame, skip 2 frames.
            if self.frame_count % self.frame_skip != 0:
                continue
            
            frame_counter += 1
            frame_for_saving = frame.copy()
            
            # Convert to RGB
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # STAGE 6: FACE DETECTION (YOLO)
            try:
                face_results = self.det_model.predict(
                    rgb_frame,
                    conf=0.4,
                    iou=0.45,
                    max_det=20
                )
            except Exception as e:
                self.logger.error(f"[ERROR] Face detection failed: {e}")
                continue
            
            # STAGE 7: POSTURE DETECTION (MediaPipe Pose)
            self.pose_frame_count += 1
            if self.pose_frame_count % self.pose_frame_skip == 0:
                try:
                    # Standing, Sitting, Unknown
                    pose_results = self.pose.process(rgb_frame)
                except Exception as e:
                    self.logger.error(f"[ERROR] Pose detection failed: {e}")
                    pose_results = None
            else:
                pose_results = None
            
            # Initialize tracking variables
            current_frame_faces = 0
            annotations = []
            all_boxes = []
            self.individuals.clear()
            
            # Get postures for all detected faces
            # list of face bounding boxes detected by YOLO.
            all_postures = ["Unknown"] * len(face_results[0].boxes.xyxy)
            for i, (box, conf) in enumerate(zip(
                face_results[0].boxes.xyxy.cpu().numpy(),
                face_results[0].boxes.conf.cpu().numpy()
            )):
                posture = "Unknown"
                if pose_results and pose_results.pose_landmarks:
                    posture = detect_posture(pose_results.pose_landmarks, frame.shape[0])
                all_postures[i] = posture
            
            # Process each zone
            for zone_name, zone in self.zones.items():
                zone['current_count'] = 0
                zone_boxes = []
                zone_indices = []
                postures = []
                posture_counts = {'Standing': 0, 'Sitting': 0, 'Unknown': 0}
                
                # Process detections in zone
                for i, (box, conf) in enumerate(zip(
                    face_results[0].boxes.xyxy.cpu().numpy(),
                    face_results[0].boxes.conf.cpu().numpy()
                )):
                    x1, y1, x2, y2 = map(int, box)
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    all_boxes.append(box)
                    
                    # Check if detection is in ROI
                    # zone['roi'] is your manually set rectangle via /api/set_roi/<camera_id>.
                    if zone['roi']:
                        roi_x1, roi_y1, roi_x2, roi_y2 = zone['roi']
                        if (x1 >= roi_x1 and x2 <= roi_x2 and 
                            y1 >= roi_y1 and y2 <= roi_y2):
                            
                            posture = all_postures[i]
                            zone['current_count'] += 1
                            zone_boxes.append(box)
                            zone_indices.append(i)
                            postures.append(posture)
                            posture_counts[posture] += 1
                            current_frame_faces += 1
                            
                            # Process face for recognition "process_face"
                            face_crop = rgb_frame[y1:y2, x1:x2]
                            if face_crop.size > 0:
                                result = self.process_face(
                                    face_crop, box, conf,
                                    frame_counter, i, posture, frame
                                )
                                if result:
                                    annotations.append((result, i))
                
                # === CROWD DETECTION SECTION ===
                if zone['roi'] and len(zone['roi']) == 4:
                    roi_x1, roi_y1, roi_x2, roi_y2 = map(int, zone['roi'])
                    cv.rectangle(
                        frame_for_saving,
                        (roi_x1, roi_y1),
                        (roi_x2, roi_y2),
                        (0, 255, 255),
                        4
                    )
                    
                    current_count = zone['current_count']
                    current_time = datetime.now()
                    
                    self.logger.debug(
                        f"[DEBUG] {zone_name}: {current_count} people in ROI"
                    )
                    
                    # Calculate proximity and find groups
                    has_crowd, crowd_groups = self.calculate_proximity_and_groups(
                        zone_boxes,  # all faces inside that ROI.
                        zone_indices,
                        postures
                    )
                    
                    # Handling each crowd group (NEW vs EXISTING)
                    if has_crowd and crowd_groups:
                        self.logger.info(
                            f"[ALERT] Crowd detected in {zone_name}: "
                            f"{len(crowd_groups)} group(s)"
                        )
                        
                        # Process each crowd group
                        for group_idx, group in enumerate(crowd_groups):
                            # Create stable group ID based on member positions
                            group_sorted = sorted(group)
                            group_id = f"{zone_name}_group_{hash(tuple(group_sorted)) % 1000}"
                            group_size = len(group)
                            
                            self.logger.debug(
                                f"[DEBUG] Processing group {group_id} with {group_size} members"
                            )
                            
                            # NEW GROUP DETECTED
                            if group_id not in zone['crowd_groups']:
                                # Calculate initial positions
                                initial_positions = {}
                                for member_idx in group:
                                    if member_idx < len(all_boxes):
                                        box = all_boxes[member_idx]
                                        cx = int((box[0] + box[2]) / 2)
                                        cy = int((box[1] + box[3]) / 2)
                                        initial_positions[member_idx] = (cx, cy)
                                
                                zone['crowd_groups'][group_id] = {
                                    'start_time': current_time,
                                    'last_seen': current_time,
                                    'members': group,
                                    'captured': False,
                                    'last_positions': initial_positions,
                                    'is_static': False
                                }
                                
                                self.logger.warning(
                                    f"[ALERT] NEW CROWD GROUP: {group_id} "
                                    f"with {group_size} people at {current_time.strftime('%H:%M:%S')}"
                                )
                                
                                # Save initial capture
                                executor.submit(
                                    self.save_crowd_capture,
                                    frame_for_saving,
                                    current_time,
                                    group_size,
                                    zone_name,
                                    group_id
                                )
                                
                                # Log to database
                                executor.submit(
                                    self.log_crowd_event,
                                    group_size,
                                    current_time,
                                    0,
                                    posture_counts,
                                    frame_for_saving
                                )
                                
                                crowd_detected_count += 1
                            
                            # EXISTING GROUP - UPDATE
                            else:
                                group_info = zone['crowd_groups'][group_id]
                                # Meaning:
                                # The group is still alive (they still appear in the frame)
                                # Update the list of member indices (people in this group)
                                group_info['last_seen'] = current_time
                                group_info['members'] = group
                                
                                # Calculate current positions of all members
                                current_positions = {}
                                for member_idx in group:
                                    if member_idx < len(all_boxes):
                                        box = all_boxes[member_idx]
                                        cx = int((box[0] + box[2]) / 2)
                                        cy = int((box[1] + box[3]) / 2)
                                        current_positions[member_idx] = (cx, cy)
                                
                                # Check if group is static (hasn't moved much)
                                # ✔ If people move less than 50 pixels, they are considered standing still
                                # Measure movement vs last_positions
                                # Euclidean distance in pixels
                                movement_threshold = 50  # pixels
                                total_movement = 0
                                movement_count = 0
                                
                                for member_idx in group:
                                    if (member_idx in group_info['last_positions'] and
                                        member_idx in current_positions):
                                        
                                        old_pos = group_info['last_positions'][member_idx]
                                        new_pos = current_positions[member_idx]
                                        
                                        # This is basic distance formula.
                                        movement = np.sqrt(
                                            (old_pos[0] - new_pos[0])**2 +
                                            (old_pos[1] - new_pos[1])**2
                                        )
                                        
                                        total_movement += movement
                                        movement_count += 1
                                
                                # Average movement
                                avg_movement = (total_movement / movement_count 
                                            if movement_count > 0 else 0)
                                
                                # ✔ If people move less than 50 pixels, they are considered standing still
                                # ✔ Saves the latest positions for next frame comparison
                                is_static = avg_movement < movement_threshold
                                group_info['is_static'] = is_static
                                group_info['last_positions'] = current_positions
                                
                                # Calculate duration
                                # How long the crowd existed
                                # Compute duration in seconds since start_time
                                duration = int(
                                    (current_time - group_info['start_time']).total_seconds()
                                )
                                
                                self.logger.debug(
                                    f"[DEBUG] Group {group_id}: duration={duration}s, "
                                    f"static={is_static}, avg_movement={avg_movement:.1f}px"
                                )
                                
                                # Draw crowd lines between group members
                                self.draw_crowd_lines(
                                    frame_for_saving,
                                    all_boxes,
                                    group,
                                    is_static=is_static
                                )
                                
                                # TRIGGER 3-MINUTE ALERT
                                if (duration >= self.standing_duration_threshold and
                                    not group_info['captured']):
                                    
                                    self.logger.warning(
                                        f"[ALERT] Crowd group {group_id} persistent "
                                        f"for {duration}s (threshold: {self.standing_duration_threshold}s)!"
                                    )
                                    
                                    # Save special 3-minute capture
                                    executor.submit(
                                        self.save_crowd_capture,
                                        frame_for_saving,
                                        current_time,
                                        group_size,
                                        zone_name,
                                        f"{group_id}_3min"
                                    )
                                    
                                    # Log to database with full duration
                                    executor.submit(
                                        self.log_crowd_event,
                                        group_size,
                                        group_info['start_time'],
                                        duration,
                                        posture_counts,
                                        frame_for_saving
                                    )
                                    
                                    group_info['captured'] = True
                                    crowd_detected_count += 1
                    
                    # CLEANUP DISPERSED GROUPS
                    groups_to_remove = []
                    for group_id, group_info in zone['crowd_groups'].items():
                        time_since_seen = (
                            current_time - group_info['last_seen']
                        ).total_seconds()
                        
                        # This checks:
                        # ✔ Has the group been missing for more than 10 seconds?
                        # ✔ If yes → assume they left → close the group.
                        if time_since_seen > 10:  # 10 seconds timeout
                            duration = int(
                                (group_info['last_seen'] - group_info['start_time']).total_seconds()
                            )
                            
                            self.logger.info(
                                f"[OK] Crowd group {group_id} dispersed "
                                f"after {duration}s total duration"
                            )
                            
                            # Final log entry
                            executor.submit(
                                self.log_crowd_event,
                                len(group_info['members']),
                                group_info['start_time'],
                                duration,
                                posture_counts,
                                frame_for_saving
                            )
                            
                            groups_to_remove.append(group_id)
                    
                    # Remove dispersed groups
                    for group_id in groups_to_remove:
                        del zone['crowd_groups'][group_id]
                    
                    # Display zone status on frame
                    status = f"{zone_name}: {current_count}/{zone['max_count']}"
                    if has_crowd:
                        status += f" (CROWD: {len(crowd_groups)} groups)"
                    
                    cv.putText(
                        frame_for_saving,
                        status,
                        (roi_x1, roi_y1 - 10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )
            
            # 8. Highlight crowd members visually
            # Draw bounding boxes and labels
            crowd_member_indices = set()
            for zone_name, zone in self.zones.items():
                for group_id, group_info in zone['crowd_groups'].items():
                    crowd_member_indices.update(group_info['members'])
            
            for (x1, y1, x2, y2, full_name, conf, posture), track_id in annotations:
                # Red for crowd members, Green for others
                color = (0, 0, 255) if track_id in crowd_member_indices else (0, 255, 0)
                
                cv.rectangle(frame_for_saving, (x1, y1), (x2, y2), color, 3)
                label = f"{full_name} ({posture})"
                cv.putText(
                    frame_for_saving, label, (x1, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
                cv.putText(
                    frame_for_saving, f"{conf:.2f}", (x1, y2 + 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
            
            faces_detected_count += current_frame_faces
            
            # Display stats on frame
            fps = frame_counter / (datetime.now() - start_time).total_seconds()
            cv.putText(frame_for_saving, f"FPS: {fps:.2f}", (30, 30),
                    cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
            cv.putText(frame_for_saving, f"Faces: {current_frame_faces}", (30, 60),
                    cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(frame_for_saving, f"Frame: {frame_counter}", (30, 90),
                    cv.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            
            timestamp_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv.putText(
                frame_for_saving, timestamp_text,
                (self.width - 300, self.height - 20),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            # Update current frame for streaming
            with self.current_frame_lock:
                self.current_frame = frame_for_saving.copy()
        
        # Cleanup
        self.stop_thread.set()
        self.logger.info(
            f"[COMPLETE] Processed {frame_counter} frames, "
            f"detected {faces_detected_count} faces, "
            f"{crowd_detected_count} crowd events"
        )
        
        # Save CSV logs
        if self.csv_logs:
            try:
                with open(log_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
                    fieldnames = [
                        "Frame", "TrackID", "EmployeeID", "Name", "Department",
                        "CenterX", "CenterY", "FaceDetected", "FaceMatched",
                        "MaxSimilarity", "Timestamp", "EventType", "FaceCaptured",
                        "CapturePath", "CrowdDetected", "CrowdSize",
                        "StandingDuration", "Posture"
                    ]
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    for log_entry in self.csv_logs:
                        writer.writerow(log_entry)
                self.logger.info(f"[OK] CSV log saved to {log_csv_path}")
            except Exception as e:
                self.logger.error(f"[ERROR] Error saving CSV log: {e}")
        
        # Close MediaPipe
        self.face_mesh.close()
        self.pose.close()
        self.logger.info(f"[SHUTDOWN] {self.camera_id} processing complete")


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Assuming templates are in a 'templates' folder at the app root
@app.route('/templates/<path:filename>')
def serve_templates(filename):
    templates_dir = os.path.join(app.root_path, 'templates')  # Safely get templates path
    return send_from_directory(templates_dir, filename)



# === ADD THESE ROUTES ===
# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route('/')
def home():
    """Show landing page as the first page"""
    return render_template('landpage.html')

@app.route('/login_page')
def login_page():
    """Show login page"""
    return render_template('login.html')

# -------------------------------------------------
# INDEX – MAIN PAGE AFTER LOGIN
# -------------------------------------------------
@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html', username=session['username'])


# -------------------------------------------------
# REGISTER
# -------------------------------------------------
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    location = data.get('location')
    module   = data.get('module')

    if not all([username, password, location, module]):
        return jsonify({'status': 'error', 'message': 'Please fill all fields'})

    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO dbo.UserDetails (username, password, location, module) VALUES (?, ?, ?, ?)",
            (username, hashed.decode('utf-8'), location, module)
        )
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Registration successful!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'})


# -------------------------------------------------
# LOGIN
# -------------------------------------------------
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT password FROM dbo.UserDetails WHERE username = ?", (username,))
    row = cur.fetchone()

    if row and bcrypt.checkpw(password.encode('utf-8'), row[0].encode('utf-8')):
        session['username'] = username
        conn.close()
        return jsonify({'status': 'success', 'redirect': url_for('index')})
    else:
        conn.close()
        return jsonify({'status': 'error', 'message': 'Invalid username or password'})


# -------------------------------------------------
# FORGOT PASSWORD
# -------------------------------------------------
@app.route('/forgot', methods=['POST'])
def forgot_password():
    data = request.get_json()
    username = data.get('username')
    new_password = data.get('new_password')

    if not username or not new_password:
        return jsonify({'status': 'error', 'message': 'Both fields required'})

    hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE dbo.UserDetails SET password = ? WHERE username = ?", (hashed.decode('utf-8'), username))
    conn.commit()
    affected = cur.rowcount
    conn.close()

    if affected:
        return jsonify({'status': 'success', 'message': 'Password updated!'})
    else:
        return jsonify({'status': 'error', 'message': 'Username not found'})


# -------------------------------------------------
# DASHBOARD (optional – keep if you need it)
# -------------------------------------------------
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('dashboard.html', username=session['username'])


# -------------------------------------------------
# LOGOUT
# -------------------------------------------------
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/employee-logs')
def employee_logs():
    return render_template('employee_logs.html')

@app.route('/attendance-logs')
def attendance_logs():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('attendance_logs.html', username=session['username'])

@app.route('/attendance-dashboard')
def attendance_dashboard():
    return render_template('attendance_dashboard.html')

@app.route('/crowd-detection-dashboard')
def crowd_detection():
    return render_template('crowd_detection.html')

# Placeholder routes for pages not yet implemented
@app.route('/id-card-compliance')
def id_card_compliance():
    return render_template('id_card_compliance.html')  # Create this file or redirect

@app.route('/unattended-system')
def unattended_system():
    return render_template('unattended_system.html')

@app.route('/sleep-mealtime-detection')
def sleep_mealtime_detection():
    return render_template('sleep_mealtime_detection.html')

@app.route('/geo-fence-intrusions')
def geo_fence_intrusions():
    return render_template('geo_fence_intrusions.html')

@app.route('/alerts-violations')
def alerts_violations():
    return render_template('alerts_violations.html')

@app.route('/analytics-trends')
def analytics_trends():
    return render_template('analytics_trends.html')

@app.route('/settings')
def settings_page():
    """Serve the camera settings page"""
    return render_template('settings.html')


@app.route('/api/cameras')
def get_cameras():
    module_configs = {
        "Module 1": [
            {"url": f"rtsp://{rtsp_user}:{rtsp_password}@{rtsp_ip_m1}:8554/Streaming/Channels/201/?rtsp_transport=tcp", "id": "Turnstile_Entry_M1"},
            {"url": f"rtsp://{rtsp_user}:{rtsp_password}@{rtsp_ip_m1}:8554/Streaming/Channels/301/?rtsp_transport=tcp", "id": "Turnstile_Exit_M1"},
            {"url": f"rtsp://{rtsp_user}:{rtsp_password}@{rtsp_ip_m1}:8554/Streaming/Channels/501/?rtsp_transport=tcp", "id": "DataTeam_cabin_1_M1"},
            {"url": f"rtsp://{rtsp_user}:{rtsp_password}@{rtsp_ip_m1}:8554/Streaming/Channels/3401/?rtsp_transport=tcp", "id": "DataTeam_cabin_2_M1"},
        ],
        "Module 3": [
            {"url": f"rtsp://{rtsp_user}:{rtsp_password}@{rtsp_ip_m3}:554/Streaming/Channels/201/?rtsp_transport=tcp", "id": "Reception_M3"},
            {"url": f"rtsp://{rtsp_user}:{rtsp_password}@{rtsp_ip_m3}:554/Streaming/Channels/2801/?rtsp_transport=tcp", "id": "Turnstile_Entry_M3"},
            {"url": f"rtsp://{rtsp_user}:{rtsp_password}@{rtsp_ip_m3}:554/Streaming/Channels/101/?rtsp_transport=tcp", "id": "Turnstile_Exit_M3"},
        ]
    }
    return jsonify(module_configs)

@app.route('/api/start/<camera_id>', methods=['POST'])
def start_camera(camera_id):
    if camera_id in processors:
        app_logger.warning(f"Start requested but {camera_id} already running")
        return jsonify({"status": "error", "message": f"{camera_id} already running"}), 400

    data = request.get_json()
    url = data.get('url')
    if not url:
        app_logger.error(f"Start failed for {camera_id}: No URL provided")
        return jsonify({"status": "error", "message": "URL required"}), 400

    try:
        processor = CameraProcessor(url, camera_id)
        thread = threading.Thread(target=processor.process, daemon=True)
        thread.start()
        processors[camera_id] = (processor, thread)
        app_logger.info(f"Started camera: {camera_id}")
        return jsonify({"status": "success", "message": f"Started {camera_id}"})
    except Exception as e:
        app_logger.error(f"Failed to start {camera_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/stop/<camera_id>', methods=['POST'])
def stop_camera(camera_id):
    if camera_id not in processors:
        return jsonify({"status": "error", "message": f"{camera_id} not running"}), 400
    
    processor, thread = processors[camera_id]
    processor.stop_thread.set()
    thread.join(timeout=15)
    if processor.out:
        processor.out.release()
    del processors[camera_id]
    return jsonify({"status": "success", "message": f"Stopped {camera_id}"})

@app.route('/api/start_all', methods=['POST'])
def start_all_cameras():
    data = request.get_json()
    cameras = data.get('cameras', [])
    results = []
    for camera in cameras:
        camera_id = camera['id']
        url = camera['url']
        if camera_id not in processors:
            try:
                processor = CameraProcessor(url, camera_id)
                thread = threading.Thread(target=processor.process)
                thread.daemon = True
                thread.start()
                processors[camera_id] = (processor, thread)
                results.append({"camera_id": camera_id, "status": "started"})
            except Exception as e:
                results.append({"camera_id": camera_id, "status": "error", "message": str(e)})
        else:
            results.append({"camera_id": camera_id, "status": "already_running"})
    return jsonify({"status": "success", "results": results})

@app.route('/api/stop_all', methods=['POST'])
def stop_all_cameras():
    results = []
    for camera_id in list(processors.keys()):
        processor, thread = processors[camera_id]
        processor.stop_thread.set()
        thread.join(timeout=15)
        if processor.out:
            processor.out.release()
        del processors[camera_id]
        results.append({"camera_id": camera_id, "status": "stopped"})
    return jsonify({"status": "success", "results": results})

@app.route('/api/status')
def get_status():
    status = {}
    for camera_id in processors:
        processor, thread = processors[camera_id]
        status[camera_id] = {
            "running": thread.is_alive(),
            "location": processor.location,
            "frame_count": processor.frame_count
        }
    return jsonify(status)

@app.route('/api/video_feed/<camera_id>')
def video_feed(camera_id):
    if camera_id not in processors:
        return jsonify({"status": "error", "message": "Camera not running"}), 404
    
    def generate():
        processor, _ = processors[camera_id]
        while True:
            # Calls get_frame() to get the latest processed frame(with face boxes, posture labels, crowd lines, etc.)
            frame = processor.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Why?
            # Browsers cannot display raw NumPy images
            # So the frame is converted into a JPEG binary format
            # Using compression quality 85 (good balance)
            ret, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            # Converts the JPEG image into raw bytes ready for HTTP streaming.
            frame_bytes = buffer.tobytes()

            # What this does:
            # This is MJPEG streaming format
            # Each frame is sent as a separate "chunk"
            # Browser updates image without refreshing page
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # You will receive multiple frames , Each frame replaces the previous one
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/set_roi/<camera_id>', methods=['POST'])
def set_roi(camera_id):
    if camera_id not in processors:
        return jsonify({"status": "error", "message": "Camera not running"}), 404
    
    data = request.get_json()
    x1 = data.get('x1')
    y1 = data.get('y1')
    x2 = data.get('x2')
    y2 = data.get('y2')
    
    if None in [x1, y1, x2, y2]:
        return jsonify({"status": "error", "message": "Invalid ROI coordinates"}), 400
    
    processor, _ = processors[camera_id]
    processor.zones['Zone1']['roi'] = [x1, y1, x2, y2]
    processor.roi_defined = True
    
    return jsonify({"status": "success", "message": "ROI set successfully"})

@app.route('/api/reset_roi/<camera_id>', methods=['POST'])
def reset_roi(camera_id):
    if camera_id not in processors:
        return jsonify({"status": "error", "message": "Camera not running"}), 404
    
    processor, _ = processors[camera_id]
    processor.zones = {
        'Zone1': {
            'roi': None,
            'max_count': 10,
            'current_count': 0,
            'last_count': 0,
            'standing_start': {},
            'crowd_groups': {},
            'crowd_lines': []
        }
    }
    processor.roi_defined = False
    
    return jsonify({"status": "success", "message": "ROI reset successfully"})
 

@app.route('/api/logs', methods=['GET'])
def api_get_all_logs():
    """Fast & paginated attendance logs with optional filters and always include face_image if available"""
    try:
        # === Pagination & Filters ===
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        camera = request.args.get('camera')
        employee_id = request.args.get('employee_id')
        from_date = request.args.get('from_date')  
        to_date = request.args.get('to_date')     

        offset = (page - 1) * per_page

        conn = get_db_connection()
        cursor = conn.cursor()

        # Base query - ALWAYS include FaceImage BLOB
        query = """
            SELECT 
                al.EmployeeID,
                al.Timestamp,
                al.CameraID,
                al.Location,
                al.ConfidenceScore,
                al.Status,
                e.FullName,
                e.Department,
                al.Description,
                e.FaceImage
            FROM AttendanceLogs al
            LEFT JOIN Employees e ON al.EmployeeID = e.EmployeeID
            WHERE 1=1
        """
        params = []

        # Employee filter
        if employee_id:
            query += " AND al.EmployeeID = ?"
            params.append(employee_id)

        # Camera filter
        if camera and camera != 'All Cameras':
            query += " AND al.CameraID = ?"
            params.append(camera)

        if from_date:
            try:
                from_dt = datetime.strptime(from_date, '%Y-%m-%d')
                query += " AND al.Timestamp >= ?"
                params.append(from_dt)
            except:
                return jsonify({"error": "Invalid from_date"}), 400

        if to_date:
            try:
                to_dt = datetime.strptime(to_date, '%Y-%m-%d') + timedelta(days=1)
                query += " AND al.Timestamp < ?"
                params.append(to_dt)
            except:
                return jsonify({"error": "Invalid to_date"}), 400

        # Count total for pagination (no change to FaceImage here)
        count_query = "SELECT COUNT(*) FROM AttendanceLogs al WHERE 1=1"
        count_params = []
        if employee_id:
            count_query += " AND al.EmployeeID = ?"
            count_params.append(employee_id)
        if camera and camera != 'All Cameras':
            count_query += " AND al.CameraID = ?"
            count_params.append(camera)
        if from_date:
            count_query += " AND al.Timestamp >= ?"
            count_params.append(from_dt)
        if to_date:
            count_query += " AND al.Timestamp < ?"
            count_params.append(to_dt)

        cursor.execute(count_query, count_params)
        total = cursor.fetchone()[0]

        # Final query with ordering and pagination
        # Always sort latest first
        # OFFSET ? → how many rows to skip
        # FETCH NEXT ? ROWS ONLY → how many rows to return
        query += " ORDER BY al.Timestamp DESC"
        query += " OFFSET ? ROWS FETCH NEXT ? ROWS ONLY"
        params.extend([offset, per_page])

        cursor.execute(query, params)
        logs = cursor.fetchall()
        cursor.close()
        conn.close()

        # Build results - always encode face_image if present
        results = []
        for row in logs:
            face_img = row[9]  # FaceImage is always the 10th column (index 9)
            result = {
                "EmployeeID": row[0],
                "Timestamp": row[1].strftime('%Y-%m-%d %H:%M:%S') if row[1] else "N/A",
                "CameraID": row[2],
                "Location": row[3] or "Unknown",
                "Confidence": f"{row[4]:.3f}" if row[4] else "N/A",
                "EventType": row[5] or "Entry",
                "Name": row[6] or "Unknown",
                "Department": row[7] or "Unknown",
                "Description": row[8] or "Unknown",
                "PhotoAvailable": face_img is not None,
            }
            if face_img:
                result["face_image"] = base64.b64encode(face_img).decode('utf-8')
            results.append(result)

        return jsonify({
            "results": results,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page
        }), 200

    except Exception as e:
        logging.error(f"API error in get_all_logs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/attendance-summary', methods=['GET'])
def api_get_attendance_summary():
    """Fetch attendance summary for employees with both First Entry and Last Exit, grouped by date"""
    try:
        employee_id = request.args.get('employee_id')  
        from_date_str = request.args.get('from_date') 
        to_date_str = request.args.get('to_date')      
        
        conn = get_db_connection()
        cursor = conn.cursor()

        # Dynamically fetch entry and exit camera IDs from camera_location_map
        entry_cameras = [cam_id for cam_id, info in camera_location_map.items() if info.get("Type") == "Entry"]
        exit_cameras = [cam_id for cam_id, info in camera_location_map.items() if info.get("Type") == "Exit"]
        relevant_cameras = entry_cameras + exit_cameras

        if not relevant_cameras:
            print("[ERROR] No entry or exit cameras found in camera_location_map")
            return jsonify({"error": "No entry or exit cameras configured"}), 400

        # Main query to get first entry and last exit grouped by date
        query = """
            SELECT
                al.EmployeeID,
                CAST(al.Timestamp AS DATE) AS AttendanceDate,
                MIN(CASE WHEN al.Status = 'Entry' THEN al.Timestamp END) AS 'First Entry',
                MAX(CASE WHEN al.Status = 'Exit' THEN al.Timestamp END) AS 'Last Exit',
                MAX(e.Department) AS Department,
                MAX(al.Location) AS Location,
                MAX(e.FullName) AS FullName,
                MAX(al.Description) AS Description
            FROM AttendanceLogs al
            LEFT JOIN Employees e ON al.EmployeeID = e.EmployeeID
            WHERE al.CameraID IN ({})
        """.format(','.join(['?'] * len(relevant_cameras)))
        
        params = relevant_cameras.copy()
        
        # Add optional date range filtering
        if from_date_str:
            try:
                from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
                query += " AND al.Timestamp >= ?"
                params.append(from_date)
            except ValueError:
                return jsonify({"error": "Invalid from_date format. Use YYYY-MM-DD"}), 400
        
        if to_date_str:
            try:
                to_date = datetime.strptime(to_date_str, '%Y-%m-%d') + timedelta(days=1)
                query += " AND al.Timestamp < ?"
                params.append(to_date)
            except ValueError:
                return jsonify({"error": "Invalid to_date format. Use YYYY-MM-DD"}), 400
        
        # Add optional employee filter
        if employee_id:
            query += " AND al.EmployeeID = ?"
            params.append(employee_id)
        
        query += """
            GROUP BY al.EmployeeID, CAST(al.Timestamp AS DATE)
            HAVING MIN(CASE WHEN al.Status = 'Entry' THEN al.Timestamp END) IS NOT NULL
            AND MAX(CASE WHEN al.Status = 'Exit' THEN al.Timestamp END) IS NOT NULL
            ORDER BY AttendanceDate DESC, al.EmployeeID
        """
        
        cursor.execute(query, params)
        logs = cursor.fetchall()
        cursor.close()
        conn.close()

        if not logs:
            return jsonify({"message": "No attendance summary found for employees with both entry and exit"}), 200

        results = []
        for row in logs:
            employee_id_val = row[0]
            attendance_date = row[1]
            first_entry = row[2]
            last_exit = row[3]
            department = row[4] or "Unknown"
            location = row[5] or "Unknown"
            full_name = row[6] or "Unknown"
            description = row[7] or "Unknown"
            
            first_entry_str = first_entry.strftime('%Y-%m-%d %H:%M:%S') if first_entry else "N/A"
            last_exit_str = last_exit.strftime('%Y-%m-%d %H:%M:%S') if last_exit else "N/A"
            
            # Calculate total time
            time_diff = last_exit - first_entry if first_entry and last_exit else timedelta(0)
            total_time = str(timedelta(seconds=int(time_diff.total_seconds())))
            
            results.append({
                "EmployeeID": employee_id_val,
                "Date": attendance_date.strftime('%Y-%m-%d') if isinstance(attendance_date, date) else str(attendance_date),
                "First Entry": first_entry_str,
                "Last Exit": last_exit_str,
                "Total Time": total_time,
                "Department": department,
                "Location": location,
                "Name": full_name,
                "Description": description
            })

        return jsonify({"results": results, "total": len(results)}), 200
    except Exception as e:
        print(f"API error in attendance_summary: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route('/crowd-detection', methods=['GET'])
def api_get_crowd_detection():
    """Fetch all crowd detection logs including PeopleCount, Duration, Posture, and ImageEmbedding"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            SELECT 
                CrowdDetectionID,
                CameraID,
                Location,
                Description,
                PeopleCount,
                DetectionTime,
                Duration,
                Posture,
                ImageEmbedding
            FROM CrowdDetection
            ORDER BY DetectionTime DESC
        """
        cursor.execute(query)
        logs = cursor.fetchall()
        cursor.close()
        conn.close()
        if not logs:
            return jsonify({"message": "No crowd detection logs found in the database"}), 200
        results = [
            {
                "CrowdDetectionID": crowd_detection_id,
                "CameraID": camera_id,
                "Location": location or "Unknown",
                "Description": description or "Unknown",
                "PeopleCount": people_count,
                "DetectionTime": detection_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(detection_time, datetime) else str(detection_time) if detection_time else "N/A",
                "Duration": duration,
                "Posture": posture or "N/A",
                "ImageEmbedding": base64.b64encode(image_embedding).decode('utf-8') if image_embedding and isinstance(image_embedding, bytes) else None
            }
            for crowd_detection_id, camera_id, location, description, people_count, detection_time, duration, posture, image_embedding in logs
        ]
        return jsonify({"results": results, "total": len(logs)}), 200
    except Exception as e:
        print(f"API error in get_crowd_detection: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app_logger.info("=== NETRICA.AI STARTUP ===")
    
    # Load camera locations
    try:
        load_camera_locations()
    except Exception as e:
        app_logger.critical("Failed to load camera locations. Exiting.")
        sys.exit(1)

    # Load embeddings
    app_logger.info("Loading known face embeddings from database...")
    try:
        embeddings = get_all_embeddings()
        known_embeddings.update({
            emp_id: np.frombuffer(emb, dtype=np.float32)
            for emp_id, emb in embeddings
        })
        app_logger.info(f"Loaded {len(known_embeddings)} employee embeddings")
    except Exception as e:
        app_logger.error(f"Failed to load embeddings: {e}")

    # Start Flask
    app_logger.info("Starting Flask server on port 5004...")
    try:
        app.run(host='0.0.0.0', port=5004, debug=False, threaded=True)
    except Exception as e:
        app_logger.critical(f"Flask server crashed: {e}")
    finally:
        app_logger.info("=== NETRICA.AI SHUTDOWN ===")