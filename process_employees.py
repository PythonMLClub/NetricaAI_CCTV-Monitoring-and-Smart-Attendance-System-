import os
import csv
import cv2
import numpy as np
from utils.db_handler import insert_employee, get_employee_details
from utils.arcface_embedder import ArcFaceEmbedder  

# Base directory containing the script and img folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "Employee_images", "Employesbase")
LOG_FILE = os.path.join(BASE_DIR, "embedding_failures.log")

# Load ArcFace embedder
try:
    embedder = ArcFaceEmbedder(os.path.join(BASE_DIR, "models", "arcface.onnx"))
except Exception as e:
    print(f"❌ Failed to load ArcFace model: {e}")
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"Failed to load ArcFace model: {e}\n")
    exit(1)

# Load face detection model
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise ValueError("Failed to load Haar cascade classifier")
except Exception as e:
    print(f"❌ Failed to load face detection model: {e}")
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"Failed to load face detection model: {e}\n")
    exit(1)

# Read employee details from CSV
def read_employee_csv(csv_path):
    employees = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Normalize EmpCode by removing case-insensitive 'inc' prefix
                emp_code = row['EmpCode'].lower().replace("inc", "").strip()
                employees[emp_code] = {
                    'full_name': row['Name'],
                    'department': row['Department'],
                    'role': row['Designation'],
                    'shift_time': row['ShiftName'],  # Use exact shift time from CSV
                    'original_emp_code': row['EmpCode']  # Store original EmpCode
                }
                print(f"Processed CSV row - emp_id: {emp_code}, original_emp_code: {row['EmpCode']}, shift_time: {row['ShiftName']}")
        return employees
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"Error reading CSV file {csv_path}: {e}\n")
        return {}

# Detect and extract face from image
def detect_and_extract_face(image):
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))
        
        if len(faces) == 0:
            return None, "No face detected in the image"
        
        # Get the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Add some padding around the face
        padding = int(0.2 * min(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        return face_region, "Face detected successfully"
        
    except Exception as e:
        return None, f"Face detection error: {e}"

# Process images and generate embeddings
def process_images(employees):
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    with open(LOG_FILE, 'a') as log_file:
        for emp_id, details in employees.items():
            try:
                # Check if employee already exists in the database
                full_name, _, _ = get_employee_details(int(emp_id))
                if full_name:
                    print(f"Skipping existing employee with emp_id: {emp_id}")
                    log_file.write(f"Skipped existing employee - emp_id: {emp_id}\n")
                    continue
                
                # Look for image matching the EmpCode case-insensitively
                image_path = None
                for ext in supported_extensions:
                    for prefix in ['inc', 'INC', 'Inc', '']:
                        potential_path = os.path.join(IMG_DIR, f"{prefix}{emp_id}{ext}")
                        if os.path.exists(potential_path):
                            image_path = potential_path
                            print(f"Found image at: {image_path}")
                            break
                    if image_path:
                        break
                
                if not image_path:
                    log_path = os.path.join(IMG_DIR, f"[inc/INC/Inc]{emp_id}.*")
                    print(f"Image not found for emp_id: {emp_id}")
                    log_file.write(f"Image not found for emp_id: {emp_id} at {log_path}\n")
                    continue
                
                # Read and process image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image for emp_id: {emp_id} at {image_path}")
                    log_file.write(f"Failed to load image for emp_id: {emp_id} at {image_path}\n")
                    continue
                
                # Resize image to a standard size to reduce binary data size
                # resizing before detection can make faces harder to find if they get too small.
                image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
                
                # Detect and extract face
                # the cropped face image.
                face_region, face_message = detect_and_extract_face(image)
                if face_region is None:
                    print(f"Face detection failed for emp_id: {emp_id}: {face_message}")
                    log_file.write(f"Face detection failed for emp_id: {emp_id}: {face_message}\n")
                    continue
                
                print(f"Face detection for emp_id {emp_id}: {face_message}")
                
                # Generate embedding
                # Feeds the cropped face into the ArcFace ONNX model.
                # The model produces a 512-dimensional vector (like [0.121, -0.093, 0.542, …]).
                # This vector is unique for that person’s face.
                embedding = embedder.get_embedding(face_region)
                if embedding is None:
                    print(f"Failed to generate embedding for emp_id: {emp_id}")
                    log_file.write(f"Failed to generate embedding for emp_id: {emp_id} at {image_path}\n")
                    continue
                
                # Convert embedding and image to bytes
                embedding_bytes = embedding.astype(np.float32).tobytes()
                image_bytes = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tobytes()
                print(f"Image bytes size: {len(image_bytes)} bytes, Embedding bytes size: {len(embedding_bytes)} bytes")
                
                # Validate binary data size
                if len(image_bytes) > 2**31:
                    print(f"Image size too large for emp_id: {emp_id} ({len(image_bytes)} bytes)")
                    log_file.write(f"Image size too large for emp_id: {emp_id} ({len(image_bytes)} bytes)\n")
                    continue
                if len(embedding_bytes) > 2**31:
                    print(f"Embedding size too large for emp_id: {emp_id} ({len(embedding_bytes)} bytes)")
                    log_file.write(f"Embedding size too large for emp_id: {emp_id} ({len(embedding_bytes)} bytes)\n")
                    continue
                
                # Use the exact shift time from CSV
                shift_time = details['shift_time']
                print(f"Using shift_time for emp_id {emp_id}: {shift_time}")
                
                # Save to database
                emp_id_int = int(emp_id)
                success, message = insert_employee(
                    emp_id_int,
                    details['full_name'],
                    details['department'],
                    details['role'],
                    shift_time,
                    image_bytes,
                    embedding_bytes
                )
                print(f"Employee {emp_id}: {message}")
                log_file.write(f"Employee {emp_id}: {message}\n")
                
            except Exception as e:
                print(f"Error processing employee {emp_id}: {e}")
                log_file.write(f"Error processing employee {emp_id}: {e}\n")
                continue

if __name__ == "__main__":
    # Path to CSV file (adjust as needed)
    csv_path = os.path.join(BASE_DIR, "employeeReport1759986085.csv")
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found at: {csv_path}")
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"CSV file not found at: {csv_path}\n")
    else:
        employees = read_employee_csv(csv_path)
        if employees:
            print(f"Processing {len(employees)} employees from CSV")
            process_images(employees)
        else:
            print("No employee data found in CSV")
            with open(LOG_FILE, 'a') as log_file:
                log_file.write("No employee data found in CSV\n")