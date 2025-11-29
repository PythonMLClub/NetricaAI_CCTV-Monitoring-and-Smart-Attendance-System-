import streamlit as st
from utils.db_handler import insert_employee
from utils.arcface_embedder import ArcFaceEmbedder  # Import your embedding class
import cv2
import numpy as np
import os

# Get the absolute path of the directory containing face_registration.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize the face embedder (do this once to avoid reloading the model) decorators
@st.cache_resource
def load_face_embedder():
    try:
        embedder = ArcFaceEmbedder("models/arcface.onnx")
        return embedder
    except Exception as e:
        st.error(f"Failed to load face embedding model: {e}")
        return None

# Page configuration
st.set_page_config(page_title="Employee Face Registration", layout="centered")
st.title("👤 Employee Face Registration")

# Load face embedder
face_embedder = load_face_embedder()

# Input fields for employee details
emp_id = st.text_input("🔢 Employee ID")
full_name = st.text_input("👤 Full Name")
department = st.text_input("🏢 Department")
role = st.text_input("💼 Role")

# Shift time dropdown
shift_options = [
    "09:00 to 18:00", "09:30 to 18:30", "10:00 to 19:00",
    "10:30 to 19:30", "11:00 to 20:00", "11:30 to 20:30"
]
shift_time = st.selectbox("⏰ Shift Time", shift_options)

# Option to upload image or capture via webcam
image_source = st.radio("📸 Select Image Source", ("Upload Photo", "Capture via Webcam"))

image_bytes = None
if image_source == "Upload Photo":
    uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_bytes = uploaded_file.read()
elif image_source == "Capture via Webcam":
    picture = st.camera_input("Take a picture")
    if picture:
        image_bytes = picture.getvalue()

def detect_and_extract_face(image):
    """
    Detect face in the image and return the cropped face region
    Returns the largest face found in the image
    """
    try:
        # Load face detection model (you can use OpenCV's or any other face detector)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

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

# Register button
if st.button("✅ Register"):
    if emp_id and full_name and department and role and shift_time and image_bytes and face_embedder:
        try:
            # Convert image bytes to OpenCV format
            # Decode Image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("Failed to decode the uploaded image")
            else:
                # Detect and extract face from the image
                # Face Detection & Cropping
                with st.spinner("Detecting face in the image..."):
                    face_region, face_message = detect_and_extract_face(image) # Detects the biggest face and extracts it with padding.
                
                if face_region is None:
                    st.error(f"Face detection failed: {face_message}")
                else:
                    st.success(face_message)
                    
                    # Generate face embedding
                    with st.spinner("Generating face embedding..."):
                        embedding = face_embedder.get_embedding(face_region) # That call returns a 512-dim float32 L2-normalized embedding that you store in SQL
                    
                    if embedding is None:
                        st.error("Failed to generate face embedding")
                    else:
                        st.success("Face embedding generated successfully")
                        
                        # Convert embedding to bytes for database storage
                        embedding_bytes = embedding.astype(np.float32).tobytes()
                        
                        # Save the original image as .jpg for visual verification
                        images_dir = os.path.join(BASE_DIR, "images")
                        os.makedirs(images_dir, exist_ok=True)
                        image_path = os.path.join(images_dir, f"employee_{emp_id}.jpg")
                        
                        # Save face region for verification
                        face_path = os.path.join(images_dir, f"employee_{emp_id}_face.jpg")
                        
                        try:
                            cv2.imwrite(image_path, image)
                            cv2.imwrite(face_path, face_region)
                            st.success(f"Images saved: {image_path} and {face_path}")
                        except Exception as e:
                            st.warning(f"Failed to save verification images: {e}")
                        
                        # Convert emp_id to integer and insert into database
                        emp_id_int = int(emp_id)
                        success, message = insert_employee(
                            emp_id_int, 
                            full_name, 
                            department, 
                            role,  # Pass the role to the insert_employee function
                            shift_time, 
                            image_bytes, 
                            embedding_bytes
                        )
                        
                        if success:
                            st.success("✅ " + message)
                            st.info(f"🔢 Embedding vector size: {len(embedding)} dimensions")
                            
                            # Optional: Display some embedding statistics
                            with st.expander("📊 Embedding Statistics"):
                                st.write(f"Embedding shape: {embedding.shape}")
                                st.write(f"Min value: {embedding.min():.4f}")
                                st.write(f"Max value: {embedding.max():.4f}")
                                st.write(f"Mean value: {embedding.mean():.4f}")
                                st.write(f"Standard deviation: {embedding.std():.4f}")
                        else:
                            st.error("❌ " + message)
                        
        except ValueError:
            st.error("Employee ID must be a valid integer.")
        except Exception as e:
            st.error(f"Registration error: {e}")
            st.exception(e)  # This will show the full traceback for debugging
    else:
        missing_items = []
        if not emp_id: missing_items.append("Employee ID")
        if not full_name: missing_items.append("Full Name")
        if not department: missing_items.append("Department")
        if not role: missing_items.append("Role")
        if not shift_time: missing_items.append("Shift Time")
        if not image_bytes: missing_items.append("Image")
        if not face_embedder: missing_items.append("Face Embedding Model")
        
        st.warning(f"⚠️ Please provide: {', '.join(missing_items)}")

# Optional: Add a section to test face detection on uploaded images
if image_bytes:
    st.subheader("🔍 Image Preview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Image:**")
        st.image(image_bytes, caption="Uploaded Image", width=300)
    
    with col2:
        st.write("**Face Detection Preview:**")
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is not None:
            face_region, _ = detect_and_extract_face(image)
            if face_region is not None:
                # Convert BGR to RGB for proper display
                face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                st.image(face_rgb, caption="Detected Face", width=300)
            else:
                st.write("No face detected")