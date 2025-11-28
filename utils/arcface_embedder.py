import onnxruntime as ort
import cv2
import numpy as np
import os

class ArcFaceEmbedder:
    def __init__(self, model_path="models/arcface.onnx"):
        """
        Initialize ArcFace embedding model
        Args:
            model_path: Path to the ONNX model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Create a session with optimized settings
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1  # one thread per operator
            session_options.inter_op_num_threads = 1  # one thread for graph
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Initialize ONNX Runtime session
            self.session = ort.InferenceSession(model_path, session_options=session_options)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape  # input_shape – expected shape (batch, channels, height, width) → usually [1,3,112,112]
            self.output_shape = self.session.get_outputs()[0].shape # output_shape – raw embedding shape → [1,512].
            
            print(f"✅ ArcFace model loaded successfully")
            print(f"Input shape: {self.input_shape}")
            print(f"Output shape: {self.output_shape}")
            
        except Exception as e:
            print(f"❌ Failed to load ArcFace model: {e}")
            raise

    def preprocess(self, face_img):
        """
        Preprocess face image for ArcFace model input
        Args:
            face_img: Face image in BGR format (OpenCV format)
        Returns:
            Preprocessed image array ready for model input
        """
        try:
            if face_img is None:
                return None
                
            # Check if image is valid
            if face_img.size == 0:
                print("❌ Empty face image")
                return None
            
            # Resize face to 112x112 as required by ArcFace
            face = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR to RGB
            # OpenCV reads BGR, ONNX model expects RGB.
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            # Transpose to (C, H, W) format and convert to float32
            face = np.transpose(face, (2, 0, 1)).astype(np.float32)
            
            # Normalize pixel values to [-1, 1] range
            # Same scaling used during training → critical for correct embeddings.
            face = (face - 127.5) / 128.0
            
            # Add batch dimension: (1, C, H, W)
            face = np.expand_dims(face, axis=0)
            
            return face
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None

    def get_embedding(self, face_img):
        """
        Generate face embedding from face image
        Args:
            face_img: Face image in BGR format (OpenCV format)
        Returns:
            512-dimensional normalized face embedding as numpy array
        """
        # Goal: One face to one 512-dim L2-normalized vector.
        try:
            # Preprocess the face image
            preprocessed = self.preprocess(face_img)
            if preprocessed is None:
                print("❌ Preprocessing failed")
                return None

            # Run inference to get embedding
            outputs = self.session.run(None, {self.input_name: preprocessed})
            embedding = outputs[0][0]  # Remove batch dimension
            
            # L2 normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm == 0:
                print("❌ Zero norm embedding")
                return None
                
            normalized_embedding = embedding / norm
            
            # Verify embedding properties
            if np.isnan(normalized_embedding).any():
                print("❌ NaN values in embedding")
                return None
                
            if np.isinf(normalized_embedding).any():
                print("❌ Inf values in embedding")
                return None
            
            return normalized_embedding.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Embedding generation error: {e}")
            return None

    # def compute_similarity(self, embedding1, embedding2):
    #     """
    #     Compute cosine similarity between two embeddings
    #     Args:
    #         embedding1: First embedding vector
    #         embedding2: Second embedding vector
    #     Returns:
    #         Cosine similarity score (0 to 1, where 1 is identical)
    #     """
    #     try:
    #         if embedding1 is None or embedding2 is None:
    #             return 0.0
            
    #         # Ensure embeddings are normalized
    #         norm1 = np.linalg.norm(embedding1)
    #         norm2 = np.linalg.norm(embedding2)
            
    #         if norm1 == 0 or norm2 == 0:
    #             return 0.0
            
    #         embedding1_norm = embedding1 / norm1
    #         embedding2_norm = embedding2 / norm2
            
    #         # Compute cosine similarity (dot product of normalized vectors)
    #         similarity = np.dot(embedding1_norm, embedding2_norm)
            
    #         # Ensure similarity is in [0, 1] range
    #         similarity = (similarity + 1) / 2
            
    #         return float(similarity)
            
    #     except Exception as e:
    #         print(f"❌ Similarity computation error: {e}")
    #         return 0.0

    # def batch_embeddings(self, face_images):
    #     """
    #     Generate embeddings for multiple face images
    #     Args:
    #         face_images: List of face images in BGR format
    #     Returns:
    #         List of embeddings (None for failed images)
    #     """
    #     embeddings = []
    #     for i, face_img in enumerate(face_images):
    #         try:
    #             embedding = self.get_embedding(face_img)
    #             embeddings.append(embedding)
    #             if embedding is not None:
    #                 print(f"✅ Generated embedding {i+1}/{len(face_images)}")
    #             else:
    #                 print(f"❌ Failed to generate embedding {i+1}/{len(face_images)}")
    #         except Exception as e:
    #             print(f"❌ Error processing image {i+1}: {e}")
    #             embeddings.append(None)
        
    #     return embeddings

    # def __del__(self):
    #     # Releases the ONNX session (frees GPU/CPU memory).
    #     """Cleanup when object is destroyed"""
    #     try:
    #         if hasattr(self, 'session') and self.session:
    #             del self.session
    #     except:
    #         pass