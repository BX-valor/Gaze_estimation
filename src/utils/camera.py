import cv2
import mediapipe as mp
import numpy as np

class Camera:
    def __init__(self, camera_id=0):
        self.camera = cv2.VideoCapture(camera_id)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def get_frame(self):
        """Capture a frame from the webcam and return it."""
        ret, frame = self.camera.read()
        if not ret:
            return None
        return frame
    
    def get_eye_regions(self, frame):
        """Extract eye regions from the frame using MediaPipe."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None, None
            
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get eye landmarks (MediaPipe face mesh indices for eyes)
        left_eye_indices = [33, 133, 157, 158, 159, 160, 161, 173]
        right_eye_indices = [362, 263, 384, 385, 386, 387, 388, 398]
        
        h, w = frame.shape[:2]
        
        def get_eye_region(indices):
            points = []
            for idx in indices:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                points.append((x, y))
            return points
        
        left_eye = get_eye_region(left_eye_indices)
        right_eye = get_eye_region(right_eye_indices)
        
        return left_eye, right_eye
    
    def preprocess_eye_region(self, eye_points, frame):
        """Preprocess eye region for the model."""
        if not eye_points:
            return None
            
        # Get bounding box of eye region
        x_coords = [p[0] for p in eye_points]
        y_coords = [p[1] for p in eye_points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        # Extract and resize eye region
        eye_region = frame[y_min:y_max, x_min:x_max]
        eye_region = cv2.resize(eye_region, (64, 64))
        eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        eye_region = eye_region.astype(np.float32) / 255.0
        
        return eye_region
    
    def release(self):
        """Release camera resources."""
        self.camera.release()
        cv2.destroyAllWindows() 