import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import glob

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ..models.model import load_model
from ..config.config import MODEL
from ..utils.camera_calibration import load_calibration_data

def get_available_models():
    """Get list of available trained models."""
    model_dir = MODEL['save_path']
    model_files = glob.glob(str(model_dir / "best_model_*.pth"))
    return sorted(model_files)

def preprocess_face_image(face_img):
    """Preprocess face image for the model."""
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to model input size
    resized = cv2.resize(gray, (224, 224))
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Add channel dimension and batch dimension
    normalized = np.expand_dims(normalized, axis=0)  # Add channel dimension
    normalized = np.expand_dims(normalized, axis=0)  # Add batch dimension
    
    # Convert to tensor
    tensor = torch.FloatTensor(normalized)
    
    return tensor

def main():
    # Load camera calibration data
    mtx, dist = load_calibration_data()
    if mtx is None:
        print("No camera calibration data found. Please run src/utils/camera_calibration.py first.")
        return
    
    # Get available models
    available_models = get_available_models()
    if not available_models:
        print("No trained models found. Please train a model first.")
        return
    
    # Display available models
    print("\nAvailable models:")
    for i, model_path in enumerate(available_models, 1):
        print(f"{i}. {Path(model_path).name}")
    
    # Let user select a model
    while True:
        try:
            choice = int(input("\nSelect a model number: "))
            if 1 <= choice <= len(available_models):
                model_path = available_models[choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Load the selected model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nLoading model: {Path(model_path).name}")
    model = load_model(model_path, device)
    model.eval()
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get screen dimensions
    screen_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    screen_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print("\nStarting gaze estimation. Press 'q' to quit.")
    print("Make sure the 'Gaze Estimation' window is in focus when pressing 'q'")
    
    # Create a named window
    cv2.namedWindow('Gaze Estimation', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Undistort frame using calibration data
        frame = cv2.undistort(frame, mtx, dist)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Preprocess face image
            face_tensor = preprocess_face_image(face_img)
            face_tensor = face_tensor.to(device)
            
            # Get gaze prediction
            with torch.no_grad():
                gaze_pred = model(face_tensor)
                gaze_pred = gaze_pred.cpu().numpy()[0]
            
            # Convert normalized coordinates to screen coordinates
            screen_x = int((gaze_pred[0] + 1) * screen_width / 2)
            screen_y = int((gaze_pred[1] + 1) * screen_height / 2)
            
            # Draw gaze point
            cv2.circle(frame, (screen_x, screen_y), 5, (0, 255, 0), -1)
            
            # Draw line from face center to gaze point
            face_center = (x + w//2, y + h//2)
            cv2.line(frame, face_center, (screen_x, screen_y), (0, 255, 0), 2)
            
            # Draw detection box around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Add text label
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Gaze Estimation', frame)
        
        # Check for key press
        key = cv2.waitKey(1)
        if key != -1:  # If a key was pressed
            print(f"Key pressed: {key}")  # Debug information
            if key & 0xFF == ord('q'):
                print("Quitting...")
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")

if __name__ == '__main__':
    main() 