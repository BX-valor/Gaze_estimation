import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.model import create_model, load_model
from src.config.config import MODEL

def list_available_models():
    """List all available trained models."""
    model_dir = MODEL['save_path']
    if not model_dir.exists():
        print("No model directory found.")
        return []
    
    models = list(model_dir.glob(MODEL['name_pattern'].replace('{:03d}', '*')))
    if not models:
        print("No trained models found.")
        return []
    
    print("\nAvailable models:")
    for i, model_path in enumerate(models, 1):
        print(f"{i}. {model_path.name}")
    
    return models

def select_model():
    """Let user select a model from available models."""
    models = list_available_models()
    if not models:
        return None
    
    while True:
        try:
            choice = int(input("\nSelect a model number (or 0 to exit): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(models):
                return models[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def preprocess_image(image):
    """Preprocess image for model input."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to model input size
    image = cv2.resize(image, (224, 224))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    
    return torch.FloatTensor(image)

def main():
    # Select model
    model_path = select_model()
    if model_path is None:
        return
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"Loading model: {model_path.name}")
    model = load_model(model_path, device)
    model.eval()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\nPress 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Preprocess image
            input_tensor = preprocess_image(frame)
            input_tensor = input_tensor.to(device)
            
            # Get prediction
            with torch.no_grad():
                gaze_point = model(input_tensor)
                gaze_point = gaze_point.cpu().numpy()[0]
            
            # Denormalize gaze point (assuming normalized to [-1, 1])
            height, width = frame.shape[:2]
            x = int((gaze_point[0] + 1) * width / 2)
            y = int((gaze_point[1] + 1) * height / 2)
            
            # Draw gaze point
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            
            # Show frame
            cv2.imshow('Gaze Estimation', frame)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 