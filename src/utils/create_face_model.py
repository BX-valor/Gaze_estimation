import numpy as np
import scipy.io as sio
from pathlib import Path

def create_face_model(output_path):
    """
    Create a custom face model file with reasonable default values.
    
    Args:
        output_path: Path to save the face model file
    """
    # Create a simple 3D face model with 6 points
    # Points are in the order: left eye corners, right eye corners, nose tip
    model_points = np.array([
        [-30, 0, 0],    # Left eye left corner
        [-10, 0, 0],    # Left eye right corner
        [10, 0, 0],     # Right eye left corner
        [30, 0, 0],     # Right eye right corner
        [0, -20, 0],    # Nose tip
        [0, 0, 0]       # Face center
    ], dtype=np.float32)
    
    # Indices for eye corners (1-based indexing)
    leye_idx = np.array([1, 2], dtype=np.int32)  # Left eye corners
    reye_idx = np.array([3, 4], dtype=np.int32)  # Right eye corners
    
    # Camera matrix (assuming 640x480 resolution)
    camera_matrix = np.array([
        [640, 0, 320],
        [0, 640, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Create dictionary with all required fields
    face_model = {
        'model_points': model_points,
        'leye_idx': leye_idx,
        'reye_idx': reye_idx,
        'camera_matrix': camera_matrix
    }
    
    # Save as .mat file
    sio.savemat(str(output_path), face_model)
    print(f"Created face model file at: {output_path}")

if __name__ == '__main__':
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / 'datasets' / 'MPIIGaze' / '6 points-based face model.mat'
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the face model file
    create_face_model(output_path) 