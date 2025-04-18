import cv2
import numpy as np
import pickle
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def calibrate_camera(camera_id=0, num_corners=(9, 6), square_size=0.025):
    """
    Calibrate the camera using a chessboard pattern.
    
    Args:
        camera_id: Camera device ID (default: 0 for first camera)
        num_corners: Number of inner corners in the chessboard (width, height)
        square_size: Size of chessboard squares in meters
    """
    # Initialize video capture
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None, None
    
    # Prepare object points
    objp = np.zeros((num_corners[0] * num_corners[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_corners[0], 0:num_corners[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    print("\nCamera Calibration:")
    print("1. Print a chessboard pattern (9x6 inner corners)")
    print("2. Hold it in front of the camera")
    print("3. Press 'c' to capture a frame")
    print("4. Capture at least 10 different orientations")
    print("5. Press 'q' to finish calibration")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, num_corners, None)
        
        # If found, draw corners
        if ret:
            cv2.drawChessboardCorners(frame, num_corners, corners, ret)
            cv2.putText(frame, f"Captured: {count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Camera Calibration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and ret:
            # Refine corner detection
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            count += 1
            print(f"Frame {count} captured")
            
        elif key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    if count < 10:
        print("Warning: At least 10 frames are recommended for good calibration")
        return None, None
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    if not ret:
        print("Calibration failed")
        return None, None
    
    # Save calibration data
    calibration_data = {
        'camera_matrix': mtx,
        'distortion_coefficients': dist,
        'rotation_vectors': rvecs,
        'translation_vectors': tvecs,
        'image_size': gray.shape[::-1]
    }
    
    # Create camera information directory if it doesn't exist
    camera_info_dir = project_root / 'camera_information'
    camera_info_dir.mkdir(exist_ok=True)
    
    # Save calibration data
    calibration_file = camera_info_dir / 'camera_calibration.pkl'
    with open(calibration_file, 'wb') as f:
        pickle.dump(calibration_data, f)
    
    print(f"\nCalibration completed. Data saved to {calibration_file}")
    print(f"Camera matrix:\n{mtx}")
    print(f"Distortion coefficients:\n{dist}")
    
    return mtx, dist

def load_calibration_data():
    """Load camera calibration data if it exists."""
    calibration_file = project_root / 'camera_information' / 'camera_calibration.pkl'
    if calibration_file.exists():
        with open(calibration_file, 'rb') as f:
            calibration_data = pickle.load(f)
        return calibration_data['camera_matrix'], calibration_data['distortion_coefficients']
    return None, None

def main():
    # Check if calibration data already exists
    mtx, dist = load_calibration_data()
    if mtx is not None:
        print("Calibration data already exists.")
        print(f"Camera matrix:\n{mtx}")
        print(f"Distortion coefficients:\n{dist}")
        choice = input("Do you want to recalibrate? (y/n): ")
        if choice.lower() != 'y':
            return
    
    # Start calibration process
    print("\nStarting camera calibration...")
    mtx, dist = calibrate_camera()
    
    if mtx is not None:
        print("\nCalibration successful!")
    else:
        print("\nCalibration failed. Please try again.")

if __name__ == '__main__':
    main() 