import cv2
import numpy as np
import pickle
from pathlib import Path
import sys
import glob

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def calibrate_from_images(image_folder, num_corners=(10, 7), square_size=0.02):
    """
    Calibrate the camera using chessboard pattern images from a folder.
    
    Args:
        image_folder: Path to folder containing calibration images
        num_corners: Number of inner corners in the chessboard (width=10, height=7)
        square_size: Size of chessboard squares in meters
    """
    # Convert image_folder to Path object if it's a string
    image_folder = Path(image_folder)
    
    if not image_folder.exists():
        print(f"Error: Image folder '{image_folder}' does not exist")
        return None, None
    
    # Get list of image files
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_files.extend(glob.glob(str(image_folder / ext)))
    
    if not image_files:
        print(f"Error: No images found in '{image_folder}'")
        return None, None
    
    print(f"\nFound {len(image_files)} images in {image_folder}")
    
    # Prepare object points
    objp = np.zeros((num_corners[0] * num_corners[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_corners[0], 0:num_corners[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    successful_images = 0
    image_size = None
    
    print("\nProcessing images:")
    for i, image_file in enumerate(image_files, 1):
        # Read image
        image = cv2.imread(image_file)
        if image is None:
            print(f"Error: Could not read image {image_file}")
            continue
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Store image size for first valid image
        if image_size is None:
            image_size = gray.shape[::-1]
        elif gray.shape[::-1] != image_size:
            print(f"Warning: Image {image_file} has different size than first image. Skipping.")
            continue
        
        # Find chessboard corners
        found, corners = cv2.findChessboardCorners(gray, num_corners, None)
        
        if found and corners.shape[0] == num_corners[0] * num_corners[1]:
            # Refine corners
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Store points
            objpoints.append(objp)
            imgpoints.append(refined_corners)
            successful_images += 1
            
            # Draw and display the corners (optional)
            display_img = image.copy()
            cv2.drawChessboardCorners(display_img, num_corners, refined_corners, found)
            
            # Save the visualization
            output_dir = project_root / 'data' / 'camera_calibration' / 'visualizations'
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = output_dir / f'corners_{i:02d}.jpg'
            cv2.imwrite(str(output_path), display_img)
            
            print(f"✓ Image {i}/{len(image_files)}: Successfully detected corners")
        else:
            print(f"✗ Image {i}/{len(image_files)}: Failed to detect corners")
    
    print(f"\nSuccessfully processed {successful_images} out of {len(image_files)} images")
    
    if successful_images < 10:
        print("Error: Need at least 10 successful images for reliable calibration")
        return None, None
    
    print("\nPerforming camera calibration...")
    try:
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )
        
        if not ret:
            print("Error: Calibration failed")
            return None, None
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        mean_error = mean_error/len(objpoints)
        
        print(f"\nCalibration successful!")
        print(f"Average reprojection error: {mean_error:.5f} pixels")
        
        # Save calibration data
        calibration_data = {
            'camera_matrix': mtx,
            'distortion_coefficients': dist,
            'rotation_vectors': rvecs,
            'translation_vectors': tvecs,
            'image_size': image_size,
            'reprojection_error': mean_error,
            'num_images_used': successful_images
        }
        
        # Create camera calibration directory if it doesn't exist
        calibration_dir = project_root / 'data' / 'camera_calibration'
        calibration_dir.mkdir(exist_ok=True, parents=True)
        
        # Save calibration data
        calibration_file = calibration_dir / 'calibration.pkl'
        with open(calibration_file, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print(f"\nCalibration data saved to {calibration_file}")
        print(f"Camera matrix:\n{mtx}")
        print(f"Distortion coefficients:\n{dist}")
        
        return mtx, dist
        
    except Exception as e:
        print(f"\nError during calibration: {str(e)}")
        return None, None

def load_calibration_data():
    """Load camera calibration data if it exists."""
    calibration_file = project_root / 'data' / 'camera_calibration' / 'calibration.pkl'
    if calibration_file.exists():
        with open(calibration_file, 'rb') as f:
            calibration_data = pickle.load(f)
        return calibration_data['camera_matrix'], calibration_data['distortion_coefficients']
    return None, None

def main():
    # Default calibration images folder
    default_folder = project_root / 'data' / 'calibration_images'
    
    # Check if calibration data already exists
    mtx, dist = load_calibration_data()
    if mtx is not None:
        print("Calibration data already exists.")
        print(f"Camera matrix:\n{mtx}")
        print(f"Distortion coefficients:\n{dist}")
        choice = input("Do you want to recalibrate? (y/n): ")
        if choice.lower() != 'y':
            return
    
    # Get calibration images folder
    folder = input(f"Enter path to calibration images folder [{default_folder}]: ").strip()
    if not folder:
        folder = default_folder
    
    # Start calibration process
    print("\nStarting camera calibration...")
    mtx, dist = calibrate_from_images(folder)
    
    if mtx is not None:
        print("\nCalibration successful!")
    else:
        print("\nCalibration failed. Please check the calibration images and try again.")

if __name__ == '__main__':
    main() 