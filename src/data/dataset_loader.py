import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import scipy.io as sio
from ..config.config import DATASETS, TRAINING
from ..utils.visualization import visualize_eye_extraction, visualize_batch, plot_gaze_distribution

def normalize_eye_region(img, landmarks, eye_width=60, eye_height=36):
    """
    Normalize eye region based on landmarks.
    
    Args:
        img: Input image
        landmarks: Eye corner landmarks [left_corner, right_corner]
        eye_width: Desired width of normalized eye region
        eye_height: Desired height of normalized eye region
    """
    left_corner, right_corner = landmarks
    eye_center = (left_corner + right_corner) / 2
    eye_length = np.linalg.norm(right_corner - left_corner)
    
    # Calculate scaling factor
    scale = eye_width / eye_length
    
    # Calculate rotation matrix
    angle = np.arctan2(right_corner[1] - left_corner[1],
                      right_corner[0] - left_corner[0])
    rotation_matrix = cv2.getRotationMatrix2D(
        tuple(eye_center), np.degrees(angle), scale)
    
    # Adjust translation to center the eye
    rotation_matrix[0, 2] += eye_width/2 - eye_center[0]
    rotation_matrix[1, 2] += eye_height/2 - eye_center[1]
    
    # Apply transformation
    normalized_eye = cv2.warpAffine(img, rotation_matrix, (eye_width, eye_height))
    return normalized_eye

class MPIIGazeDataset(Dataset):
    """
    MPIIGaze dataset loader.
    
    The dataset expects the following structure:
    - Annotation Subset/
        - p00/
            - *.txt (annotation files)
        - p01/
            ...
    - Data/
        - p00/
            - *.jpg (image files)
        - p01/
            ...
    - Evaluation Subset/
        - p00/
            - *.txt and *.jpg files
        ...
    - 6 points-based face model.mat:
        A MATLAB .mat file containing:
        - leye_idx: Indices for left eye corners (1-based)
        - reye_idx: Indices for right eye corners (1-based)
        - model_points: 3D face model points
        - camera_matrix: Camera calibration matrix
    """
    
    def __init__(self, dataset_type='train', transform=None, debug_visualization=False):
        """
        Initialize the MPIIGaze dataset loader.
        
        Args:
            dataset_type (str): 'train', 'val', or 'evaluation'
            transform (callable, optional): Optional transform to be applied on a sample
            debug_visualization (bool): Whether to show debug visualizations
        """
        self.transform = transform
        self.debug_visualization = debug_visualization
        self.samples = []
        
        # Get dataset configuration
        self.config = DATASETS['mpiigaze']
        
        # Validate dataset paths
        self._validate_dataset_paths()
        
        # Load and validate face model
        self._load_face_model()
        
        if dataset_type == 'evaluation':
            self._load_evaluation_set()
        else:
            self._load_training_set(dataset_type == 'val')
            
        print(f"Loaded {len(self.samples)} samples for {dataset_type} set")
        
        # Plot gaze distribution if in debug mode
        if self.debug_visualization:
            gazes = np.array([sample['gaze'] for sample in self.samples])
            plot_gaze_distribution(gazes)
    
    def _validate_dataset_paths(self):
        """Validate that all required dataset paths exist."""
        required_paths = [
            ('annotation_path', 'Annotation Subset'),
            ('data_path', 'Data directory'),
            ('evaluation_path', 'Evaluation Subset'),
            ('face_model_path', '6 points-based face model')
        ]
        
        for path_key, description in required_paths:
            path = self.config[path_key]
            if not path.exists():
                raise FileNotFoundError(
                    f"{description} not found at {path}. "
                    "Please ensure the dataset is properly set up."
                )
    
    def _load_face_model(self):
        """Load and validate the face model file."""
        try:
            face_model_data = sio.loadmat(str(self.config['face_model_path']))
            
            # Validate required fields
            required_fields = ['leye_idx', 'reye_idx', 'model_points', 'camera_matrix']
            for field in required_fields:
                if field not in face_model_data:
                    raise ValueError(f"Face model missing required field: {field}")
            
            # Convert to 0-based indexing and store
            self.face_model = {
                'leye_idx': face_model_data['leye_idx'].flatten() - 1,
                'reye_idx': face_model_data['reye_idx'].flatten() - 1,
                'model_points': face_model_data['model_points'],
                'camera_matrix': face_model_data['camera_matrix']
            }
            
            # Validate data shapes
            if len(self.face_model['leye_idx']) != 2 or len(self.face_model['reye_idx']) != 2:
                raise ValueError("Eye indices should contain exactly 2 points each")
                
        except Exception as e:
            raise RuntimeError(f"Error loading face model: {str(e)}")
    
    def _load_training_set(self, is_validation=False):
        """Load training or validation data from Annotation Subset."""
        annotation_path = self.config['annotation_path']
        data_path = self.config['data_path']
        
        # Get all person directories
        person_dirs = sorted(list(annotation_path.glob('p*')))
        
        # Split into train/val if using annotation subset
        if is_validation:
            person_dirs = person_dirs[::5]  # Every 5th person for validation
        else:
            person_dirs = [d for i, d in enumerate(person_dirs) if i % 5 != 0]  # Rest for training
        
        for person_dir in person_dirs:
            person_id = person_dir.name
            annotation_files = sorted(list(person_dir.glob('*.txt')))
            
            for annotation_file in annotation_files:
                try:
                    # Get corresponding image path
                    img_name = annotation_file.stem + '.jpg'
                    img_path = data_path / person_id / img_name
                    
                    if not img_path.exists():
                        print(f"Warning: Image not found: {img_path}")
                        continue
                    
                    # Load annotation
                    with open(annotation_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            data = line.strip().split()
                            if len(data) < 4:  # Ensure we have at least gaze coordinates
                                continue
                            
                            # Parse gaze coordinates and head pose
                            gaze_x, gaze_y = float(data[0]), float(data[1])
                            head_pose = np.array([float(x) for x in data[2:]])
                            
                            # Load and process image
                            img = cv2.imread(str(img_path))
                            if img is None:
                                print(f"Warning: Could not load image {img_path}")
                                continue
                                
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            
                            # Extract eye regions using face model
                            left_eye, right_eye = self._extract_eye_regions(img, head_pose)
                            
                            if left_eye is not None and right_eye is not None:
                                self.samples.append({
                                    'left_eye': left_eye,
                                    'right_eye': right_eye,
                                    'gaze': np.array([gaze_x, gaze_y]),
                                    'head_pose': head_pose
                                })
                            
                except Exception as e:
                    print(f"Error processing {annotation_file}: {str(e)}")
                    continue
    
    def _load_evaluation_set(self):
        """Load data from Evaluation Subset."""
        evaluation_path = self.config['evaluation_path']
        
        for person_dir in evaluation_path.glob('p*'):
            try:
                # Load evaluation data
                eval_files = sorted(list(person_dir.glob('*.txt')))
                
                for eval_file in eval_files:
                    img_path = eval_file.with_suffix('.jpg')
                    if not img_path.exists():
                        continue
                    
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Load evaluation data
                    with open(eval_file, 'r') as f:
                        data = f.read().strip().split()
                        gaze_x, gaze_y = float(data[0]), float(data[1])
                        head_pose = np.array([float(x) for x in data[2:]])
                    
                    # Extract eye regions
                    left_eye, right_eye = self._extract_eye_regions(img, head_pose)
                    
                    if left_eye is not None and right_eye is not None:
                        self.samples.append({
                            'left_eye': left_eye,
                            'right_eye': right_eye,
                            'gaze': np.array([gaze_x, gaze_y]),
                            'head_pose': head_pose
                        })
                        
            except Exception as e:
                print(f"Error processing {person_dir}: {str(e)}")
                continue
    
    def _extract_eye_regions(self, img, head_pose):
        """
        Extract left and right eye regions using face model and head pose.
        
        Args:
            img: Input grayscale image
            head_pose: Head pose parameters [rotation_vector, translation_vector]
            
        Returns:
            tuple: (left_eye, right_eye) normalized eye regions
        """
        try:
            # Validate input
            if img is None or head_pose is None:
                raise ValueError("Invalid input image or head pose")
            
            if len(head_pose) != 6:  # 3 for rotation, 3 for translation
                raise ValueError(f"Invalid head pose parameters. Expected 6 values, got {len(head_pose)}")
            
            # Split head pose into rotation and translation
            rotation_vector = head_pose[:3].reshape(3, 1)
            translation_vector = head_pose[3:].reshape(3, 1)
            
            # Project 3D face model points to image plane
            model_points = self.face_model['model_points']
            camera_matrix = self.face_model['camera_matrix']
            image_points, _ = cv2.projectPoints(
                model_points,
                rotation_vector,
                translation_vector,
                camera_matrix,
                None
            )
            image_points = image_points.reshape(-1, 2)
            
            # Get eye corner landmarks
            left_eye_points = image_points[self.face_model['leye_idx']]
            right_eye_points = image_points[self.face_model['reye_idx']]
            
            # Validate projected points
            if not (self._is_points_valid(left_eye_points, img.shape) and 
                   self._is_points_valid(right_eye_points, img.shape)):
                raise ValueError("Projected eye points outside image boundaries")
            
            # Get eye corners
            left_eye_corners = np.array([
                left_eye_points[0],  # Left corner
                left_eye_points[1]   # Right corner
            ])
            right_eye_corners = np.array([
                right_eye_points[0],  # Left corner
                right_eye_points[1]   # Right corner
            ])
            
            # Normalize eye regions
            left_eye = normalize_eye_region(img, left_eye_corners)
            right_eye = normalize_eye_region(img, right_eye_corners)
            
            # Resize to target size if needed
            if left_eye.shape != self.config['image_size']:
                left_eye = cv2.resize(left_eye, self.config['image_size'])
            if right_eye.shape != self.config['image_size']:
                right_eye = cv2.resize(right_eye, self.config['image_size'])
            
            # Show debug visualization if enabled
            if self.debug_visualization:
                visualize_eye_extraction(
                    img, left_eye_corners, right_eye_corners, 
                    left_eye, right_eye
                )
            
            return left_eye, right_eye
            
        except Exception as e:
            print(f"Error extracting eye regions: {str(e)}")
            return None, None
    
    def _is_points_valid(self, points, image_shape):
        """Check if points lie within image boundaries."""
        h, w = image_shape[:2]
        return np.all((points >= 0) & (points[:, 0] < w) & (points[:, 1] < h))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensor and normalize
        left_eye = torch.FloatTensor(sample['left_eye']).unsqueeze(0) / 255.0
        right_eye = torch.FloatTensor(sample['right_eye']).unsqueeze(0) / 255.0
        gaze = torch.FloatTensor(sample['gaze'])
        
        if self.transform:
            left_eye = self.transform(left_eye)
            right_eye = self.transform(right_eye)
        
        return {
            'left_eye': left_eye,
            'right_eye': right_eye,
            'gaze': gaze
        }

def get_dataloaders(dataset_path=None, dataset_type='mpiigaze', batch_size=None, num_workers=None):
    """
    Create train and validation dataloaders.
    
    Args:
        dataset_path (str, optional): Not used for MPIIGaze dataset
        dataset_type (str): Type of dataset
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for data loading
    
    Returns:
        train_loader, val_loader: PyTorch DataLoader objects
    """
    # Use configuration values if not specified
    if batch_size is None:
        batch_size = TRAINING['batch_size']
    if num_workers is None:
        num_workers = TRAINING['num_workers']
    
    if dataset_type == 'mpiigaze':
        # Create datasets
        train_dataset = MPIIGazeDataset(dataset_type='train')
        
        if TRAINING['use_evaluation_set']:
            val_dataset = MPIIGazeDataset(dataset_type='evaluation')
        else:
            val_dataset = MPIIGazeDataset(dataset_type='val')
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader 