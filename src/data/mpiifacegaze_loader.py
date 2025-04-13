import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import scipy.io as sio
from ..config.config import DATASETS, TRAINING

class MPIIFaceGazeDataset(Dataset):
    """
    MPIIFaceGaze dataset loader.
    
    The dataset expects the following structure:
    - p00/
        - day01/
            - *.jpg (image files)
        - day02/
            ...
        - Calibration/
            - Camera.mat
            - monitorPose.mat
            - screenSize.mat
        - p00.txt (annotation file)
    - p01/
        ...
    
    Each annotation line in pxx.txt contains:
    - Image path
    - Gaze location (2D screen coordinates)
    - 6 facial landmarks (4 eye corners, 2 mouth corners)
    - 3D head pose (rotation and translation)
    - Face center (3D)
    - Gaze target (3D)
    - Evaluation eye indicator
    """
    
    def __init__(self, dataset_type='train', transform=None, debug_visualization=False, dataset_path=None):
        """
        Initialize the MPIIFaceGaze dataset loader.
        
        Args:
            dataset_type (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on a sample
            debug_visualization (bool): Whether to show debug visualizations
            dataset_path (str, optional): Path to the dataset directory
        """
        self.transform = transform
        self.debug_visualization = debug_visualization
        self.samples = []
        
        # Get dataset configuration
        self.config = DATASETS['mpiifacegaze']
        
        # Validate dataset paths
        self._validate_dataset_paths()
        
        # Load calibration data
        self._load_calibration_data()
        
        # Load samples
        self._load_samples(dataset_type == 'val')
            
        print(f"Loaded {len(self.samples)} samples for {dataset_type} set")
    
    def _validate_dataset_paths(self):
        """Validate that all required dataset paths exist."""
        if not self.config['path'].exists():
            raise FileNotFoundError(
                f"Dataset directory not found at {self.config['path']}. "
                "Please ensure the dataset is properly set up."
            )
    
    def _load_calibration_data(self):
        """Load camera calibration data from the first participant."""
        calibration_path = self.config['path'] / 'p00' / 'Calibration'
        
        # Load camera parameters
        camera_file = calibration_path / 'Camera.mat'
        if not camera_file.exists():
            raise FileNotFoundError(f"Camera calibration file not found at {camera_file}")
        
        camera_data = sio.loadmat(str(camera_file))
        self.camera_matrix = camera_data['cameraMatrix']
        self.dist_coeffs = camera_data['distCoeffs']
        
        # Load screen size
        screen_file = calibration_path / 'screenSize.mat'
        if not screen_file.exists():
            raise FileNotFoundError(f"Screen size file not found at {screen_file}")
        
        screen_data = sio.loadmat(str(screen_file))
        self.screen_size = {
            'width_pixel': screen_data['width_pixel'][0][0],
            'height_pixel': screen_data['height_pixel'][0][0],
            'width_mm': screen_data['width_mm'][0][0],
            'height_mm': screen_data['height_mm'][0][0]
        }
    
    def _load_samples(self, is_validation=False):
        """Load training or validation samples."""
        # Get all participant directories
        participant_dirs = sorted(list(self.config['path'].glob('p*')))
        
        # Split participants into train and validation sets
        num_participants = len(participant_dirs)
        val_split = int(num_participants * (1 - TRAINING['train_val_split']))
        
        if is_validation:
            participant_dirs = participant_dirs[-val_split:]
        else:
            participant_dirs = participant_dirs[:-val_split]
        
        # Load samples from each participant
        for participant_dir in participant_dirs:
            annotation_file = participant_dir / f"{participant_dir.name}.txt"
            
            if not annotation_file.exists():
                print(f"Warning: Annotation file not found: {annotation_file}")
                continue
            
            # Read annotation file
            with open(annotation_file, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) < 28:  # Ensure we have all required fields
                        continue
                    
                    # Parse data
                    img_path = participant_dir / data[0]
                    gaze_screen = np.array([float(data[1]), float(data[2])])
                    landmarks = np.array([float(x) for x in data[3:15]]).reshape(6, 2)
                    head_pose = np.array([float(x) for x in data[15:21]])
                    face_center = np.array([float(x) for x in data[21:24]])
                    gaze_target = np.array([float(x) for x in data[24:27]])
                    
                    # Load and process image
                    img = self._preprocess_image(img_path)
                    
                    # Normalize gaze coordinates to [-1, 1]
                    gaze_normalized = np.array([
                        gaze_screen[0] / self.screen_size['width_pixel'] * 2 - 1,
                        gaze_screen[1] / self.screen_size['height_pixel'] * 2 - 1
                    ])
                    
                    self.samples.append({
                        'image': img,
                        'gaze': gaze_normalized,
                        'landmarks': landmarks,
                        'head_pose': head_pose,
                        'face_center': face_center,
                        'gaze_target': gaze_target
                    })
    
    def _preprocess_image(self, image_path):
        """Load and preprocess an image."""
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Resize to model input size
        image = cv2.resize(image, self.config['image_size'])
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensor and normalize
        image = torch.FloatTensor(sample['image'])
        gaze = torch.FloatTensor(sample['gaze'])
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'gaze': gaze
        }

def get_dataloaders(dataset_path=None, dataset_type='mpiifacegaze', batch_size=None, num_workers=None):
    """
    Get training and validation dataloaders.
    """
    if batch_size is None:
        batch_size = TRAINING['batch_size']
    if num_workers is None:
        num_workers = TRAINING['num_workers']
    
    # Use dataset path from config if not specified
    dataset_path = dataset_path or str(DATASETS[dataset_type]['path'])
    
    # Create datasets with dataset path
    train_dataset = MPIIFaceGazeDataset('train', dataset_path=dataset_path)
    val_dataset = MPIIFaceGazeDataset('val', dataset_path=dataset_path)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, val_loader 