import os
from pathlib import Path

# Get the project root directory (where src/ and datasets/ are located)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Dataset paths and configuration
DATASETS = {
    'mpiigaze': {
        'path': PROJECT_ROOT / 'datasets' / 'MPIIGaze',
        'annotation_path': PROJECT_ROOT / 'datasets' / 'MPIIGaze' / 'Annotation Subset',
        'data_path': PROJECT_ROOT / 'datasets' / 'MPIIGaze' / 'Data',
        'evaluation_path': PROJECT_ROOT / 'datasets' / 'MPIIGaze' / 'Evaluation Subset',
        'face_model_path': PROJECT_ROOT / 'datasets' / 'MPIIGaze' / '6 points-based face model.mat',
        'image_size': (64, 64),
        'eye_region': (100, 200, 100, 200)  # (y_min, y_max, x_min, x_max)
    },
    'gazecapture': {
        'path': PROJECT_ROOT / 'datasets' / 'gazecapture',
        'image_size': (64, 64),
        'eye_region': (100, 200, 100, 200)  # (y_min, y_max, x_min, x_max)
    },
    'mpiifacegaze': {
        'path': PROJECT_ROOT / 'datasets' / 'MPIIFaceGaze',
        'data_path': PROJECT_ROOT / 'datasets' / 'MPIIFaceGaze' / 'Data',
        'face_model_path': PROJECT_ROOT / 'datasets' / 'MPIIFaceGaze' / 'face_model.mat',
        'image_size': (224, 224),
        'eye_region': (100, 200, 100, 200)
    }
}

# Model configuration
MODEL = {
    'input_channels': 1,
    'output_size': 2,  # (x, y) gaze coordinates
    'save_path': PROJECT_ROOT / 'model_weights',
    'name_pattern': 'best_model_{:03d}.pth',  # Format: best_model_001.pth, best_model_002.pth, etc.
}

# Training configuration
TRAINING = {
    'batch_size': 2,
    'num_epochs': 30,
    'num_workers': 1,
    'learning_rate': 0.001,
    'scheduler_patience': 5,
    'train_val_split': 0.8,
    'use_evaluation_set': True  # Whether to use the evaluation subset for validation
}

# Create necessary directories
for dataset in DATASETS.values():
    os.makedirs(dataset['path'], exist_ok=True)
os.makedirs(MODEL['save_path'], exist_ok=True) 