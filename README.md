# Gaze Estimation Project

This project implements a gaze estimation system using deep learning. It captures facial images from a webcam and estimates the gaze point using a neural network model.

## Features
- Real-time webcam capture
- Face and eye detection using MediaPipe
- Gaze estimation using a custom neural network
- Training and inference capabilities

## Setup
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
- `camera.py`: Webcam capture and preprocessing
- `model.py`: Neural network architecture
- `train.py`: Training script
- `inference.py`: Real-time inference
- `utils.py`: Utility functions
- `data/`: Directory for training data
- `models/`: Directory for saved models

## Usage
1. Training:
```bash
python train.py
```

2. Inference:
```bash
python inference.py
```

## Model Architecture
The gaze estimation model uses a CNN-based architecture that takes eye images as input and outputs gaze coordinates. The model is trained using a combination of:
- Eye region images
- Head pose information
- Ground truth gaze coordinates 