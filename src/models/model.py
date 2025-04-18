import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path

from src.config.config import MODEL

class GazeEstimationModel(nn.Module):
    def __init__(self):
        super(GazeEstimationModel, self).__init__()
        
        # Feature extractor for full face image
        self.feature_extractor = nn.Sequential(
            # First block
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fifth block
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Flatten()
        )
        
        # Calculate the size of flattened features
        # self.feature_size = 512 * 7 * 7  # For 224x224 input
        # 动态计算特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 224, 224)  # 假设输入为 224x224
            dummy_output = self.feature_extractor(dummy_input)
            self.feature_size = dummy_output.shape[1]
        
        # Fully connected layers with batch normalization
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 1024),  # Use calculated feature size
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, MODEL['output_size'])  # Output: (x, y) gaze coordinates
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, image):
        # Extract features from the full face image
        features = self.feature_extractor(image)
        # print(features.shape)
        # Estimate gaze point
        gaze_point = self.fc(features)
        
        return gaze_point

def create_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Create and initialize the gaze estimation model."""
    model = GazeEstimationModel()
    model = model.to(device)
    return model

def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load a trained model from disk."""
    model = create_model(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    return model 