# Gaze Estimation Project

## Project Structure
```
gaze_estimate_lzx/
├── src/                       # Source code
│   ├── data/                  # Data loading and processing
│   ├── models/                # Model architectures
│   ├── config/                # Configuration files
│   ├── utils/                 # Utility functions
│   ├── train/                 # Training scripts
│   └── inference/             # Inference scripts
│
├── datasets/                  # Training datasets
│   ├── MPIIGaze/
│   ├── MPIIFaceGaze/
│   └── gazecapture/
│
├── camera_data/              # Camera calibration related data
│   ├── calibration_images/   # Input images for calibration
│   └── calibration_results/  # Calibration results and visualizations
│
├── model_weights/            # Saved model weights
│
├── training_records/         # Training logs and records
│
├── tests/                    # Unit tests
│
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup file
└── README.md                 # Project documentation
```

## Setup Instructions

1. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

3. 相机标定:
   - 将标定用的棋盘格图片放入 `camera_data/calibration_images/` 目录
   - 运行标定程序:
   ```bash
   python src/utils/camera_calibration.py
   ```
   - 标定结果将保存在 `camera_data/calibration_results/` 目录

4. 准备数据集:
   - 将 MPIIGaze 数据集放在 `datasets/MPIIGaze/` 目录
   - 将 MPIIFaceGaze 数据集放在 `datasets/MPIIFaceGaze/` 目录
   - 将 GazeCapture 数据集放在 `datasets/gazecapture/` 目录

5. 训练模型:
```bash
python src/train/train.py
```

6. 运行推理:
```bash
python src/inference/inference.py
```

## 功能特性
- 实时网络摄像头捕获
- 基于 MediaPipe 的人脸和眼睛检测
- 使用自定义神经网络的视线估计
- 支持模型训练和推理

## 模型架构
该视线估计模型使用基于 CNN 的架构，以眼睛图像作为输入，输出视线坐标。模型训练使用以下组合数据：
- 眼睛区域图像
- 头部姿态信息
- 真实视线坐标 