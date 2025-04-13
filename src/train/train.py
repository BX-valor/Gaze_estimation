import os
import sys
from pathlib import Path
import logging
import math
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from src.models.model import create_model
from src.data.mpiifacegaze_loader import get_dataloaders
from src.config.config import DATASETS, MODEL, TRAINING

# 初始化日志配置
def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler('training_records/training.log')
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger()

def calculate_angular_error(predictions, targets):
    """计算预测值与真实值之间的角度误差（单位：度）"""
    pred_normalized = F.normalize(predictions, p=2, dim=1)
    target_normalized = F.normalize(targets, p=2, dim=1)
    dot_product = (pred_normalized * target_normalized).sum(dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    return torch.acos(dot_product) * (180 / math.pi)


def get_available_gpus():
    """
    Get the list of available GPU devices.
    
    Returns:
        list: List of available GPU indices
        str: Information about available GPUs
    """
    if not torch.cuda.is_available():
        return [], "No GPU available. Using CPU."
    
    gpu_info = []
    available_gpus = []
    
    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)
        total_memory = gpu.total_memory / 1024**2  # Convert to MB
        # Get current GPU memory usage
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
        memory_cached = torch.cuda.memory_reserved(i) / 1024**2
        free_memory = total_memory - memory_allocated - memory_cached
        
        gpu_info.append(
            f"GPU {i}: {gpu.name}\n"
            f"   - Total Memory: {total_memory:.0f}MB\n"
            f"   - Free Memory: {free_memory:.0f}MB\n"
            f"   - Used Memory: {memory_allocated:.0f}MB"
        )
        available_gpus.append(i)
    
    info_str = "Available GPUs:\n" + "\n".join(gpu_info)
    return available_gpus, info_str

def setup_device(gpu_ids=None):
    """
    Set up the training device(s).
    
    Args:
        gpu_ids (list, optional): List of GPU indices to use. If None, use all available GPUs.
    
    Returns:
        device: PyTorch device
        bool: Whether using multiple GPUs
    """
    available_gpus, gpu_info = get_available_gpus()
    logger.info(gpu_info)
    
    if not available_gpus:
        return torch.device('cpu'), False
    
    if gpu_ids is None:
        valid_gpu_ids = available_gpus
    else:
        valid_gpu_ids = [i for i in gpu_ids if i in available_gpus]
    
    use_multi_gpu = len(valid_gpu_ids) > 1
    
    if not valid_gpu_ids:
        print("Warning: No valid GPU IDs provided. Using CPU.")
        return torch.device('cpu'), False, []
    
    # Set device
    if use_multi_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device(f'cuda:{valid_gpu_ids[0]}') if valid_gpu_ids else torch.device('cpu')
    
    return device, use_multi_gpu, valid_gpu_ids

def train(model, train_loader, val_loader, device, use_multi_gpu, num_epochs=None):
    if num_epochs is None:
        num_epochs = TRAINING['num_epochs']
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAINING['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=TRAINING['scheduler_patience']
    )
    
    best_metrics = {
        'val_loss': float('inf'),
        'angular_error': float('inf'),
        'epoch': -1
    }
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, position=1)
        
        for batch in batch_pbar:
            image = batch['image'].to(device)
            gaze_point = batch['gaze'].to(device)
            
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, gaze_point)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        total_angular_error = 0.0
        val_pbar = tqdm(val_loader, desc='Validating', leave=False, position=2)
        
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(device)
                gaze_point = batch['gaze'].to(device)
                
                outputs = model(image)
                loss = criterion(outputs, gaze_point)
                val_loss += loss.item()
                angular_error = calculate_angular_error(outputs, gaze_point)
                total_angular_error += angular_error.mean().item()
                val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        avg_angular_error = total_angular_error / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update epoch progress bar description
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'angle_err': f'{avg_angular_error:.2f}°',
            'lr': f'{current_lr:.6f}'
        })

        # 记录日志
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Angular Error: {avg_angular_error:.2f}° | "
            f"LR: {current_lr:.6f}"
        )
        
        # 更新最佳模型
        if val_loss < best_metrics['val_loss']:
            best_metrics.update({
                'val_loss': val_loss,
                'angular_error': avg_angular_error,
                'epoch': epoch+1
            })
            # 根据多GPU标志获取模型参数
            best_weights = model.module.state_dict() if use_multi_gpu else model.state_dict()
    
    # 保存最佳模型
    if best_weights is not None:
        save_dir = MODEL['save_path']
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = save_dir / f"best_model_{timestamp}.pth"
        torch.save(best_weights, model_path)
        logger.info(f"\n🎯 Best Model at Epoch {best_metrics['epoch']}:")
        logger.info(f"Val Loss: {best_metrics['val_loss']:.4f} | Angular Error: {best_metrics['angular_error']:.2f}°")
            

def main():
    parser = argparse.ArgumentParser(description='Train gaze estimation model')
    parser.add_argument('--dataset_path', type=str,
                      help='Path to the dataset (overrides config)')
    parser.add_argument('--batch_size', type=int,
                      help='Batch size for training (overrides config)')
    parser.add_argument('--num_epochs', type=int,
                      help='Number of training epochs (overrides config)')
    parser.add_argument('--num_workers', type=int,
                      help='Number of workers for data loading (overrides config)')
    parser.add_argument('--gpu_ids', type=int, nargs='+',
                      help='GPU IDs to use (e.g., --gpu_ids 0 1 2). If not specified, use all available GPUs')
    
    args = parser.parse_args()
    
    # Set up device and GPU configuration
    device, use_multi_gpu, valid_gpu_ids = setup_device(args.gpu_ids)
    print(f"Using device: {device}")
    logger.info(f"Using GPUs: {valid_gpu_ids} | Multi-GPU: {use_multi_gpu}")
    
    model = create_model(device)

    # 多GPU包装
    if use_multi_gpu:
        logger.info(f"Initializing DataParallel with GPUs: {valid_gpu_ids}")
        model = nn.DataParallel(model, device_ids=valid_gpu_ids)

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        args.dataset_path,
        'mpiifacegaze',
        args.batch_size,
        args.num_workers
    )
    
    try:
        train(model, train_loader, val_loader, device, use_multi_gpu, num_epochs=args.num_epochs)
        logger.info("✅ Training completed successfully")
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise
   

if __name__ == '__main__':
    main() 