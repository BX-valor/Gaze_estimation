import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import glob

from src.models.model import create_model
from src.data.mpiifacegaze_loader import get_dataloaders
from src.config.config import DATASETS, MODEL, TRAINING

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
    print(gpu_info)
    
    if not available_gpus:
        return torch.device('cpu'), False
    
    if gpu_ids is None:
        gpu_ids = available_gpus
    else:
        # Validate GPU IDs
        gpu_ids = [i for i in gpu_ids if i in available_gpus]
        if not gpu_ids:
            print("Warning: No valid GPU IDs provided. Using CPU.")
            return torch.device('cpu'), False
    
    if len(gpu_ids) == 1:
        return torch.device(f'cuda:{gpu_ids[0]}'), False
    else:
        return torch.device('cuda'), True

def train(model, train_loader, val_loader, device, gpu_ids=None, num_epochs=None):
    if num_epochs is None:
        num_epochs = TRAINING['num_epochs']
    
    # Set up multi-GPU if available
    is_multi_gpu = isinstance(model, nn.DataParallel)
    if not is_multi_gpu and gpu_ids and len(gpu_ids) > 1:
        print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
        is_multi_gpu = True
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAINING['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=TRAINING['scheduler_patience']
    )
    
    best_val_loss = float('inf')
    best_model_weights = None
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress')
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        
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
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(device)
                gaze_point = batch['gaze'].to(device)
                
                outputs = model(image)
                loss = criterion(outputs, gaze_point)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update epoch progress bar description
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'lr': f'{current_lr:.6f}'
        })
        
        # 更新最佳模型参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存当前模型参数
            if isinstance(model, nn.DataParallel):
                best_model_weights = model.module.state_dict()
            else:
                best_model_weights = model.state_dict()
    
     # 训练结束后保存最佳模型
    if best_model_weights is not None:
        save_dir = MODEL['save_path']
        save_dir.mkdir(exist_ok=True)
        
        # 生成模型编号
        existing_models = glob.glob(str(save_dir / MODEL['name_pattern'].replace('{:03d}', '*')))
        numbers = []
        for model_path in existing_models:
            try:
                number = int(Path(model_path).stem.split('_')[-1])
                numbers.append(number)
            except ValueError:
                continue
        next_num = max(numbers) + 1 if numbers else 1
        
        model_name = MODEL['name_pattern'].format(next_num)
        model_path = save_dir / model_name
        
        # 保存模型
        torch.save(best_model_weights, model_path)
        print(f"\nSaved final best model: {model_path}")
            

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
    device, is_multi_gpu = setup_device(args.gpu_ids)
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        args.dataset_path,
        'mpiifacegaze',
        args.batch_size,
        args.num_workers
    )
    
    # Create and train model
    model = create_model(device)
    train(model, train_loader, val_loader, device, args.gpu_ids, args.num_epochs)

if __name__ == '__main__':
    main() 