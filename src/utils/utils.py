import torch
import random
import numpy as np
import os
from typing import Optional, Tuple

def set_seed(seed: int = 42):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_checkpoint(model: torch.nn.Module,
                   checkpoint_path: str,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
                   ) -> Tuple[torch.nn.Module, dict]:
    """
    加载模型检查点
    
    Args:
        model: 神经网络模型
        checkpoint_path: 检查点文件路径
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        
    Returns:
        tuple: (加载检查点后的模型, 检查点字典)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    return model, checkpoint

def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: 神经网络模型
        
    Returns:
        int: 可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_experiment_dir(base_dir: str = 'experiments') -> str:
    """
    创建实验目录
    
    Args:
        base_dir: 基础目录路径
        
    Returns:
        str: 新创建的实验目录路径
    """
    os.makedirs(base_dir, exist_ok=True)
    experiment_id = len(os.listdir(base_dir))
    experiment_dir = os.path.join(base_dir, f'experiment_{experiment_id}')
    os.makedirs(experiment_dir)
    return experiment_dir

class AverageMeter:
    """用于计算和存储平均值和当前值的类"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """重置所有统计值"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        """
        更新统计值
        
        Args:
            val: 当前值
            n: 当前批次大小
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 