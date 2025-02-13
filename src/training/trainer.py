import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import wandb
import os
from tqdm import tqdm
import numpy as np
from ..models.eegnet import EEGNet

class EEGTrainer:
    """EEG模型训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化训练器
        
        Args:
            model: 神经网络模型
            config: 配置字典
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 计算设备
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 设置优化器
        self.optimizer = self._create_optimizer()
        
        # 设置学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 初始化最佳模型状态
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 设置Weights & Biases
        if config['training']['use_wandb']:
            wandb.init(project="eeg-deep-learning",
                      config=config)
            
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        if self.config['training']['optimizer'].lower() == 'adam':
            return optim.Adam(self.model.parameters(),
                            lr=self.config['training']['learning_rate'],
                            weight_decay=self.config['training']['weight_decay'])
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['training']['optimizer']}")
            
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        if self.config['training']['scheduler']['name'].lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['scheduler']['T_max'],
                eta_min=self.config['training']['scheduler']['eta_min']
            )
        return None
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        保存检查点
        
        Args:
            epoch: 当前轮次
            val_loss: 验证损失
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # 保存最新检查点
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, 
                  os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))
        
        # 如果是最佳模型，单独保存
        if is_best:
            torch.save(checkpoint,
                      os.path.join(checkpoint_dir, 'best_model.pth'))
            
    def train_epoch(self) -> Tuple[float, float]:
        """
        训练一个轮次
        
        Returns:
            tuple: (平均训练损失, 训练准确率)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 使用tqdm显示进度条
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item(),
                            'acc': 100. * correct / total})
            
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self) -> Tuple[float, float]:
        """
        验证模型
        
        Returns:
            tuple: (平均验证损失, 验证准确率)
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
        return val_loss / len(self.val_loader), correct / total
    
    def train(self):
        """训练模型"""
        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            
            # 训练一个轮次
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
                
            # 记录指标
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            # 记录到Weights & Biases
            if self.config['training']['use_wandb']:
                wandb.log(metrics)
                
            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # 早停
            if self.patience_counter >= self.config['training']['early_stopping']['patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
                
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%") 