import os
import torch
import wandb
from tqdm import tqdm
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    cohen_kappa_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        config: Dict,
        writer: 'torch.utils.tensorboard.SummaryWriter',
        exp_dir: str
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.writer = writer
        self.exp_dir = exp_dir
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # 记录训练历史
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'lr': []
        }
        
        # 打印模型信息
        print("\nModel Architecture:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """记录指标到tensorboard和wandb"""
        # 记录到tensorboard
        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, step)
        
        # 记录到wandb
        if self.config['training']['use_wandb']:
            wandb.log({f'{prefix}_{k}': v for k, v in metrics.items()}, step=step)
    
    def plot_confusion_matrix(self, preds: list, labels: list, epoch: int):
        """绘制混淆矩阵"""
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # 保存到文件
        cm_path = os.path.join(self.exp_dir, f'confusion_matrix_epoch_{epoch}.png')
        plt.savefig(cm_path)
        plt.close()
        
        # 记录到tensorboard和wandb
        self.writer.add_figure('Confusion Matrix', plt.gcf(), epoch)
        if self.config['training']['use_wandb']:
            wandb.log({'confusion_matrix': wandb.Image(cm_path)}, step=epoch)
    
    def plot_learning_curves(self):
        """绘制学习曲线"""
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Val')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 准确率曲线
        plt.subplot(1, 3, 2)
        plt.plot(self.history['train_acc'], label='Train')
        plt.plot(self.history['val_acc'], label='Val')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # F1分数曲线
        plt.subplot(1, 3, 3)
        plt.plot(self.history['train_f1'], label='Train')
        plt.plot(self.history['val_f1'], label='Val')
        plt.title('F1 Score Curves')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        # 保存到文件
        curves_path = os.path.join(self.exp_dir, 'learning_curves.png')
        plt.savefig(curves_path)
        plt.close()
        
        # 记录到wandb
        if self.config['training']['use_wandb']:
            wandb.log({'learning_curves': wandb.Image(curves_path)})
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        running_loss = 0.0
        running_acc = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 计算批次准确率
            preds = output.argmax(dim=1)
            acc = (preds == target).float().mean().item()
            
            # 更新统计
            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            running_acc = 0.9 * running_acc + 0.1 * acc
            
            # 收集预测结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{running_loss:.4f}',
                'acc': f'{running_acc:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        # 计算指标
        metrics = self.compute_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics, all_preds, all_labels
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        running_loss = 0.0
        running_acc = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # 计算批次准确率
                preds = output.argmax(dim=1)
                acc = (preds == target).float().mean().item()
                
                # 更新统计
                running_loss = 0.9 * running_loss + 0.1 * loss.item()
                running_acc = 0.9 * running_acc + 0.1 * acc
                
                # 收集结果
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{running_loss:.4f}',
                    'acc': f'{running_acc:.4f}'
                })
        
        # 计算指标
        metrics = self.compute_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics, all_preds, all_labels
    
    def compute_metrics(self, preds: list, labels: list) -> Dict[str, float]:
        """计算评估指标"""
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall': recall_score(labels, preds, average='weighted', zero_division=0),
            'f1': f1_score(labels, preds, average='weighted', zero_division=0),
            'kappa': cohen_kappa_score(labels, preds)
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        # 保存最新的检查点
        checkpoint_path = os.path.join(
            self.config['training']['checkpoint_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_model_path = os.path.join(
                self.config['training']['checkpoint_dir'],
                'best_model.pth'
            )
            torch.save(checkpoint, best_model_path)
            print(f"\n保存最佳模型，验证准确率: {metrics['accuracy']:.4f}")
    
    def train(self):
        """训练模型"""
        print("\n开始训练...")
        print(f"训练设备: {self.device}")
        print(f"批量大小: {self.config['data']['batch_size']}")
        print(f"学习率: {self.config['training']['learning_rate']}")
        print(f"总轮次: {self.config['training']['epochs']}")
        
        epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping']['patience']
        min_delta = self.config['training']['early_stopping']['min_delta']
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 50)
            
            # 训练
            train_metrics, train_preds, train_labels = self.train_epoch()
            
            # 验证
            val_metrics, val_preds, val_labels = self.validate()
            
            # 更新学习率
            old_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            new_lr = self.scheduler.get_last_lr()[0]
            
            # 更新历史记录
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['lr'].append(new_lr)
            
            # 记录指标
            self.log_metrics(train_metrics, epoch, 'train')
            self.log_metrics(val_metrics, epoch, 'val')
            self.writer.add_scalar('learning_rate', new_lr, epoch)
            
            # 每5个epoch绘制一次混淆矩阵
            if epoch % 5 == 0:
                self.plot_confusion_matrix(val_preds, val_labels, epoch)
            
            # 打印指标
            print("\n训练指标:")
            for metric, value in train_metrics.items():
                print(f"{metric}: {value:.4f}")
            
            print("\n验证指标:")
            for metric, value in val_metrics.items():
                print(f"{metric}: {value:.4f}")
            
            print(f"\n学习率: {old_lr:.6f} -> {new_lr:.6f}")
            
            # 检查是否是最佳模型
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                print(f"\n发现新的最佳模型! 最佳验证准确率: {self.best_val_acc:.4f}")
            else:
                self.patience_counter += 1
                print(f"\n模型性能未提升，已经 {self.patience_counter} 轮")
            
            # 保存检查点
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # 早停
            if self.patience_counter >= patience:
                print(f"\n触发早停机制，在第 {epoch} 轮停止训练")
                print(f"最佳验证准确率: {self.best_val_acc:.4f}")
                break
        
        # 训练结束，绘制学习曲线
        self.plot_learning_curves()
        print("\n训练完成!") 