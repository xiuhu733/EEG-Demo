import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from torch.utils.data import DataLoader
import os

class EEGEvaluator:
    """EEG模型评估器"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 test_loader: DataLoader,
                 config: Dict,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化评估器
        
        Args:
            model: 神经网络模型
            test_loader: 测试数据加载器
            config: 配置字典
            device: 计算设备
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.metrics = config['evaluation']['metrics']
        
    def evaluate(self) -> Dict:
        """
        评估模型性能
        
        Returns:
            dict: 评估指标字典
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        results = {}
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # 计算各项指标
        if 'accuracy' in self.metrics:
            results['accuracy'] = (all_preds == all_targets).mean()
            
        if 'precision' in self.metrics:
            results['precision'] = self._calculate_precision(all_preds, all_targets)
            
        if 'recall' in self.metrics:
            results['recall'] = self._calculate_recall(all_preds, all_targets)
            
        if 'f1' in self.metrics:
            results['f1'] = self._calculate_f1(all_preds, all_targets)
            
        # 生成可视化
        if self.config['evaluation']['visualization']['confusion_matrix']:
            self.plot_confusion_matrix(all_preds, all_targets)
            
        if self.config['evaluation']['visualization']['roc_curve']:
            self.plot_roc_curve(all_probs, all_targets)
            
        return results
    
    def _calculate_precision(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """计算精确率"""
        tp = np.sum((preds == targets) & (preds == 1))
        fp = np.sum((preds != targets) & (preds == 1))
        return tp / (tp + fp + 1e-8)
    
    def _calculate_recall(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """计算召回率"""
        tp = np.sum((preds == targets) & (preds == 1))
        fn = np.sum((preds != targets) & (preds == 0))
        return tp / (tp + fn + 1e-8)
    
    def _calculate_f1(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """计算F1分数"""
        precision = self._calculate_precision(preds, targets)
        recall = self._calculate_recall(preds, targets)
        return 2 * precision * recall / (precision + recall + 1e-8)
    
    def plot_confusion_matrix(self, preds: np.ndarray, targets: np.ndarray):
        """
        绘制混淆矩阵
        
        Args:
            preds: 预测结果
            targets: 真实标签
        """
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # 保存图像
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/confusion_matrix.png')
        plt.close()
        
    def plot_roc_curve(self, probs: np.ndarray, targets: np.ndarray):
        """
        绘制ROC曲线
        
        Args:
            probs: 预测概率
            targets: 真实标签
        """
        plt.figure(figsize=(10, 8))
        
        if probs.shape[1] == 2:  # 二分类
            fpr, tpr, _ = roc_curve(targets, probs[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:  # 多分类
            for i in range(probs.shape[1]):
                fpr, tpr, _ = roc_curve((targets == i).astype(int), probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
                
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # 保存图像
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/roc_curve.png')
        plt.close()
        
    def plot_learning_curves(self, history: Dict):
        """
        绘制学习曲线
        
        Args:
            history: 训练历史记录
        """
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # 保存图像
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/learning_curves.png')
        plt.close() 