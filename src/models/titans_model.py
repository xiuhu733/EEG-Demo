import torch
import torch.nn as nn
from titans_pytorch import NeuralMemory

class TitansEEG(nn.Module):
    """
    EEG分类模型，使用Titans架构
    """
    def __init__(self, 
                 input_channels: int = 64,
                 num_classes: int = 4,
                 hidden_dim: int = 384,
                 chunk_size: int = 32,
                 dropout_rate: float = 0.5):
        """
        初始化Titans EEG模型
        
        Args:
            input_channels: 输入EEG通道数
            num_classes: 分类类别数
            hidden_dim: 隐藏层维度
            chunk_size: 记忆块大小
            dropout_rate: Dropout比率
        """
        super(TitansEEG, self).__init__()
        
        # 空间特征提取增强
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, (input_channels, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 32, (1, 5), padding=(0, 2), bias=False),  # 时间卷积
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 投影到高维空间，增加非线性变换
        self.input_proj = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 神经记忆模块
        self.memory = NeuralMemory(
            dim=hidden_dim,
            chunk_size=chunk_size
        )
        
        # 增强分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，形状为 (batch_size, channels, time_steps)
            
        Returns:
            torch.Tensor: 分类预测结果
        """
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch_size, 1, channels, time_steps)
        
        # 空间特征提取
        x = self.spatial_conv(x)  # (batch_size, 32, 1, time_steps)
        x = x.squeeze(2)  # (batch_size, 32, time_steps)
        x = x.transpose(1, 2)  # (batch_size, time_steps, 32)
        
        # 投影到高维空间
        x = self.input_proj(x)  # (batch_size, time_steps, hidden_dim)
        
        # 应用神经记忆
        retrieved, _ = self.memory(x)  # (batch_size, time_steps, hidden_dim)
        
        # 全局平均池化
        x = retrieved.mean(dim=1)  # (batch_size, hidden_dim)
        
        # 分类
        x = self.classifier(x)
        
        return x 