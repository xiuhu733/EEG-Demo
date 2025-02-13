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
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Titans神经记忆模块
        self.memory = NeuralMemory(
            dim=hidden_dim,
            chunk_size=chunk_size
        )
        
        # 输出层
        self.classifier = nn.Sequential(
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
        # 调整输入维度
        if x.dim() == 3:
            batch_size, channels, time_steps = x.size()
            x = x.transpose(1, 2)  # (batch_size, time_steps, channels)
        
        # 投影到高维空间
        x = self.input_proj(x)  # (batch_size, time_steps, hidden_dim)
        
        # 应用神经记忆
        retrieved, _ = self.memory(x)
        
        # 全局平均池化
        x = retrieved.mean(dim=1)  # (batch_size, hidden_dim)
        
        # 分类
        x = self.classifier(x)
        
        return x 