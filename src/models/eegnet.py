import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    """
    EEGNet: 用于EEG信号分类的深度卷积神经网络
    
    参考论文：
    Lawhern et al. EEGNet: A Compact Convolutional Neural Network for EEG-based 
    Brain-Computer Interfaces. Journal of Neural Engineering, 2018.
    """
    
    def __init__(self, 
                 input_channels: int = 64,
                 num_classes: int = 4,
                 dropout_rate: float = 0.5,
                 kernel_length: int = 64,
                 num_filters: int = 8,
                 pool_size: int = 8):
        """
        初始化EEGNet模型
        
        Args:
            input_channels: 输入EEG通道数
            num_classes: 分类类别数
            dropout_rate: Dropout比率
            kernel_length: 时间卷积核长度
            num_filters: 滤波器数量
            pool_size: 池化大小
        """
        super(EEGNet, self).__init__()
        
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        
        # 第一层：时间卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_filters, (1, kernel_length), padding='same', bias=False),
            nn.BatchNorm2d(num_filters)
        )
        
        # 第二层：空间卷积
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, (input_channels, 1),
                     groups=num_filters, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.ELU(),
            nn.AvgPool2d((1, pool_size)),
            nn.Dropout(dropout_rate)
        )
        
        # 第三层：可分离卷积
        self.separable_conv = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 4, (1, 16),
                     padding='same', groups=num_filters * 2, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.ELU(),
            nn.AvgPool2d((1, pool_size)),
            nn.Dropout(dropout_rate)
        )
        
        # 输出层
        self.classifier = nn.Linear(num_filters * 4, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，形状为 (batch_size, channels, time_steps)
            
        Returns:
            torch.Tensor: 分类预测结果
        """
        # 添加通道维度
        x = x.unsqueeze(1)
        
        # 时间卷积
        x = self.conv1(x)
        
        # 空间卷积
        x = self.depthwise_conv(x)
        
        # 可分离卷积
        x = self.separable_conv(x)
        
        # 展平
        x = x.mean(dim=3)  # 在时间维度上平均池化
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别概率
        
        Args:
            x: 输入数据
            
        Returns:
            torch.Tensor: 类别概率
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
            
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        Args:
            x: 输入数据
            
        Returns:
            torch.Tensor: 预测的类别
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1) 