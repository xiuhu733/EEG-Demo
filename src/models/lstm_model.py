import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """时间注意力层"""
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # 在序列维度上归一化
        attended = torch.bmm(x.transpose(1, 2), attention_weights)  # (batch_size, hidden_dim, 1)
        return attended.squeeze(-1), attention_weights

class EEGLSTM(nn.Module):
    """
    EEG分类的LSTM模型
    
    特点：
    1. 空间特征提取层
    2. 双向LSTM层
    3. 时间注意力机制
    4. 残差连接
    5. 分层Dropout
    """
    def __init__(self,
                 input_channels: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 4,
                 dropout_rate: float = 0.3,
                 bidirectional: bool = True):
        """
        初始化LSTM模型
        
        Args:
            input_channels: 输入EEG通道数
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类类别数
            dropout_rate: Dropout比率
            bidirectional: 是否使用双向LSTM
        """
        super(EEGLSTM, self).__init__()
        
        # 1. 改进的空间特征提取
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, (input_channels, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(dropout_rate/2),
            
            # 减少卷积层数量,使用更小的卷积核
            nn.Conv2d(16, 32, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(dropout_rate/2)
        )
        
        # 2. 简化的特征投影
        self.input_proj = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate/2)
        )
        
        # 3. 改进的LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 4. 改进的注意力机制
        lstm_hidden = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.Tanh(),
            nn.Linear(lstm_hidden // 2, 1)
        )
        
        # 5. 改进的分类器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden, num_classes)
        )
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """使用正确的初始化方法"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'conv' in name and len(param.shape) > 1:
                    # 只对卷积层的权重使用 kaiming 初始化
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'lstm' in name:
                    # 使用正交初始化LSTM权重
                    if len(param.shape) > 1:  # 只对权重矩阵使用正交初始化
                        nn.init.orthogonal_(param)
                    else:  # 对偏置使用零初始化
                        nn.init.zeros_(param)
                elif len(param.shape) > 1:  # 只对2D及以上维度的权重使用xavier初始化
                    # 线性层使用xavier初始化
                    nn.init.xavier_uniform_(param)
                else:  # 1D权重使用均匀初始化
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 1. 空间特征提取
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.spatial_conv(x)
        x = x.squeeze(2)  # 移除空间维度
        x = x.transpose(1, 2)  # (batch_size, seq_len, features)
        
        # 2. 特征投影
        x = self.input_proj(x)
        
        # 3. LSTM处理
        x, _ = self.lstm(x)
        
        # 4. 注意力机制
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        x = torch.bmm(x.transpose(1, 2), attention_weights)  # (batch_size, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch_size, hidden_dim)
        
        # 5. 分类
        x = self.classifier(x)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重（用于可视化）
        
        Args:
            x: 输入数据
            
        Returns:
            torch.Tensor: 注意力权重
        """
        self.eval()
        with torch.no_grad():
            # 重复前向传播直到注意力层
            x = x.unsqueeze(1)
            x = self.spatial_conv(x)
            x = x.squeeze(2).transpose(1, 2)
            x = self.input_proj(x)
            x, _ = self.lstm[0](x)
            _, attention_weights = self.attention(x)
            
        return attention_weights.squeeze(-1)  # (batch_size, seq_len) 