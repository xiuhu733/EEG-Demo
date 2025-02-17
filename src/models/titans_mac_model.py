import torch
import torch.nn as nn
from titans_pytorch import MemoryAsContextTransformer

class TitansEEGMAC(nn.Module):
    """
    简化版的 TitansEEGMAC 模型，基于官方示例实现
    """
    def __init__(self, 
                 input_channels: int = 64,
                 num_classes: int = 4,
                 hidden_dim: int = 256,
                 chunk_size: int = 128):
        super(TitansEEGMAC, self).__init__()
        
        # 1. 投影层：将输入通道映射到隐藏维度
        self.input_proj = nn.Linear(input_channels, hidden_dim)
        
        # 2. MAC Transformer (基于官方示例配置)
        self.mac_transformer = MemoryAsContextTransformer(
            num_tokens = hidden_dim,
            dim = hidden_dim,
            depth = 2,
            segment_len = chunk_size,
            num_persist_mem_tokens = 4,
            num_longterm_mem_tokens = 16
        )
        
        # 3. 分类头
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，形状为 (batch_size, channels, time_steps)
            
        Returns:
            torch.Tensor: 分类预测结果
        """
        # 1. 转置为 (batch_size, time_steps, channels)
        x = x.transpose(1, 2).contiguous()
        
        # 2. 投影到隐藏维度
        x = self.input_proj(x)  # (batch_size, time_steps, hidden_dim)
        
        # 3. 准备 MAC Transformer 的输入
        batch_size = x.size(0)
        segment_len = self.mac_transformer.segment_len
        
        # 确保序列长度是 segment_len 的倍数
        if x.size(1) % segment_len != 0:
            pad_len = segment_len - (x.size(1) % segment_len)
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
        
        # 重塑为 MAC Transformer 所需的形状
        num_segments = x.size(1) // segment_len
        x = x.view(batch_size, num_segments, 1, segment_len, 1, -1)
        
        # 4. 通过 MAC Transformer
        x = self.mac_transformer(x)  # (batch_size, total_seq_len, hidden_dim)
        
        # 5. 全局平均池化
        x = x.mean(dim=1)  # (batch_size, hidden_dim)
        
        # 6. 分类
        x = self.classifier(x)
        
        return x 