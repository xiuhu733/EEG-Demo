import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List
import mne
from pathlib import Path
from scipy.signal import butter, filtfilt

class EEGDataset(Dataset):
    """EEG数据集类"""
    
    def __init__(self, 
                 data_dir: str,
                 dataset_type: str = "eegbci",
                 split: str = "train",
                 transform: Optional[callable] = None,
                 window_size: int = 640,
                 stride: int = 320):
        """
        初始化EEG数据集
        
        Args:
            data_dir: 数据目录路径
            dataset_type: 数据集类型 ('sample' 或 'eegbci')
            split: 数据集划分（'train', 'val', 或 'test'）
            transform: 数据转换函数
            window_size: 时间窗口大小
            stride: 滑动步长
        """
        self.data_dir = Path(data_dir) / dataset_type
        self.dataset_type = dataset_type
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        
        # 加载数据
        raw_data = np.load(self.data_dir / 'eeg_data.npy')
        
        if dataset_type == 'eegbci':
            self.labels = np.load(self.data_dir / 'labels.npy')
        else:  # sample dataset
            # 为示例数据集创建伪标签
            self.labels = np.zeros(raw_data.shape[1])
        
        # 生成通道名称
        self.channels = [f'EEG{i+1:03d}' for i in range(raw_data.shape[0])]
        
        # 使用滑动窗口切分数据
        self.data = []
        self.segment_labels = []
        
        # 对每个时间点进行滑动窗口切分
        num_segments = (raw_data.shape[1] - window_size) // stride + 1
        for i in range(num_segments):
            start = i * stride
            end = start + window_size
            segment = raw_data[:, start:end]
            self.data.append(segment)
            # 使用该窗口中最多的标签作为该段的标签
            segment_label = self.labels[start:end]
            self.segment_labels.append(np.bincount(segment_label.astype(int)).argmax())
        
        self.data = np.array(self.data)  # (num_segments, channels, window_size)
        self.segment_labels = np.array(self.segment_labels)
        
        # 划分数据集
        total_segments = len(self.data)
        indices = np.random.permutation(total_segments)
        
        train_size = int(0.8 * total_segments)
        val_size = int(0.1 * total_segments)
        
        if split == 'train':
            self.indices = indices[:train_size]
        elif split == 'val':
            self.indices = indices[train_size:train_size + val_size]
        else:  # test
            self.indices = indices[train_size + val_size:]
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取单个数据样本
        
        Args:
            idx: 数据索引
            
        Returns:
            tuple: (eeg_data, label)
        """
        real_idx = self.indices[idx]
        eeg_data = self.data[real_idx]  # 获取一个时间窗口的数据
        label = self.segment_labels[real_idx]
        
        # 转换为torch张量
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        label = torch.tensor(int(label), dtype=torch.long)
        
        if self.transform:
            eeg_data = self.transform(eeg_data)
            
        return eeg_data, label

class EEGTransform:
    """EEG数据预处理转换类"""
    
    def __init__(self, sampling_rate: int = 250):
        """
        初始化EEG数据转换
        
        Args:
            sampling_rate: 采样率（Hz）
        """
        self.sampling_rate = sampling_rate
        
    def bandpass_filter(self, data: np.ndarray, 
                       low_freq: float = 7.0, 
                       high_freq: float = 30.0) -> np.ndarray:
        """
        应用带通滤波器
        
        Args:
            data: EEG数据，形状为 (channels, time_steps)
            low_freq: 低频截止频率
            high_freq: 高频截止频率
            
        Returns:
            np.ndarray: 滤波后的数据
        """
        # 计算归一化截止频率
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # 设计滤波器
        b, a = butter(N=4, Wn=[low, high], btype='band')
        
        # 应用滤波器
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i] = filtfilt(b, a, data[i])
            
        return filtered_data
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        标准化数据
        
        Args:
            data: EEG数据
            
        Returns:
            np.ndarray: 标准化后的数据
        """
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        转换EEG数据
        
        Args:
            data: 输入数据，形状为 (channels, time_steps)
            
        Returns:
            torch.Tensor: 转换后的数据，形状为 (channels, time_steps)
        """
        # 确保输入是二维的
        if data.dim() == 1:
            data = data.unsqueeze(0)
        elif data.dim() > 2:
            raise ValueError(f"输入数据维度应为1或2，但得到{data.dim()}维")
        
        # 转换为NumPy数组进行处理
        data_np = data.numpy()
        
        # 应用带通滤波
        filtered_data = self.bandpass_filter(data_np)
        
        # 标准化
        normalized_data = self.normalize(filtered_data)
        
        # 转换回PyTorch张量并确保形状正确
        output = torch.from_numpy(normalized_data).float()
        
        # 确保输出是(channels, time_steps)形状
        if output.dim() != 2:
            raise ValueError(f"输出数据维度应为2，但得到{output.dim()}维")
            
        return output

def create_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    transform = EEGTransform(sampling_rate=config['data']['sampling_rate'])
    
    # 从配置中获取窗口参数
    window_size = config['data']['preprocessing']['window_size']
    stride = config['data']['preprocessing']['stride']
    
    # 创建数据集
    train_dataset = EEGDataset(config['data']['cache_dir'], 
                              dataset_type=config['data']['dataset_type'],
                              split='train',
                              transform=transform,
                              window_size=window_size,
                              stride=stride)
    
    val_dataset = EEGDataset(config['data']['cache_dir'],
                            dataset_type=config['data']['dataset_type'],
                            split='val',
                            transform=transform,
                            window_size=window_size,
                            stride=stride)
    
    test_dataset = EEGDataset(config['data']['cache_dir'],
                             dataset_type=config['data']['dataset_type'],
                             split='test',
                             transform=transform,
                             window_size=window_size,
                             stride=stride)
    
    # 设置数据加载器参数
    loader_kwargs = {
        'num_workers': min(config['data']['num_workers'], 4),  # 限制worker数量
        'pin_memory': torch.cuda.is_available(),  # 如果有GPU则启用pin_memory
        'persistent_workers': True if config['data']['num_workers'] > 0 else False,  # 保持worker进程存活
    }
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset,
                            batch_size=config['data']['batch_size'],
                            shuffle=True,
                            **loader_kwargs)
    
    val_loader = DataLoader(val_dataset,
                          batch_size=config['data']['batch_size'],
                          shuffle=False,
                          **loader_kwargs)
    
    test_loader = DataLoader(test_dataset,
                           batch_size=config['data']['batch_size'],
                           shuffle=False,
                           **loader_kwargs)
    
    return train_loader, val_loader, test_loader 