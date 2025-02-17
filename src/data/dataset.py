import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List
import mne
from pathlib import Path
from scipy.signal import butter, filtfilt
from .spatial_filter import CSPFilter, SpatialAttention

class EEGDataset(Dataset):
    """EEG数据集类"""
    
    def __init__(self, 
                 data_dir: str,
                 dataset_type: str = "eegbci",
                 split: str = "train",
                 transform: Optional[callable] = None,
                 augment: Optional[callable] = None,
                 window_size: int = 640,
                 stride: int = 320,
                 spatial_filter_config=None):
        """
        初始化EEG数据集
        
        Args:
            data_dir: 数据目录路径
            dataset_type: 数据集类型 ('sample' 或 'eegbci')
            split: 数据集划分（'train', 'val', 或 'test'）
            transform: 数据转换函数
            augment: 数据增强函数
            window_size: 时间窗口大小
            stride: 滑动步长
            spatial_filter_config: 空间滤波器配置
        """
        super().__init__()
        self.data_dir = Path(data_dir) / dataset_type
        self.dataset_type = dataset_type
        self.transform = transform
        self.augment = augment
        self.window_size = window_size
        self.stride = stride
        self.split = split  # 保存split参数为实例属性
        
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
        
        # Initialize spatial filter if enabled
        self.spatial_filter = None
        if spatial_filter_config and spatial_filter_config['enabled']:
            self.spatial_filter = CSPFilter(
                n_components=spatial_filter_config['n_components']
            )
            if split == 'train':
                self.spatial_filter.fit(self.data, self.segment_labels)
        
        # Initialize spatial attention if enabled
        self.spatial_attention = None
        if spatial_filter_config and spatial_filter_config['attention']:
            n_channels = self.spatial_filter.n_components if self.spatial_filter else self.data.shape[1]
            self.spatial_attention = SpatialAttention(n_channels)
        
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
        
        # 应用空间滤波（如果启用）
        if self.spatial_filter is not None:
            eeg_data = self.spatial_filter.transform(eeg_data.reshape(1, *eeg_data.shape))[0]
        
        # 应用空间注意力（如果启用）
        if self.spatial_attention is not None:
            eeg_data = self.spatial_attention(eeg_data.unsqueeze(0))[0]
        
        # 转换为torch张量
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        label = torch.tensor(int(label), dtype=torch.long)
        
        # 应用数据增强（仅在训练集上）
        if self.split == 'train' and self.augment is not None:
            if np.random.random() < 0.5:
                eeg_data = eeg_data + torch.randn_like(eeg_data) * 0.1  # Add Gaussian noise
            if np.random.random() < 0.3:
                eeg_data = torch.flip(eeg_data, dims=[0])  # Random channel flip
            if np.random.random() < 0.3:
                mask_size = int(0.1 * eeg_data.shape[1])
                mask_start = np.random.randint(0, eeg_data.shape[1] - mask_size)
                eeg_data[:, mask_start:mask_start+mask_size] = 0  # Time masking
        
        # 应用数据转换
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
        self.csp_filter = None
        
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
        应用所有预处理步骤
        
        Args:
            data: 输入数据，形状为 (channels, time_steps)
            
        Returns:
            torch.Tensor: 预处理后的数据
        """
        # 转换为numpy数组进行处理
        data_np = data.numpy()
        
        # 1. 带通滤波
        data_filtered = self.bandpass_filter(data_np)
        
        # 2. 应用CSP滤波（如果可用）
        if self.csp_filter is not None:
            # 扩展维度以匹配CSP的输入要求
            data_filtered = np.expand_dims(data_filtered, 0)  # (1, channels, time_steps)
            data_filtered = self.csp_filter.transform(data_filtered)
            data_filtered = np.squeeze(data_filtered, 0)  # (channels, time_steps)
            
            # 应用归一化
            if self.csp_filter.mean_ is not None and self.csp_filter.std_ is not None:
                data_filtered = (data_filtered - self.csp_filter.mean_[:, None]) / self.csp_filter.std_[:, None]
        
        # 转换回PyTorch张量
        return torch.from_numpy(data_filtered).float()

def create_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 创建数据转换器
    transform = EEGTransform(sampling_rate=config['data']['sampling_rate'])
    
    # 创建数据增强器（如果启用）
    augment = None
    if config['data']['augmentation']['enabled']:
        from .augmentation import EEGAugmentation
        augment = EEGAugmentation(
            p_noise=config['data']['augmentation']['p_noise'],
            p_flip=config['data']['augmentation']['p_flip'],
            p_mask=config['data']['augmentation']['p_mask'],
            noise_scale=config['data']['augmentation']['noise_scale'],
            mask_ratio=config['data']['augmentation']['mask_ratio']
        )
    
    # 创建空间滤波器（如果启用）
    spatial_filter = None
    if config['data']['spatial_filter']['enabled']:
        from .spatial_filter import CSPFilter
        spatial_filter = CSPFilter(
            n_components=config['data']['spatial_filter']['n_components']
        )
    
    # 从配置中获取窗口参数
    window_size = config['data']['preprocessing']['window_size']
    stride = config['data']['preprocessing']['stride']
    
    # 创建数据集
    train_dataset = EEGDataset(
        config['data']['cache_dir'], 
        dataset_type=config['data']['dataset_type'],
        split='train',
        transform=transform,
        augment=augment,
        window_size=window_size,
        stride=stride
    )
    
    val_dataset = EEGDataset(
        config['data']['cache_dir'],
        dataset_type=config['data']['dataset_type'],
        split='val',
        transform=transform,
        window_size=window_size,
        stride=stride
    )
    
    test_dataset = EEGDataset(
        config['data']['cache_dir'],
        dataset_type=config['data']['dataset_type'],
        split='test',
        transform=transform,
        window_size=window_size,
        stride=stride
    )
    
    # 如果启用了CSP，训练并应用空间滤波
    if spatial_filter is not None:
        # 获取所有训练数据
        X_train = train_dataset.data
        y_train = train_dataset.segment_labels
        
        # 训练CSP滤波器
        spatial_filter.fit(X_train, y_train)
        
        # 应用CSP滤波
        train_dataset.data = spatial_filter.transform(train_dataset.data)
        val_dataset.data = spatial_filter.transform(val_dataset.data)
        test_dataset.data = spatial_filter.transform(test_dataset.data)
    
    # 设置数据加载器参数
    loader_kwargs = {
        'num_workers': min(config['data']['num_workers'], 4),  # 限制worker数量
        'pin_memory': torch.cuda.is_available(),  # 如果有GPU则启用pin_memory
        'persistent_workers': True if config['data']['num_workers'] > 0 else False,  # 保持worker进程存活
    }
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        **loader_kwargs
    )
    
    return train_loader, val_loader, test_loader 