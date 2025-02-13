import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List
import mne
from pathlib import Path

class EEGDataset(Dataset):
    """EEG数据集类"""
    
    def __init__(self, 
                 data_dir: str,
                 dataset_type: str = "eegbci",
                 split: str = "train",
                 transform: Optional[callable] = None):
        """
        初始化EEG数据集
        
        Args:
            data_dir: 数据目录路径
            dataset_type: 数据集类型 ('sample' 或 'eegbci')
            split: 数据集划分（'train', 'val', 或 'test'）
            transform: 数据转换函数
        """
        self.data_dir = Path(data_dir) / dataset_type
        self.dataset_type = dataset_type
        self.transform = transform
        
        # 加载数据
        self.data = np.load(self.data_dir / 'eeg_data.npy')
        
        if dataset_type == 'eegbci':
            self.labels = np.load(self.data_dir / 'labels.npy')
        else:  # sample dataset
            # 为示例数据集创建伪标签（可以根据需要修改）
            self.labels = np.zeros(self.data.shape[1])
        
        # 加载通道信息
        if (self.data_dir / 'channels.txt').exists():
            with open(self.data_dir / 'channels.txt', 'r') as f:
                self.channels = f.read().splitlines()
        
        # 划分数据集
        total_samples = self.data.shape[1]
        indices = np.random.permutation(total_samples)
        
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)
        
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
        eeg_data = self.data[:, real_idx]
        label = self.labels[real_idx]
        
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
            data: EEG数据
            low_freq: 低频截止频率
            high_freq: 高频截止频率
            
        Returns:
            np.ndarray: 滤波后的数据
        """
        info = mne.create_info(ch_names=list(range(data.shape[0])),
                             sfreq=self.sampling_rate,
                             ch_types=['eeg'] * data.shape[0])
        raw = mne.io.RawArray(data.reshape(1, -1), info)
        raw.filter(low_freq, high_freq)
        return raw.get_data().reshape(-1)
    
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
        应用所有转换
        
        Args:
            data: 输入EEG数据
            
        Returns:
            torch.Tensor: 转换后的数据
        """
        data_np = data.numpy()
        # 应用带通滤波
        data_filtered = self.bandpass_filter(data_np)
        # 标准化
        data_normalized = self.normalize(data_filtered)
        return torch.from_numpy(data_normalized)

def create_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    transform = EEGTransform(sampling_rate=config['data']['sampling_rate'])
    
    # 创建数据集
    train_dataset = EEGDataset(config['data']['cache_dir'], 
                              dataset_type=config['data']['dataset_type'],
                              split='train',
                              transform=transform)
    
    val_dataset = EEGDataset(config['data']['cache_dir'],
                            dataset_type=config['data']['dataset_type'],
                            split='val',
                            transform=transform)
    
    test_dataset = EEGDataset(config['data']['cache_dir'],
                             dataset_type=config['data']['dataset_type'],
                             split='test',
                             transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset,
                            batch_size=config['data']['batch_size'],
                            shuffle=True,
                            num_workers=config['data']['num_workers'])
    
    val_loader = DataLoader(val_dataset,
                          batch_size=config['data']['batch_size'],
                          shuffle=False,
                          num_workers=config['data']['num_workers'])
    
    test_loader = DataLoader(test_dataset,
                           batch_size=config['data']['batch_size'],
                           shuffle=False,
                           num_workers=config['data']['num_workers'])
    
    return train_loader, val_loader, test_loader 