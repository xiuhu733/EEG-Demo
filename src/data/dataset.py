import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from typing import Tuple, Optional
import mne

class EEGDataset(Dataset):
    """EEG数据集类"""
    
    def __init__(self, 
                 dataset_name: str,
                 split: str = "train",
                 transform: Optional[callable] = None):
        """
        初始化EEG数据集
        
        Args:
            dataset_name: Hugging Face数据集名称
            split: 数据集划分（'train', 'validation', 或 'test'）
            transform: 数据转换函数
        """
        self.dataset = load_dataset(dataset_name, split=split)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取单个数据样本
        
        Args:
            idx: 数据索引
            
        Returns:
            tuple: (eeg_data, label)
        """
        item = self.dataset[idx]
        eeg_data = torch.tensor(item['eeg'], dtype=torch.float32)
        label = item['label']
        
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
                       low_freq: float = 1.0, 
                       high_freq: float = 40.0) -> np.ndarray:
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
        raw = mne.io.RawArray(data, info)
        raw.filter(low_freq, high_freq)
        return raw.get_data()
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        标准化数据
        
        Args:
            data: EEG数据
            
        Returns:
            np.ndarray: 标准化后的数据
        """
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
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
    train_dataset = EEGDataset(config['data']['dataset_name'], 
                              split='train',
                              transform=transform)
    val_dataset = EEGDataset(config['data']['dataset_name'], 
                            split='validation',
                            transform=transform)
    test_dataset = EEGDataset(config['data']['dataset_name'], 
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