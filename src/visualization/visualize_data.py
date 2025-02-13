import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from scipy import signal
from scipy.fft import fft, fftfreq
import seaborn as sns
from src.data.dataset import EEGTransform

def load_config(config_path: str = 'src/config/default.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def plot_raw_signals(data: np.ndarray, sampling_rate: int, num_channels: int = 5):
    """绘制原始EEG信号波形"""
    time = np.arange(data.shape[1]) / sampling_rate
    
    plt.figure(figsize=(15, 10))
    for i in range(min(num_channels, data.shape[0])):
        plt.subplot(num_channels, 1, i+1)
        plt.plot(time, data[i], label=f'Channel {i+1}')
        plt.ylabel('Amplitude')
        plt.legend()
        if i == num_channels-1:
            plt.xlabel('Time (s)')
    plt.suptitle('Raw EEG Signals')
    plt.tight_layout()
    
    # 保存图像
    save_dir = Path('results/visualization')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'raw_signals.png')
    plt.close()

def plot_filtered_signals(data: np.ndarray, filtered_data: np.ndarray, 
                         sampling_rate: int, num_channels: int = 5):
    """绘制滤波前后的信号对比"""
    time = np.arange(data.shape[1]) / sampling_rate
    
    plt.figure(figsize=(15, 12))
    for i in range(min(num_channels, data.shape[0])):
        # 原始信号
        plt.subplot(num_channels, 2, 2*i+1)
        plt.plot(time, data[i], label=f'Channel {i+1}')
        plt.ylabel('Amplitude')
        plt.title('Raw')
        plt.legend()
        
        # 滤波后的信号
        plt.subplot(num_channels, 2, 2*i+2)
        plt.plot(time, filtered_data[i], label=f'Channel {i+1}', color='orange')
        plt.title('Filtered')
        plt.legend()
        
        if i == num_channels-1:
            plt.xlabel('Time (s)')
    plt.suptitle('Raw vs Filtered EEG Signals')
    plt.tight_layout()
    
    # 保存图像
    save_dir = Path('results/visualization')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'filtered_signals.png')
    plt.close()

def plot_spectrum(data: np.ndarray, filtered_data: np.ndarray, 
                 sampling_rate: int, num_channels: int = 5):
    """绘制频谱分析"""
    # 计算频率点
    freqs = fftfreq(data.shape[1], 1/sampling_rate)
    mask = freqs >= 0  # 只看正频率部分
    
    plt.figure(figsize=(15, 12))
    for i in range(min(num_channels, data.shape[0])):
        # 原始信号的频谱
        plt.subplot(num_channels, 2, 2*i+1)
        spectrum = np.abs(fft(data[i]))
        plt.plot(freqs[mask], spectrum[mask], label=f'Channel {i+1}')
        plt.ylabel('Magnitude')
        plt.title('Raw Spectrum')
        plt.legend()
        plt.grid(True)
        
        # 滤波后信号的频谱
        plt.subplot(num_channels, 2, 2*i+2)
        filtered_spectrum = np.abs(fft(filtered_data[i]))
        plt.plot(freqs[mask], filtered_spectrum[mask], 
                label=f'Channel {i+1}', color='orange')
        plt.title('Filtered Spectrum')
        plt.legend()
        plt.grid(True)
        
        if i == num_channels-1:
            plt.xlabel('Frequency (Hz)')
    plt.suptitle('Frequency Spectrum Analysis')
    plt.tight_layout()
    
    # 保存图像
    save_dir = Path('results/visualization')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'spectrum.png')
    plt.close()

def plot_spectrogram(data: np.ndarray, filtered_data: np.ndarray, 
                    sampling_rate: int, num_channels: int = 5):
    """绘制时频分析（频谱图）"""
    plt.figure(figsize=(15, 12))
    for i in range(min(num_channels, data.shape[0])):
        # 原始信号的频谱图
        plt.subplot(num_channels, 2, 2*i+1)
        f, t, Sxx = signal.spectrogram(data[i], fs=sampling_rate)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Raw Channel {i+1}')
        plt.colorbar(label='Intensity')
        
        # 滤波后信号的频谱图
        plt.subplot(num_channels, 2, 2*i+2)
        f, t, Sxx = signal.spectrogram(filtered_data[i], fs=sampling_rate)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.title(f'Filtered Channel {i+1}')
        plt.colorbar(label='Intensity')
        
        if i == num_channels-1:
            plt.xlabel('Time (s)')
    plt.suptitle('Time-Frequency Analysis')
    plt.tight_layout()
    
    # 保存图像
    save_dir = Path('results/visualization')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'spectrogram.png')
    plt.close()

def plot_channel_correlation(data: np.ndarray):
    """绘制通道间相关性热力图"""
    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(data)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=range(1, data.shape[0]+1),
                yticklabels=range(1, data.shape[0]+1))
    plt.title('Channel Correlation Matrix')
    plt.xlabel('Channel')
    plt.ylabel('Channel')
    
    # 保存图像
    save_dir = Path('results/visualization')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'channel_correlation.png')
    plt.close()

def main():
    """主函数"""
    # 加载配置
    config = load_config()
    
    # 加载数据
    data_dir = Path(config['data']['cache_dir']) / config['data']['dataset_type']
    data = np.load(data_dir / 'eeg_data.npy')
    
    # 创建数据转换器
    transform = EEGTransform(sampling_rate=config['data']['sampling_rate'])
    
    # 获取滤波后的数据
    filtered_data = transform.bandpass_filter(data)
    
    print("数据形状:", data.shape)
    print(f"采样率: {config['data']['sampling_rate']} Hz")
    
    # 绘制各种可视化
    print("\n生成可视化图像...")
    plot_raw_signals(data, config['data']['sampling_rate'])
    plot_filtered_signals(data, filtered_data, config['data']['sampling_rate'])
    plot_spectrum(data, filtered_data, config['data']['sampling_rate'])
    plot_spectrogram(data, filtered_data, config['data']['sampling_rate'])
    plot_channel_correlation(data)
    print("可视化完成！图像已保存在 results/visualization 目录下。")

if __name__ == '__main__':
    main() 