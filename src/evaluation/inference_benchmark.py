import torch
import time
import numpy as np
from pathlib import Path
import yaml
import psutil
import os
from torch.utils.data import DataLoader
from src.models.model_factory import create_model
from src.data.dataset import EEGDataset, EEGTransform
from src.data.spatial_filter import CSPFilter
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(model_name):
    """加载模型配置"""
    config_path = f'src/config/{"default" if model_name == "eegnet" else "titans"}.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model(model_name, checkpoint_path):
    """加载模型和检查点"""
    config = load_config(model_name)
    model = create_model(config)
    
    print(f"\n加载检查点: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 打印检查点信息
        print(f"检查点信息:")
        print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            print(f"保存时的验证指标:")
            for metric, value in checkpoint['metrics'].items():
                print(f"- {metric}: {value:.4f}")
        
        # 验证模型状态
        print("\n模型状态验证:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"- {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
                if torch.isnan(param.data).any():
                    print(f"警告: 参数 {name} 包含 NaN 值!")
                if torch.isinf(param.data).any():
                    print(f"警告: 参数 {name} 包含 Inf 值!")
        
    except Exception as e:
        print(f"加载检查点时出错: {str(e)}")
        raise
    
    return model, config

def create_test_dataset(config, model_name, checkpoint):
    """创建测试数据集"""
    # 创建基础转换
    transform = EEGTransform(sampling_rate=config['data']['sampling_rate'])
    
    # 从检查点加载CSP滤波器
    if 'csp_filter_state' in checkpoint:
        csp_filter = CSPFilter(n_components=8)
        csp_filter.filters_ = checkpoint['csp_filter_state']['filters_']
        csp_filter.patterns_ = checkpoint['csp_filter_state']['patterns_']
        csp_filter.mean_ = checkpoint['csp_filter_state']['mean_']
        csp_filter.std_ = checkpoint['csp_filter_state']['std_']
        transform.csp_filter = csp_filter
    
    # 创建数据集
    dataset = EEGDataset(
        data_dir=config['data']['cache_dir'],
        dataset_type=config['data']['dataset_type'],
        split='test',
        transform=transform,
        window_size=config['data']['preprocessing']['window_size'],
        stride=config['data']['preprocessing']['stride']
    )
    
    return dataset

def analyze_predictions(model, data_loader, device):
    """分析模型预测的概率分布"""
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    
    print("\n预测概率分析:")
    print(f"平均概率: {np.mean(all_probs, axis=0)}")
    print(f"最大概率: {np.max(all_probs, axis=0)}")
    print(f"最小概率: {np.min(all_probs, axis=0)}")
    print(f"标准差: {np.std(all_probs, axis=0)}")
    
    # 绘制概率分布直方图
    plt.figure(figsize=(12, 4))
    for i in range(all_probs.shape[1]):
        plt.subplot(1, all_probs.shape[1], i+1)
        plt.hist(all_probs[:, i], bins=20)
        plt.title(f'Class {i} Probabilities')
    plt.tight_layout()
    plt.savefig('probability_distribution.png')
    plt.close()
    
    # 分析预测置信度
    confidences = np.max(all_probs, axis=1)
    print(f"\n预测置信度分析:")
    print(f"平均置信度: {np.mean(confidences):.4f}")
    print(f"最高置信度: {np.max(confidences):.4f}")
    print(f"最低置信度: {np.min(confidences):.4f}")
    
    return all_probs, all_preds, all_targets

def evaluate_accuracy(model, test_loader):
    """评估模型准确性"""
    model.eval()
    device = next(model.parameters()).device
    
    # 分析预测
    all_probs, all_preds, all_targets = analyze_predictions(model, test_loader, device)
    
    # 计算准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    
    # 打印详细的评估信息
    print(f"\n预测分布:")
    unique, counts = np.unique(all_preds, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"类别 {u}: {c} 个样本 ({c/len(all_preds)*100:.2f}%)")
    
    print(f"\n真实标签分布:")
    unique, counts = np.unique(all_targets, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"类别 {u}: {c} 个样本 ({c/len(all_targets)*100:.2f}%)")
    
    # 计算每个类别的准确率
    print("\n每个类别的准确率:")
    for class_id in np.unique(all_targets):
        class_mask = np.array(all_targets) == class_id
        class_acc = np.mean(np.array(all_preds)[class_mask] == class_id)
        print(f"类别 {class_id}: {class_acc*100:.2f}%")
    
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n详细分类报告:")
    print(classification_report(all_targets, all_preds))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return accuracy, all_preds, all_targets

def measure_inference_time(model, input_data, num_runs=100):
    """测量推理时间"""
    model.eval()
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
    
    # 测量时间
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_data)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

def measure_memory_usage(model, input_data):
    """测量内存使用"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.eval()
    with torch.no_grad():
        _ = model(input_data)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # 转换为MB
    return peak_memory

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型配置
    models = {
        'eegnet': 'runs/eegnet_20250214_155553/checkpoints/best_model.pth',
        'titans': 'runs/titanseeg_20250214_115329/checkpoints/best_model.pth'
    }
    
    results = {}
    
    for model_name, checkpoint_path in models.items():
        print(f"\n{'-'*20} 测试 {model_name.upper()} 模型 {'-'*20}")
        
        try:
            # 加载模型和检查点
            model, config = load_model(model_name, checkpoint_path)
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
            model = model.to(device)
            print(f"模型加载成功")
            
            # 打印模型参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"总参数量: {total_params:,}")
            print(f"可训练参数量: {trainable_params:,}")
            
            # 创建测试数据集（现在传入checkpoint）
            test_dataset = create_test_dataset(config, model_name, checkpoint)
            print(f"测试集大小: {len(test_dataset)} 个样本")
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,  # 用于测量单样本推理性能
                shuffle=False
            )
            
            # 获取单个样本用于性能测试
            sample_data, _ = next(iter(test_loader))
            sample_data = sample_data.to(device)
            
            # 1. 测量推理时间
            timing = measure_inference_time(model, sample_data)
            print(f"\n推理时间统计 (ms):")
            print(f"平均: {timing['mean']*1000:.2f} ± {timing['std']*1000:.2f}")
            print(f"最小: {timing['min']*1000:.2f}")
            print(f"最大: {timing['max']*1000:.2f}")
            
            # 2. 测量内存使用
            memory_usage = measure_memory_usage(model, sample_data)
            print(f"\n峰值显存使用: {memory_usage:.2f} MB")
            
            # 3. 测量准确性
            test_loader = DataLoader(
                test_dataset,
                batch_size=config['data']['batch_size'],
                shuffle=False
            )
            
            accuracy, predictions, targets = evaluate_accuracy(model, test_loader)
            print(f"\n测试集准确率: {accuracy*100:.2f}%")
            
            # 保存结果
            results[model_name] = {
                'inference_time': timing,
                'memory_usage': memory_usage,
                'accuracy': accuracy
            }
            
        except Exception as e:
            print(f"测试 {model_name} 时发生错误: {str(e)}")
            continue
    
    if len(results) > 0:
        # 比较结果
        print("\n\n性能对比:")
        print("-" * 50)
        print(f"{'指标':<20} {'EEGNet':<15} {'TitansEEG':<15}")
        print("-" * 50)
        print(f"{'平均推理时间(ms)':<20} {results.get('eegnet', {}).get('inference_time', {}).get('mean', 0)*1000:>15.2f} {results.get('titans', {}).get('inference_time', {}).get('mean', 0)*1000:>15.2f}")
        print(f"{'显存使用(MB)':<20} {results.get('eegnet', {}).get('memory_usage', 0):>15.2f} {results.get('titans', {}).get('memory_usage', 0):>15.2f}")
        print(f"{'准确率(%)':<20} {results.get('eegnet', {}).get('accuracy', 0)*100:>15.2f} {results.get('titans', {}).get('accuracy', 0)*100:>15.2f}")
        print("-" * 50)

if __name__ == '__main__':
    main() 