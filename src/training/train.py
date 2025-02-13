import argparse
import yaml
import torch
from src.data.dataset import create_data_loaders
from src.models.eegnet import EEGNet
from src.training.trainer import EEGTrainer
from src.evaluation.evaluator import EEGEvaluator

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train EEG model')
    parser.add_argument('--config', type=str, default='src/config/default.yaml',
                      help='Path to configuration file')
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """主训练函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 打印GPU信息
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
        # 设置GPU相关配置
        torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # 创建模型
    model = EEGNet(
        input_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate'],
        kernel_length=config['model']['kernel_length'],
        num_filters=config['model']['num_filters'],
        pool_size=config['model']['pool_size']
    )
    
    # 打印模型信息
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 创建训练器
    trainer = EEGTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # 训练模型
    print("\nStarting training...")
    trainer.train()
    print("Training completed!")
    
    # 评估模型
    print("\nEvaluating model...")
    evaluator = EEGEvaluator(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    results = evaluator.evaluate()
    
    # 打印评估结果
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main() 