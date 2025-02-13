import argparse
import yaml
import torch
from ..data.dataset import create_data_loaders
from ..models.eegnet import EEGNet
from .trainer import EEGTrainer
from ..evaluation.evaluator import EEGEvaluator

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train EEG model')
    parser.add_argument('--config', type=str, default='config/default.yaml',
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # 创建训练器
    trainer = EEGTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # 训练模型
    print("Starting training...")
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