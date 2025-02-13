import os
import argparse
import subprocess
import yaml
from datetime import datetime

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    """保存配置文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def train_model(config_path):
    """训练模型"""
    cmd = f"python src/training/train.py --config {config_path}"
    subprocess.run(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser(description='比较不同的EEG分类模型')
    parser.add_argument('--model', type=str, choices=['eegnet', 'titans'], required=True,
                      help='选择要训练的模型 (eegnet 或 titans)')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='批量大小 (可选)')
    parser.add_argument('--learning_rate', type=float, default=None,
                      help='学习率 (可选)')
    parser.add_argument('--epochs', type=int, default=None,
                      help='训练轮次 (可选)')
    args = parser.parse_args()

    # 选择配置文件
    if args.model == 'eegnet':
        config_path = 'src/config/default.yaml'
    else:
        config_path = 'src/config/titans.yaml'

    # 加载配置
    config = load_config(config_path)

    # 更新配置（如果提供了命令行参数）
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    # 创建运行目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.model}_{timestamp}"
    run_dir = os.path.join('runs', run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 保存修改后的配置
    exp_config_path = os.path.join(run_dir, 'config.yaml')
    save_config(config, exp_config_path)

    # 打印实验信息
    print("\n" + "="*50)
    print(f"开始训练 {args.model.upper()} 模型")
    print(f"运行目录: {run_dir}")
    print(f"批量大小: {config['data']['batch_size']}")
    print(f"学习率: {config['training']['learning_rate']}")
    print(f"训练轮次: {config['training']['epochs']}")
    print("="*50 + "\n")

    # 训练模型
    train_model(exp_config_path)

if __name__ == '__main__':
    main() 