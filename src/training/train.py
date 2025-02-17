import os
import torch
import yaml
import wandb
import argparse
import warnings
from torch.utils.data import DataLoader
from datetime import datetime
from src.models.model_factory import create_model
from src.data.dataset import EEGDataset
from src.utils.trainer import Trainer

# 忽略特定的警告
warnings.filterwarnings('ignore', message='Using padding=\'same\'')

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建运行目录结构
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{config['model']['name']}_{timestamp}"
    run_dir = os.path.join('runs', run_name)
    
    # 创建子目录
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    wandb_dir = os.path.join(run_dir, 'wandb')
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(wandb_dir, exist_ok=True)
    
    # 更新配置中的目录
    config['training']['checkpoint_dir'] = checkpoints_dir
    
    # 保存配置文件
    config_path = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 初始化wandb
    if config['training']['use_wandb']:
        try:
            # 设置wandb配置
            wandb_config = {
                "project": "eeg-classification",
                "name": run_name,
                "config": config,
                "dir": wandb_dir,
                "tags": [config['model']['name'], config['data']['dataset_type']],
                "notes": "EEG分类实验"
            }
            
            # 初始化wandb
            wandb.init(**wandb_config)
            
            # 定义全局步数指标
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
            
            # 定义训练指标
            for metric_config in config['monitoring']['metrics']['train']:
                wandb.define_metric(
                    f"train/{metric_config['name']}", 
                    summary=metric_config['summary']
                )
            
            # 定义验证指标
            for metric_config in config['monitoring']['metrics']['val']:
                wandb.define_metric(
                    f"val/{metric_config['name']}", 
                    summary=metric_config['summary']
                )
            
            print(f"\nWandB 实验名称: {run_name}")
            print(f"在以下地址查看训练过程：{wandb.run.url}")
        except Exception as e:
            print(f"\nWandB 初始化失败: {str(e)}")
            print("继续训练，但不会记录到WandB")
            config['training']['use_wandb'] = False
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {'GPU: ' + torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # 创建数据集和数据加载器
    train_dataset = EEGDataset(
        data_dir=config['data']['cache_dir'],
        window_size=config['data']['preprocessing']['window_size'],
        stride=config['data']['preprocessing']['stride'],
        split='train'
    )
    
    val_dataset = EEGDataset(
        data_dir=config['data']['cache_dir'],
        window_size=config['data']['preprocessing']['window_size'],
        stride=config['data']['preprocessing']['stride'],
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # 创建模型
    model = create_model(config)
    model = model.to(device)
    
    # 配置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 配置学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['scheduler']['T_max'],
        eta_min=config['training']['scheduler']['eta_min']
    )
    
    # 配置损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        exp_dir=run_dir
    )
    
    # 开始训练
    trainer.train()
    
    # 关闭wandb
    if config['training']['use_wandb']:
        wandb.finish()

if __name__ == '__main__':
    main() 