# EEG Deep Learning Project

这是一个使用深度学习处理脑电图(EEG)数据的项目。该项目采用模块化设计，使用PyTorch框架实现，并利用Hugging Face数据集进行训练和评估。

## 项目结构

```
src/
├── data/           # 数据加载和预处理模块
├── models/         # 深度学习模型定义
├── utils/          # 工具函数和辅助类
├── config/         # 配置文件
├── training/       # 训练相关代码
└── evaluation/     # 模型评估代码
```

## 功能特点

- 支持从Hugging Face加载EEG数据集
- 模块化的数据预处理流程
- 可配置的深度学习模型架构
- 完整的训练和评估流程
- 实验结果可视化
- 模型性能评估

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

## 安装

1. 克隆项目：
```bash
git clone [项目地址]
cd eeg-deep-learning
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 数据准备：
```python
python src/data/download_dataset.py
```

2. 训练模型：
```python
python src/training/train.py --config config/default.yaml
```

3. 评估模型：
```python
python src/evaluation/evaluate.py --model_path checkpoints/best_model.pth
```

## 配置说明

在 `config/` 目录下的YAML文件中可以配置：
- 数据集参数
- 模型架构参数
- 训练参数
- 评估参数

## 主要模块说明

### 数据模块 (src/data/)
- 负责数据的下载、加载和预处理
- 实现数据增强和转换
- 管理数据集的划分

### 模型模块 (src/models/)
- 定义深度学习模型架构
- 实现自定义层和损失函数
- 支持模型的保存和加载

### 训练模块 (src/training/)
- 实现训练循环
- 管理学习率调度
- 处理模型检查点
- 记录训练日志

### 评估模块 (src/evaluation/)
- 实现模型评估指标
- 生成评估报告
- 可视化模型性能

## 贡献指南

欢迎提交问题和改进建议。如需贡献代码，请：
1. Fork 项目
2. 创建新的分支
3. 提交更改
4. 发起 Pull Request

## 许可证

本项目采用 MIT 许可证 