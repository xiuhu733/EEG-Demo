# EEG Deep Learning Project

这是一个使用深度学习处理脑电图(EEG)数据的项目。该项目采用模块化设计，使用PyTorch框架实现，并使用MNE库提供的高质量EEG数据集。

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

- 支持多种EEG数据集（MNE示例数据集、EEGBCI运动想象数据集）
- 完整的数据预处理流程（滤波、标准化等）
- 模块化的深度学习模型架构
- 完整的训练和评估流程
- 实验结果可视化
- 模型性能评估

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- MNE 1.0+
- CUDA (可选，用于GPU加速)

## 安装

1. 克隆项目：
```bash
git clone https://github.com/xiuhu733/EEG-Demo.git
cd EEG-Demo
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

## 数据集

项目支持两种主要的EEG数据集：

1. MNE示例数据集（sample）
   - 包含视听实验的EEG/MEG数据
   - 高质量的预处理数据
   - 适合入门学习和测试

2. EEGBCI运动想象数据集（eegbci）
   - 包含运动想象任务的EEG数据
   - 多个受试者的数据
   - 4种不同的运动想象任务类别

### 下载数据集

1. 下载MNE示例数据集：
```bash
python src/data/download_dataset.py --dataset sample
```

2. 下载EEGBCI运动想象数据集：
```bash
python src/data/download_dataset.py --dataset eegbci
```

数据将被保存在 `data` 目录下，包含以下文件：
- `eeg_data.npy`: EEG数据
- `labels.npy`: 标签数据（对于eegbci数据集）
- `times.npy`: 时间点信息（对于sample数据集）
- `channels.txt`: 通道信息

## 配置说明

在 `src/config/default.yaml` 中配置：
- 数据集参数（类型、缓存目录、批大小等）
- 模型架构参数（通道数、层数等）
- 训练参数（学习率、优化器等）
- 评估参数（评估指标、可视化选项等）

## 使用方法

1. 下载数据集：
```bash
python src/data/download_dataset.py --dataset eegbci
```

2. 训练模型：
```bash
python src/training/train.py --config src/config/default.yaml
```

3. 评估模型：
```bash
python src/evaluation/evaluate.py --model_path checkpoints/best_model.pth
```

## 主要模块说明

### 数据模块 (src/data/)
- 支持多种数据集格式
- 实现数据预处理和增强
- 提供数据集划分功能

### 模型模块 (src/models/)
- 实现EEGNet模型架构
- 支持模型配置和扩展
- 包含预测和概率输出功能

### 训练模块 (src/training/)
- 实现训练循环
- 支持学习率调度
- 包含早停机制
- 模型检查点保存

### 评估模块 (src/evaluation/)
- 多种评估指标
- 混淆矩阵可视化
- ROC曲线分析
- 学习曲线绘制

## 贡献指南

欢迎提交问题和改进建议。如需贡献代码，请：
1. Fork 项目
2. 创建新的分支
3. 提交更改
4. 发起 Pull Request

## 许可证

本项目采用 MIT 许可证 