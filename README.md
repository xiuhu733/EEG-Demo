# EEG Deep Learning Project

这是一个使用深度学习处理脑电图(EEG)数据的项目。该项目采用模块化设计，使用PyTorch框架实现，并使用MNE库提供的高质量EEG数据集。

## 项目特点

- 支持多种深度学习模型架构（EEGNet、TitansEEG）
- 完整的数据预处理和增强流程
- 空间滤波和注意力机制
- 自适应学习率调度
- 实验跟踪和可视化
- 模型性能评估和分析

## 项目结构

```
src/
├── data/               # 数据处理模块
│   ├── dataset.py      # 数据集实现
│   ├── augmentation.py # 数据增强
│   ├── spatial_filter.py # 空间滤波
│   └── download_dataset.py # 数据下载
├── models/             # 模型定义
│   ├── eegnet.py      # EEGNet模型
│   ├── titans_model.py # TitansEEG模型
│   └── model_factory.py # 模型工厂
├── training/           # 训练相关
│   └── train.py       # 训练脚本
├── evaluation/         # 评估模块
│   └── evaluator.py   # 评估器
├── utils/             # 工具函数
│   ├── trainer.py     # 训练器
│   └── utils.py       # 通用工具
└── config/            # 配置文件
    ├── default.yaml   # EEGNet配置
    └── titans.yaml    # TitansEEG配置
```

## 模型架构

### EEGNet
- 输入层：处理64通道EEG数据
- 时间卷积层：捕获时间特征
- 深度卷积层：学习空间特征
- 可分离卷积层：高效特征提取
- 分类层：4分类输出

配置参数：
```yaml
model:
  name: "eegnet"
  input_channels: 64
  dropout_rate: 0.4
  kernel_length: 64
  num_filters: 12
  pool_size: 2
  num_classes: 4
```

### TitansEEG
- 空间特征提取层：增强空间信息处理
- 投影层：高维特征变换
- 神经记忆模块：使用基础版本的Neural Memory进行长期依赖建模
  - 基于[titans-pytorch](https://github.com/lucidrains/titans-pytorch)的NeuralMemory实现
  - 不使用MAC (Memory as Context)配置
  - 适用于EEG信号的实时处理
- 分类器：多层非线性变换

配置参数：
```yaml
model:
  name: "titanseeg"
  input_channels: 64
  hidden_dim: 128    # 神经记忆模块的隐藏维度
  chunk_size: 16     # 记忆块大小，较小的值适合较短序列
  dropout_rate: 0.2
  num_classes: 4
```

注意：我们选择使用基础版本的Neural Memory而不是完整的MAC配置，原因是：
1. 基础版本计算开销更小，更适合实时EEG处理
2. EEG序列相对较短，不需要复杂的长期记忆机制
3. 在有限的GPU内存下可以获得更好的性能平衡

## 数据处理

### 预处理
- 带通滤波：4-40Hz (Titans) / 8-30Hz (EEGNet)
- 时间窗口：320/800样本
- 步长：160/400样本
- 标准化：均值为0，标准差为1

### 数据增强
- 高斯噪声：scale=0.1，p=0.5
- 信号翻转：p=0.3
- 时间掩码：ratio=0.1，p=0.3

### 空间滤波
- CSP滤波器：16个组件
- 空间注意力机制

## 训练策略

### 优化器设置
- 优化器：AdamW
- 学习率：0.001
- 权重衰减：0.0001
- 批量大小：16 (Titans) / 64 (EEGNet)

### 学习率调度
- 余弦退火调度
- 最小学习率：0.00001
- 调度周期：300轮

### 早停策略
- 耐心值：50轮
- 最小改进：0.0005
- 监控指标：验证准确率

## 性能监控

### 训练指标
- 损失值
- 准确率
- F1分数
- 精确率
- 召回率
- Kappa系数

### 可视化
- 损失曲线
- 准确率曲线
- 混淆矩阵
- ROC曲线
- PR曲线
- 学习率变化

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 下载数据集：
```bash
python src/data/download_dataset.py --dataset eegbci
```

3. 训练模型：
```bash
# 训练EEGNet
python src/compare_models.py --model eegnet

# 训练TitansEEG
python src/compare_models.py --model titans
```

4. 查看训练过程：
```bash
tensorboard --logdir=runs/模型名称_时间戳/tensorboard
```

## 性能优化建议

1. 内存优化：
   - 调整批量大小
   - 减小时间窗口
   - 降低模型维度

2. 训练优化：
   - 调整学习率
   - 修改早停参数
   - 增加数据增强

3. 模型优化：
   - 调整dropout率
   - 修改网络架构
   - 添加正则化

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (推荐)
- 8GB+ GPU内存

## 主要依赖

```
torch
numpy
pandas
mne
scikit-learn
moabb
matplotlib
seaborn
tensorboard
wandb
tqdm
pyyaml
titans-pytorch
pyriemann
```

## 许可证

本项目采用 MIT 许可证
