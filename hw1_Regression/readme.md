# COVID-19 病例预测模型

基于深度学习的 COVID-19 病例数预测项目。

## 目录结构

├── data/                   # 数据相关文件
│ ├── raw/ # 原始数据
│ │ ├── covid.train.csv
│ │ └── covid.test.csv
│ ├── dataset.py # 数据集类定义
│ └── preprocess.py # 数据预处理函数
├── models/ # 模型相关
│ └── model.py # 模型定义
├── options/ # 配置文件
│ ├── options.py # 配置参数
│ └── util.py # 工具函数
├── utils/ # 工具函数
│ └── logger.py # 日志工具
├── train.py # 训练脚本
├── test.py # 测试脚本
└── requirements.txt # 项目依赖

## 快速开始

### 1. 环境配置
- `pip install -r requirements.txt`

### 2. 数据准备

- 将训练数据和测试数据放入 `data/raw/` 目录

### 3. 训练模型

- `python train.py`  

### 4. 预测 

- `python test.py` 

## 可视化训练过程

- `tensorboard --logdir=runs/`  

## 注意事项

1. 确保已安装 CUDA（使用 GPU）
2. 训练日志保存在 `train.log`
3. 预测结果保存在 `predictions/pred.csv`
4. 最佳模型保存在 `models/model.ckpt`

