# 量化分析系统

这是一个基于GRU的量化分析系统，用于处理时序数据并进行回归预测。

## 系统概述

本系统专门用于量化分析，实现了：

- **数据处理**: 加载和处理量化特征数据
- **模型架构**: 基于GRU的时序回归模型
- **训练框架**: 支持多GPU训练和分布式训练
- **评估指标**: RMSE、R2、Pearson相关系数等回归指标

## 文件结构

```
├── params.py              # 配置文件
├── dataset.py             # 数据集加载和处理
├── model.py               # GRU模型定义
├── learner.py             # 训练逻辑
├── metrics.py             # 评估指标
├── train.py               # 训练入口
├── inference.py           # 推理脚本
├── test_quantitative.py   # 系统测试脚本
├── train.sh               # 多GPU训练脚本
├── run_single.sh          # 单GPU训练脚本
└── infer.sh               # 推理脚本
```

## 配置说明

### params.py 主要配置

```python
# 数据路径配置
params.feat_dir = "/path/to/feature/data"
params.label_path = "/path/to/label/data"

# 训练参数
params.batch_size = 2048
params.epochs = 50
params.learning_rate = 1e-3

# 数据维度配置
params.time_steps = 15      # 时间步数
params.num_nodes = 12       # 节点数
params.num_samples = 8      # 样本数
params.flat_dim = 96        # 展平后的特征维度

# 模型配置
params.gru_config = {
    "input_dim": 96,        # 输入维度
    "hidden_dim": 256,      # 隐藏层维度
    "num_layers": 2,        # GRU层数
    "dropout": 0.05,        # Dropout率
    "fc_dims": [64],        # 全连接层维度
    "pooling": "tail_mean", # 池化方式
    "tail_k": 8,           # 尾部平均的k值
}
```

## 使用方法

### 1. 环境准备

确保安装了所需的依赖包：

```bash
pip install torch torchvision
pip install numpy pandas scikit-learn scipy
pip install matplotlib tqdm
```

### 2. 数据准备

确保数据路径配置正确：

- 特征数据路径：`params.feat_dir`
- 标签数据路径：`params.label_path`

数据格式：
- 特征数据：pickle格式，包含时序特征
- 标签数据：pickle格式，包含回归标签

### 3. 系统测试

运行测试脚本验证系统是否正常工作：

```bash
python test_quantitative.py
```

### 4. 训练模型

#### 单GPU训练

```bash
bash run_single.sh
```

#### 多GPU训练

```bash
bash train.sh
```

#### 自定义训练

```bash
python train.py ./output_dir ./data/dummy.txt ./data/dummy
```

### 5. 模型推理

```bash
bash infer.sh path/to/model_checkpoint.pt
```

或者直接使用Python脚本：

```bash
python inference.py \
    --model_path path/to/model_checkpoint.pt \
    --batch_size 512 \
    --output_file results.csv
```

## 模型架构

### GRUOnlyModel

主要组件：
- **TemporalModule**: GRU时序模块 + 全连接层
- **线性跳跃连接**: 直接连接输入和输出的线性层
- **池化策略**: 支持mean、max、last、tail_mean等

输入格式：`[batch_size, time_steps, num_nodes, num_samples]`
输出格式：`[batch_size, 1]`

### 数据处理流程

1. **数据加载**: 从pickle文件加载特征和标签
2. **数据分割**: 按时间分割为训练/验证/测试集
3. **数据预处理**: 特征展平和标准化
4. **批次处理**: 创建DataLoader进行批次训练

## 评估指标

系统使用以下回归指标：

- **RMSE**: 均方根误差
- **R²**: 决定系数
- **Pearson**: 皮尔逊相关系数
- **Accuracy**: 基于相对误差的准确率（10%阈值）

## 训练监控

训练过程中会记录：

- 训练损失
- 验证指标
- 学习曲线
- 模型检查点

可以通过TensorBoard查看训练过程：

```bash
tensorboard --logdir ./exports_quantitative
```

## 注意事项

1. **数据路径**: 确保在`params.py`中正确配置数据路径
2. **GPU内存**: 根据GPU内存调整batch_size
3. **数据格式**: 确保数据格式符合预期
4. **依赖版本**: 确保PyTorch版本兼容

## 故障排除

### 常见问题

1. **数据加载失败**
   - 检查数据路径是否正确
   - 确认数据文件格式

2. **内存不足**
   - 减小batch_size
   - 减少num_workers

3. **模型训练不收敛**
   - 调整学习率
   - 检查数据预处理
   - 调整模型配置

### 调试建议

1. 先运行测试脚本验证系统
2. 使用小数据集进行快速测试
3. 检查日志输出和错误信息
4. 验证数据格式和维度

## 扩展功能

可以根据需要扩展以下功能：

- 添加更多模型架构
- 实现数据增强
- 添加更多评估指标
- 支持更多数据格式
- 实现模型集成

## 联系信息

如有问题，请检查：
1. 数据路径配置
2. 依赖包版本
3. 系统日志输出
4. 测试脚本结果
