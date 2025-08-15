# 量化分析系统

基于深度学习的量化分析系统，支持多种模型架构进行时序回归预测。

## 系统概述

本系统专门用于量化分析，支持两种主要的模型架构：

1. **GRU模型** - 轻量级时序模型，适合快速实验
2. **GraphTransformer模型** - 高级图神经网络模型，具有更强的表达能力

### 主要特性

- 🚀 **多模型支持**: 支持GRU和GraphTransformer两种模型架构
- 📊 **回归预测**: 专门用于量化回归任务
- 🎯 **图结构建模**: GraphTransformer模型能够建模指标间的复杂关系
- ⚡ **高效训练**: 支持多GPU训练和分布式训练
- 🧪 **完整测试**: 每个组件都有独立的测试功能
- 📈 **丰富指标**: 支持RMSE、R²、Pearson相关系数等多种评估指标

## 模型架构

### GRU模型
- **输入**: `[B, 15, 12, 8]` - 批次大小 × 时间步 × 节点数 × 特征维度
- **处理**: 将每个时间步的12×8特征展平为96维，送入GRU
- **输出**: `[B, 1]` - 回归预测值
- **参数量**: 约1-2M参数

### GraphTransformer模型
- **图编码器**: 使用多头自注意力机制建模12个指标节点间的关系
- **时序模块**: GRU处理图表示的时间序列
- **可学习边偏置**: 自动学习指标间的相关性
- **CLS汇聚**: 使用CLS token汇聚图信息
- **参数量**: 约20-30M参数（可配置）

## 文件结构

```
├── params.py              # 配置文件（包含模型选择）
├── dataset.py             # 数据集加载和处理
├── model.py               # 模型定义（GRU + GraphTransformer）
├── learner.py             # 训练逻辑
├── metrics.py             # 评估指标
├── train.py               # 训练入口
├── inference.py           # 推理脚本
├── test_quantitative.py   # 综合测试脚本
├── test_graph_transformer.py  # GraphTransformer专用测试
├── run_tests.py           # 测试运行器
├── train.sh               # 多GPU训练脚本
├── run_single.sh          # 单GPU训练脚本
├── infer.sh               # 推理脚本
├── README.md              # 项目说明文档
├── requirements.txt       # 依赖文件
└── CHANGELOG.md           # 变更日志
```

## 配置说明

### 模型选择配置

在 `params.py` 中设置模型类型：

```python
# 模型选择配置
params.model_type = "graph_transformer"  # "gru" 或 "graph_transformer"

# GRU模型配置
params.gru_config = {
    "input_dim": 96,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.05,
    "fc_dims": [64],
    "use_batch_norm": False,
    "bidirectional": False,
    "pooling": "tail_mean",
    "tail_k": 8,
}

# GraphTransformer模型配置
params.graph_transformer_config = {
    "graph_cfg": {
        "num_nodes": 12,
        "in_dim": 8,
        "d_model": 512,     # 图表示维度
        "n_heads": 8,
        "n_layers": 8,      # 8层图Transformer
        "dropout": 0.10,
        "ff_mult": 4,
        "use_node_type_embed": True,
        "prior_matrix": None,     # 可传入相关性先验
        "prior_strength": 0.2,
    },
    "temporal_cfg": {
        "input_dim": 512,  # 自动接管
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.05,
        "fc_dims": [128, 64],
        "use_batch_norm": False,
        "bidirectional": False,
        "pooling": "tail_mean",
        "tail_k": 8
    }
}
```

### 大模型配置

对于需要更强表达能力的场景，可以使用大模型配置：

```python
params.graph_transformer_config = params.graph_transformer_large_config
```

大模型配置包含：
- 更大的隐藏维度（640）
- 更多的注意力头（10）
- 更多的层数（10层）
- 约30M参数

## 使用方法

### 快速测试

```bash
# 运行所有测试
python run_tests.py

# 运行单个组件测试
python dataset.py
python model.py
python learner.py
python test_quantitative.py

# 专门测试GraphTransformer模型
python test_graph_transformer.py
```

### 训练模型

```bash
# 单GPU训练
bash run_single.sh

# 多GPU训练
bash train.sh
```

### 模型推理

```bash
bash infer.sh path/to/model_checkpoint.pt
```

## 模型切换

### 使用GRU模型

```python
# 在params.py中设置
params.model_type = "gru"
```

### 使用GraphTransformer模型

```python
# 在params.py中设置
params.model_type = "graph_transformer"
```

### 使用大模型

```python
# 在params.py中设置
params.model_type = "graph_transformer"
params.graph_transformer_config = params.graph_transformer_large_config
```

## 技术细节

### GraphTransformer架构

1. **图编码器**:
   - 节点特征投影: `[B*T, 12, 8]` → `[B*T, 12, D]`
   - 节点类型嵌入: 为每个指标节点添加可学习的嵌入
   - 可学习边偏置: 每个注意力头一个12×12的偏置矩阵
   - CLS token: 用于汇聚图信息
   - 多层Transformer: 8层自注意力机制

2. **时序模块**:
   - GRU处理: `[B, 15, D]` → `[B, H]`
   - 池化策略: tail_mean（取最后k天平均）
   - Skip连接: 原始特征的线性组合
   - 回归头: 多层感知机输出预测值

### 训练策略

- **损失函数**: SmoothL1Loss (Huber Loss)
- **优化器**: AdamW with weight decay
- **学习率调度**: OneCycleLR
- **混合精度**: 自动混合精度训练
- **梯度裁剪**: 防止梯度爆炸

### 评估指标

- **RMSE**: 均方根误差
- **R²**: 决定系数
- **Pearson**: 皮尔逊相关系数
- **相对误差**: 自定义准确率指标

## 性能对比

| 模型 | 参数量 | 训练速度 | 表达能力 | 适用场景 |
|------|--------|----------|----------|----------|
| GRU | ~1-2M | 快 | 中等 | 快速实验、资源受限 |
| GraphTransformer | ~20-30M | 中等 | 强 | 生产环境、高精度需求 |
| GraphTransformer-Large | ~30M+ | 慢 | 最强 | 研究、最高精度需求 |

## 故障排除

### 常见问题

1. **内存不足**:
   - 减小batch_size
   - 使用GRU模型替代GraphTransformer
   - 启用梯度累积

2. **训练速度慢**:
   - 使用多GPU训练
   - 启用混合精度训练
   - 调整数据加载器worker数量

3. **模型不收敛**:
   - 检查学习率设置
   - 调整dropout率
   - 检查数据预处理

### 调试技巧

1. **启用测试模式**:
   ```python
   params.test_mode = True
   ```

2. **检查模型参数**:
   ```python
   python model.py
   ```

3. **验证数据加载**:
   ```python
   python dataset.py
   ```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证。

## 更新日志

详见 [CHANGELOG.md](CHANGELOG.md)
