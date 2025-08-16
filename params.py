# -*- coding: utf-8 -*-
"""
配置参数文件 - 量化分析系统
"""
from types import SimpleNamespace

# 基础配置
params = SimpleNamespace()

# 数据路径配置
# 新增CSV数据路径配置
params.feat_path = "./test_feat.csv"  # 特征数据CSV文件路径
params.label_csv_path = "./test_label.csv"  # 标签数据CSV文件路径

# 训练参数
params.batch_size = 64
params.epochs = 50
params.learning_rate = 1e-3
params.patience = 10
params.num_workers = 8
params.max_grad_norm = 1.0

# 输出配置
params.output_root = "./project_haoxin/output/gru_only_baseline"

# 数据维度配置
params.time_steps = 15
params.num_nodes = 12
params.num_samples = 8
params.flat_dim = params.num_nodes * params.num_samples  # 96

# 模型选择配置
params.model_type = "graph_transformer"  # "gru" 或 "graph_transformer"

# GRU模型配置
params.gru_config = {
    "input_dim": params.flat_dim,   # 96
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
    # 图编码器配置
    "graph_cfg": {
        "num_nodes": 12,
        "in_dim": 8,
        "d_model": 512,     # 图表示维度F（增大=更强表达）
        "n_heads": 8,
        "n_layers": 8,      # 8层图Transformer（≈16-18M图侧参数）
        "dropout": 0.10,
        "ff_mult": 4,
        "use_node_type_embed": True,
        "prior_matrix": None,     # 如有 12x12 相关性先验，可传入 torch.Tensor
        "prior_strength": 0.2,
    },
    # 时序模块配置
    "temporal_cfg": {
        "input_dim": 512,  # 自动接管，无需手填
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

# 大模型配置（可选）
params.graph_transformer_large_config = {
    # 更大容量（≈3千万级参数）
    "graph_cfg": {
        "num_nodes": 12,
        "in_dim": 8,
        "d_model": 640,     # 或 768
        "n_heads": 10,      # d_model 必须能整除 n_heads
        "n_layers": 10,
        "dropout": 0.10,
        "ff_mult": 4,
        "use_node_type_embed": True,
        "prior_matrix": None,
        "prior_strength": 0.2,
    },
    "temporal_cfg": {
        "input_dim": 640,
        "hidden_dim": 320,   # 适配更大F
        "num_layers": 2,
        "dropout": 0.05,
        "fc_dims": [160, 64],
        "use_batch_norm": False,
        "bidirectional": False,
        "pooling": "tail_mean",
        "tail_k": 8
    }
}

# 随机种子
params.random_seeds = [42, 123, 456, 789, 1000]

# 时间窗口配置
params.time_window = "day"

# 实验配置
params.n_experiments = 5

# 测试模式配置
params.test_mode = True  # 是否启用测试模式
