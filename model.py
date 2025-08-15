import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ---------- GRU模型定义 ----------
class TemporalModule(nn.Module):
    """时序模块：GRU + 全连接层"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=config["input_dim"],
            hidden_size=config["hidden_dim"],
            num_layers=config["num_layers"],
            batch_first=True,
            dropout=config["dropout"] if config["num_layers"] > 1 else 0,
            bidirectional=config["bidirectional"]
        )
        self.gru_output_dim = config["hidden_dim"] * (2 if config["bidirectional"] else 1)
        self.batch_norm = nn.BatchNorm1d(self.gru_output_dim) if config["use_batch_norm"] else None

        fc_layers, prev = [], self.gru_output_dim
        for d in config["fc_dims"]:
            fc_layers += [nn.Linear(prev, d), nn.ReLU(), nn.Dropout(config["dropout"])]
            prev = d
        fc_layers.append(nn.Linear(prev, 1))
        self.fc_block = nn.Sequential(*fc_layers)

        self.linear_skip = nn.Sequential(
            nn.Linear(config["input_dim"], 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, x):  # x: [B, T, C]
        gru_out, _ = self.gru(x)
        p = self.config["pooling"]
        if p == "mean":
            fused = gru_out.mean(1)
        elif p == "max":
            fused = gru_out.max(1)[0]
        elif p == "last":
            fused = gru_out[:, -1, :]
        elif p == "tail_mean":
            fused = gru_out[:, -self.config["tail_k"]:, :].mean(1)
        else:
            raise ValueError(f"不支持的池化方式: {p}")
        if self.batch_norm is not None: 
            fused = self.batch_norm(fused)
        main_out = self.fc_block(fused)
        if p == "tail_mean":
            k = self.config.get("tail_k", 16)
            skip_out = self.linear_skip(x[:, -k:, :].mean(1))
        else:
            skip_out = self.linear_skip(x[:, -1, :])
        return main_out + skip_out

class GRUOnlyModel(nn.Module):
    """GRU模型：把每个时间步的 12×8 展平为 96 维，直接送入 GRU"""
    def __init__(self, config):
        super().__init__()
        self.temporal = TemporalModule(config)

    def forward(self, x):  # x: [B, 15, 12, 8]
        B, T, N, F = x.shape
        x = x.view(B, T, N * F).contiguous()   # [B, T, 96]
        return self.temporal(x)

# ---------- GraphTransformer模型定义 ----------
class GraphTransformerLayer(nn.Module):
    """
    Pre-LN Transformer Encoder Layer with additive attention bias.
    x: [B, L, D]  (这里 L=节点数或节点数+CLS；D=隐藏维度)
    attn_bias: 
        - 若提供 [H, L, L] 或 [1, H, L, L]，会在forward中广播到 [B, H, L, L]
        - 这是加法bias（加到注意力score上），不是mask（非 -inf）
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, L, D]
        attn_bias:
            - None
            - [H, L, L] or [1, H, L, L] or [B, H, L, L]
        """
        B, L, D = x.shape
        h = self.n_heads
        d = self.d_head

        # --- Self-Attention ---
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm)  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, h, d).transpose(1, 2)  # [B, H, L, d]
        k = k.view(B, L, h, d).transpose(1, 2)  # [B, H, L, d]
        v = v.view(B, L, h, d).transpose(1, 2)  # [B, H, L, d]

        # scaled dot-product attention
        # PyTorch 2.x 的 F.scaled_dot_product_attention 支持 float 型 additive attn_mask
        # mask/偏置形状: [B, H, L, S]
        if attn_bias is not None:
            if attn_bias.dim() == 3:  # [H, L, L]
                attn_bias = attn_bias.unsqueeze(0)  # [1, H, L, L]
            # 广播到 batch
            if attn_bias.size(0) == 1 and attn_bias.size(1) == h and attn_bias.size(2) == L and attn_bias.size(3) == L:
                attn_bias = attn_bias.expand(B, -1, -1, -1).contiguous()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=self.attn_drop.p if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        out = self.proj_drop(self.proj(out))
        x = x + out

        # --- FFN ---
        x2 = self.ff(self.norm2(x))
        x = x + x2
        return x

class GraphEncoder(nn.Module):
    """
    把每天的 "12节点×8维" 小图编码为一个图向量 F=D（使用CLS汇聚）。

    输入:
      node_feats: [B*T, N=12, C=8]
    输出:
      graph_repr: [B*T, D]
    """
    def __init__(
        self,
        num_nodes: int = 12,
        in_dim: int = 8,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
        use_node_type_embed: bool = True,
        prior_matrix: Optional[torch.Tensor] = None,  # [N, N], 可传入相关性/先验
        prior_strength: float = 0.2,                  # 先验强度（会作为可学习缩放的初值）
    ):
        super().__init__()
        self.N = num_nodes
        self.D = d_model
        self.use_node_type_embed = use_node_type_embed

        # 节点初始编码： (节点类型嵌入) + (特征MLP投影)
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        if use_node_type_embed:
            self.node_type_embed = nn.Embedding(num_nodes, d_model)
        else:
            self.register_parameter("node_type_embed", None)

        # 可学习的注意力偏置（每个头一个 N×N）
        self.edge_bias = nn.Parameter(torch.zeros(n_heads, num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.edge_bias)

        # 先验（如指标相关性矩阵）作为buffer
        if prior_matrix is not None:
            assert prior_matrix.shape == (num_nodes, num_nodes)
            self.register_buffer("prior", prior_matrix.float())
            self.prior_alpha = nn.Parameter(torch.tensor(prior_strength, dtype=torch.float32))
        else:
            self.register_buffer("prior", None)
            self.prior_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # 置0等于不用先验

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 多层图Transformer
        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_model=d_model, n_heads=n_heads, dropout=dropout, ff_mult=ff_mult)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

    def _build_attn_bias(self, L: int) -> torch.Tensor:
        """
        构建 [H, L, L] 的注意力加性bias，包含：
          - 可学习边偏置 (扩展到包含CLS的大小)
          - 可选的先验偏置
        """
        H = self.edge_bias.size(0)
        N = self.N
        assert L in (N, N + 1)  # N节点 or N+CLS

        # 基础：learnable edge bias
        if L == N:
            bias = self.edge_bias  # [H, N, N]
        else:
            # pad到 [H, N+1, N+1]，CLS行列置零（不引入偏置）
            bias = F.pad(self.edge_bias, (0, 1, 0, 1))  # [H, N+1, N+1]

        # 加上先验
        if self.prior is not None and self.prior_alpha is not None:
            if L == N:
                prior = self.prior  # [N, N]
            else:
                prior = F.pad(self.prior, (0, 1, 0, 1))  # [N+1, N+1]
            # 扩到每个头
            prior = prior.unsqueeze(0).expand(H, L, L)
            bias = bias + self.prior_alpha * prior

        return bias  # [H, L, L]

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        """
        node_feats: [B*T, N, C]
        return:     [B*T, D]
        """
        BT, N, C = node_feats.shape
        assert N == self.N

        # 节点特征投影
        x = self.node_encoder(node_feats)  # [BT, N, D]
        if self.use_node_type_embed:
            node_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(BT, N)  # [BT, N]
            x = x + self.node_type_embed(node_ids)  # [BT, N, D]

        # CLS拼接
        cls = self.cls_token.expand(BT, -1, -1)     # [BT, 1, D]
        x = torch.cat([cls, x], dim=1)              # [BT, N+1, D]
        L = N + 1

        # 构造注意力bias并广播到batch
        attn_bias = self._build_attn_bias(L)  # [H, L, L]

        # 多层Transformer
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias.unsqueeze(0))  # [1,H,L,L]在内部会广播到B

        x = self.final_ln(x)  # [BT, L, D]
        graph_repr = x[:, 0, :]  # 取CLS
        return graph_repr  # [BT, D]

class TemporalGRUHead(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=config["input_dim"],
            hidden_size=config["hidden_dim"],
            num_layers=config["num_layers"],
            batch_first=True,
            dropout=config["dropout"] if config["num_layers"] > 1 else 0.0,
            bidirectional=config["bidirectional"],
        )
        self.gru_out_dim = config["hidden_dim"] * (2 if config["bidirectional"] else 1)
        self.bn = nn.BatchNorm1d(self.gru_out_dim) if config.get("use_batch_norm", False) else None

        # FC head
        dims = [self.gru_out_dim] + list(config["fc_dims"]) + [1]
        fcs = []
        for i in range(len(dims) - 2):
            fcs += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(config["dropout"])]
        fcs += [nn.Linear(dims[-2], dims[-1])]
        self.fc = nn.Sequential(*fcs)

        # 原始特征的skip支路（取最后k天的原始"12*8=96维/天"聚合）
        self.tail_k = config.get("tail_k", 8)
        self.linear_skip = nn.Sequential(
            nn.Linear(96, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def forward(self, seq_feat: torch.Tensor, raw_last_k_flat: torch.Tensor) -> torch.Tensor:
        """
        seq_feat:        [B, T, F]    # 图表示序列
        raw_last_k_flat: [B, 96]      # 最近k天原始特征（每天展平96后再平均）
        """
        out, _ = self.gru(seq_feat)   # [B, T, H*dir]
        p = self.config["pooling"]
        if p == "mean":
            fused = out.mean(1)
        elif p == "max":
            fused = out.max(1)[0]
        elif p == "last":
            fused = out[:, -1, :]
        elif p == "tail_mean":
            k = min(self.tail_k, out.size(1))
            fused = out[:, -k:, :].mean(1)
        else:
            raise ValueError(f"不支持的池化方式: {p}")

        if self.bn is not None:
            fused = self.bn(fused)

        main_out = self.fc(fused)             # [B, 1]
        skip_out = self.linear_skip(raw_last_k_flat)  # [B, 1]
        return main_out + skip_out            # [B, 1]

class StockGraphTemporalModel(nn.Module):
    def __init__(self,
                 graph_cfg: dict,
                 temporal_cfg: dict):
        super().__init__()
        self.N = graph_cfg.get("num_nodes", 12)
        self.C = graph_cfg.get("in_dim", 8)
        self.T = 15  # 你的设定（若变长可改为动态、不强制）

        self.graph_encoder = GraphEncoder(**graph_cfg)

        # 将图编码输出尺寸接到GRU
        temporal_cfg = dict(temporal_cfg)
        temporal_cfg["input_dim"] = graph_cfg["d_model"]
        self.temporal_head = TemporalGRUHead(temporal_cfg)

    @staticmethod
    def _flatten_last_k_days(x_raw: torch.Tensor, k: int) -> torch.Tensor:
        """
        x_raw: [B, T, N, C]
        返回:  [B, 96]  # 最近k天，每天展平为96维后求平均
        """
        B, T, N, C = x_raw.shape
        k = min(k, T)
        xk = x_raw[:, -k:, :, :]                     # [B, k, N, C]
        xk = xk.reshape(B, k, N * C)                 # [B, k, 96]
        xk = xk.mean(dim=1)                          # [B, 96]
        return xk

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw: [B, 15, 12, 8]
        """
        B, T, N, C = x_raw.shape
        assert N == self.N and C == self.C

        # (1) 图编码：把每一天的12x8图 -> D维图向量
        xt = x_raw.reshape(B * T, N, C)              # [B*T, 12, 8]
        gt = self.graph_encoder(xt)                  # [B*T, D]
        gt = gt.view(B, T, -1)                       # [B, 15, D]

        # (2) 原始特征skip：最近k天flatten聚合为[96]
        tail_k = self.temporal_head.tail_k
        raw_last_k = self._flatten_last_k_days(x_raw, k=tail_k)  # [B, 96]

        # (3) GRU时序 + 回归
        y_hat = self.temporal_head(gt, raw_last_k)  # [B, 1]
        return y_hat

# ---------- 模型工厂函数 ----------
def create_model(config):
    """创建模型实例"""
    model_type = config.get("model_type", "gru")
    
    if model_type == "gru":
        return GRUOnlyModel(config["gru_config"])
    elif model_type == "graph_transformer":
        graph_cfg = config["graph_transformer_config"]["graph_cfg"]
        temporal_cfg = config["graph_transformer_config"]["temporal_cfg"]
        return StockGraphTemporalModel(graph_cfg, temporal_cfg)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

# ---------- 模型测试函数 ----------
def test_model_forward_pass(model, device):
    """测试模型前向传播"""
    print("测试模型前向传播...")
    
    # 创建不同大小的测试输入
    test_batch_sizes = [1, 4, 16]
    
    for batch_size in test_batch_sizes:
        # 创建测试输入
        dummy_input = torch.randn(batch_size, 15, 12, 8).to(device)
        
        # 前向传播
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  批次大小 {batch_size}: 输入 {dummy_input.shape} -> 输出 {output.shape}")
        
        # 检查输出是否合理
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"  警告: 批次大小 {batch_size} 的输出包含 NaN 或 Inf")
        else:
            print(f"  输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")

def test_model_parameters(model):
    """测试模型参数"""
    print("测试模型参数...")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 检查参数是否合理
    if total_params > 0:
        print("  ✓ 模型参数正常")
    else:
        print("  ✗ 模型参数异常")

def test_model_gradients(model, device):
    """测试模型梯度"""
    print("测试模型梯度...")
    
    # 创建测试输入和目标
    dummy_input = torch.randn(4, 15, 12, 8).to(device)
    dummy_target = torch.randn(4, 1).to(device)
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 前向传播
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    print(f"  损失值: {loss.item():.4f}")
    print(f"  梯度范数: {grad_norm:.4f}")
    
    if grad_norm > 0:
        print("  ✓ 梯度计算正常")
    else:
        print("  ✗ 梯度计算异常")

# ---------- 主函数 ----------
if __name__ == "__main__":
    from params import params
    
    print("=" * 50)
    print("模型测试")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"模型类型: {params.model_type}")

    # 创建模型
    model = create_model(params).to(device)
    print(f"✓ 模型创建成功")
    
    # 测试模型参数
    test_model_parameters(model)
    
    # 测试前向传播
    test_model_forward_pass(model, device)
    
    # 测试梯度计算
    test_model_gradients(model, device)
    
    print("\n" + "=" * 50)
    print("所有测试完成！")
    print("=" * 50)
