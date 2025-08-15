import os
import torch
from torch import nn
import numpy as np

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

# ---------- 模型工厂函数 ----------
def create_model(config):
    """创建模型实例"""
    return GRUOnlyModel(config)

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
    print("GRU模型测试")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    model = GRUOnlyModel(params.gru_config).to(device)
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
