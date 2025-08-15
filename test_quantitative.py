#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试量化模型和数据集的脚本
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# 添加当前目录到路径
sys.path.append('.')

from params import params
from dataset import load_data, split_time_series_data, StockDataset
from model import create_model
from torch.utils.data import DataLoader

def test_data_loading():
    """测试数据加载"""
    print("=== 测试数据加载 ===")
    try:
        # 启用测试模式
        params.test_mode = True
        
        X, y, time_stamps = load_data()
        print(f"✓ 数据加载成功")
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
        print(f"  - 时间戳数量: {len(time_stamps)}")
        print(f"  - 时间范围: {time_stamps[0]} 至 {time_stamps[-1]}")
        
        # 测试数据分割
        (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
        print(f"✓ 数据分割成功")
        print(f"  - 训练集: {X_train.shape}, {y_train.shape}")
        print(f"  - 验证集: {X_val.shape}, {y_val.shape}")
        print(f"  - 测试集: {X_test.shape}, {y_test.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False

def test_dataset():
    """测试数据集"""
    print("\n=== 测试数据集 ===")
    try:
        # 启用测试模式
        params.test_mode = True
        
        X, y, time_stamps = load_data()
        (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
        
        # 创建数据集
        train_dataset = StockDataset(X_train[:100], y_train[:100])  # 只取前100个样本测试
        print(f"✓ 数据集创建成功")
        print(f"  - 数据集大小: {len(train_dataset)}")
        
        # 测试数据获取
        X_sample, y_sample = train_dataset[0]
        print(f"✓ 数据获取成功")
        print(f"  - X sample shape: {X_sample.shape}")
        print(f"  - y sample shape: {y_sample.shape}")
        print(f"  - X sample dtype: {X_sample.dtype}")
        print(f"  - y sample dtype: {y_sample.dtype}")
        
        # 测试数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        print(f"✓ 数据加载器创建成功")
        print(f"  - 批次数量: {len(train_loader)}")
        
        # 测试一个批次
        for X_batch, y_batch in train_loader:
            print(f"✓ 批次数据获取成功")
            print(f"  - X batch shape: {X_batch.shape}")
            print(f"  - y batch shape: {y_batch.shape}")
            break
        
        return True
    except Exception as e:
        print(f"✗ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model():
    """测试模型"""
    print("\n=== 测试模型 ===")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 创建模型
        model = create_model(params).to(device)
        print(f"✓ 模型创建成功")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  - 可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        # 测试前向传播
        dummy_input = torch.randn(2, 15, 12, 8).to(device)  # [B, T, N, F]
        print(f"✓ 模型前向传播测试")
        print(f"  - 输入形状: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"  - 输出形状: {output.shape}")
            print(f"  - 输出类型: {output.dtype}")
            print(f"  - 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """测试训练步骤"""
    print("\n=== 测试训练步骤 ===")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型和优化器
        model = create_model(params).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=1e-4)
        criterion = torch.nn.SmoothL1Loss(beta=0.5)
        
        # 创建小数据集
        params.test_mode = True
        X, y, time_stamps = load_data()
        (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
        train_dataset = StockDataset(X_train[:100], y_train[:100])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        print(f"✓ 训练组件创建成功")
        
        # 测试一个训练步骤
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            print(f"✓ 训练步骤成功")
            print(f"  - 损失值: {loss.item():.4f}")
            print(f"  - 输出形状: {output.shape}")
            print(f"  - 目标形状: {y_batch.shape}")
            break
        
        return True
    except Exception as e:
        print(f"✗ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_learner():
    """测试Learner"""
    print("\n=== 测试Learner ===")
    try:
        from learner import Learner
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型和优化器
        model = create_model(params).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=1e-4)
        
        # 创建小数据集
        params.test_mode = True
        X, y, time_stamps = load_data()
        (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
        train_dataset = StockDataset(X_train[:100], y_train[:100])
        val_dataset = StockDataset(X_val[:50], y_val[:50])
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        # 创建Learner
        learner = Learner('./test_output', model, train_loader, None, optimizer, params, dev_dataset=val_loader)
        print(f"✓ Learner创建成功")
        
        # 测试训练步骤
        for batch in train_loader:
            loss = learner.train_step(batch, 1)
            print(f"✓ Learner训练步骤成功")
            print(f"  - 损失值: {loss:.4f}")
            break
        
        # 测试评估
        rmse, r2, pearson = learner.evaluate_dev()
        if rmse is not None:
            print(f"✓ Learner评估成功")
            print(f"  - RMSE: {rmse:.4f}")
            print(f"  - R2: {r2:.4f}")
            print(f"  - Pearson: {pearson:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Learner测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试量化模型和数据集...")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_dataset,
        test_model,
        test_training_step,
        test_learner
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"测试 {test.__name__} 出现异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常使用。")
        return True
    else:
        print("❌ 部分测试失败，请检查配置和数据。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
