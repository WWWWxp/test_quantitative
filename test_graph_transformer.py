#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Graph Transformer模型
"""
import sys
import os
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from params import params
from dataset import load_data, split_time_series_data, StockDataset
from model import create_model
from torch.utils.data import DataLoader

def test_graph_transformer():
    """测试Graph Transformer模型"""
    print("=" * 60)
    print("测试Graph Transformer模型")
    print("=" * 60)
    
    try:
        # 启用测试模式
        params.test_mode = True
        
        # 加载数据
        print("1. 加载数据...")
        X, y, time_stamps = load_data()
        (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
        
        print(f"   训练集: X={X_train.shape}, y={y_train.shape}")
        print(f"   验证集: X={X_val.shape}, y={y_val.shape}")
        print(f"   测试集: X={X_test.shape}, y={y_test.shape}")
        
        # 创建数据集
        print("2. 创建数据集...")
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        print(f"   训练批次数量: {len(train_loader)}")
        print(f"   验证批次数量: {len(val_loader)}")
        
        # 创建模型
        print("3. 创建模型...")
        model = create_model(params)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"   模型类型: {type(model).__name__}")
        print(f"   设备: {device}")
        
        # 测试前向传播
        print("4. 测试前向传播...")
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                output = model(X_batch)
                
                print(f"   输入形状: {X_batch.shape}")
                print(f"   输出形状: {output.shape}")
                print(f"   目标形状: {y_batch.shape}")
                print(f"   输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
                break
        
        # 测试训练步骤
        print("5. 测试训练步骤...")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)
        criterion = torch.nn.MSELoss()
        
        for batch in train_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            print(f"   损失值: {loss.item():.4f}")
            break
        
        print("✓ Graph Transformer模型测试通过")
        return True
        
    except Exception as e:
        print(f"✗ Graph Transformer模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_graph_transformer()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 所有测试通过！")
    else:
        print("❌ 测试失败，请检查错误信息")
    print("=" * 60)
