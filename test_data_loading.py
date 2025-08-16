#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据加载功能
"""
import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from params import params
from dataset import load_data, split_time_series_data, StockDataset
from torch.utils.data import DataLoader

def test_csv_loading():
    """测试CSV数据加载"""
    print("=" * 60)
    print("测试CSV数据加载")
    print("=" * 60)
    
    try:
        # 测试标签数据加载
        print("1. 测试标签数据加载...")
        label_df = pd.read_csv(params.label_csv_path)
        print(f"   标签数据形状: {label_df.shape}")
        print(f"   列名: {list(label_df.columns)}")
        print(f"   前5行:")
        print(label_df.head())
        
        # 测试特征数据加载
        print("\n2. 测试特征数据加载...")
        feat_df = pd.read_csv(params.feat_path)
        print(f"   特征数据形状: {feat_df.shape}")
        print(f"   前5行:")
        print(feat_df.head())
        
        print("\n✓ CSV数据加载测试通过")
        return True
        
    except Exception as e:
        print(f"✗ CSV数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing():
    """测试数据处理"""
    print("\n" + "=" * 60)
    print("测试数据处理")
    print("=" * 60)
    
    try:
        # 启用测试模式
        params.test_mode = True
        
        # 测试数据加载
        print("1. 测试数据加载函数...")
        X, y, time_stamps = load_data()
        print(f"   X形状: {X.shape}")
        print(f"   y形状: {y.shape}")
        print(f"   时间戳数量: {len(time_stamps)}")
        
        # 测试数据分割
        print("\n2. 测试数据分割...")
        (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
        print(f"   训练集: X={X_train.shape}, y={y_train.shape}")
        print(f"   验证集: X={X_val.shape}, y={y_val.shape}")
        print(f"   测试集: X={X_test.shape}, y={y_test.shape}")
        
        # 测试数据集类
        print("\n3. 测试数据集类...")
        dataset = StockDataset(X_train[:100], y_train[:100])
        print(f"   数据集大小: {len(dataset)}")
        
        # 测试数据加载器
        print("\n4. 测试数据加载器...")
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        print(f"   批次数量: {len(loader)}")
        
        # 测试一个批次
        for batch in loader:
            X_batch, y_batch = batch
            print(f"   批次形状: X={X_batch.shape}, y={y_batch.shape}")
            break
        
        print("\n✓ 数据处理测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 数据处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_data_loading():
    """测试真实数据加载"""
    print("\n" + "=" * 60)
    print("测试真实数据加载")
    print("=" * 60)
    
    try:
        # 禁用测试模式
        params.test_mode = False
        
        # 检查文件是否存在
        if not os.path.exists(params.label_csv_path):
            print(f"   标签文件不存在: {params.label_csv_path}")
            return False
            
        if not os.path.exists(params.feat_path):
            print(f"   特征文件不存在: {params.feat_path}")
            return False
        
        # 测试数据加载
        print("1. 测试真实数据加载...")
        X, y, time_stamps = load_data()
        print(f"   X形状: {X.shape}")
        print(f"   y形状: {y.shape}")
        print(f"   时间戳数量: {len(time_stamps)}")
        
        # 测试数据分割
        print("\n2. 测试真实数据分割...")
        (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
        print(f"   训练集: X={X_train.shape}, y={y_train.shape}")
        print(f"   验证集: X={X_val.shape}, y={y_val.shape}")
        print(f"   测试集: X={X_test.shape}, y={y_test.shape}")
        
        print("\n✓ 真实数据加载测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 真实数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始数据加载测试...")
    
    # 测试1: CSV数据加载
    test1_passed = test_csv_loading()
    
    # 测试2: 数据处理
    test2_passed = test_data_processing()
    
    # 测试3: 真实数据加载
    test3_passed = test_real_data_loading()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"CSV数据加载: {'✓ 通过' if test1_passed else '✗ 失败'}")
    print(f"数据处理: {'✓ 通过' if test2_passed else '✗ 失败'}")
    print(f"真实数据加载: {'✓ 通过' if test3_passed else '✗ 失败'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 所有测试通过！")
        return True
    else:
        print("\n❌ 部分测试失败，请检查错误信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
