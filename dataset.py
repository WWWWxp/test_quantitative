# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, List, Optional
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from multiprocessing import cpu_count

from params import params

warnings.filterwarnings("ignore")

# ---------- 量化数据集相关函数 ----------
def create_adjacency_matrix():
    """创建邻接矩阵（占位函数，与GNN版本接口保持一致）"""
    return np.eye(12, dtype=np.float32)

def create_test_data():
    """创建测试用的模拟数据"""
    print("创建测试数据...")
    
    # 创建模拟特征数据
    num_samples = 1000
    time_steps = params.time_steps
    num_nodes = params.num_nodes
    num_samples_per_node = params.num_samples
    
    # 生成随机特征数据
    X = np.random.randn(num_samples, time_steps, num_nodes, num_samples_per_node).astype(np.float32)
    
    # 生成模拟标签数据（基于特征的简单线性组合）
    y = np.random.randn(num_samples).astype(np.float32)
    
    # 生成时间戳
    time_stamps = pd.date_range('2020-01-01', periods=num_samples, freq='D').strftime('%Y-%m-%d').values
    
    print(f"测试数据创建完成: X.shape={X.shape}, y.shape={y.shape}")
    return X, y, time_stamps

def load_data():
    """加载量化数据"""
    import logging
    logger = logging.getLogger(__name__)
    
    # 检查是否为测试模式
    if getattr(params, 'test_mode', False):
        print("使用测试模式，创建模拟数据...")
        return create_test_data()
    
    logger.info("开始加载标签数据")
    with open(params.label_path, "rb") as f:
        df_label = pickle.load(f).fillna(0)
    df_label.index = df_label.index.set_levels(
        df_label.index.levels[0].strftime("%Y-%m-%d"), level=0
    )
    
    X_list, y_list, time_stamps_list = [], [], []
    years_list = sorted([f for f in os.listdir(params.feat_dir) if f.endswith(".pickle")])[:10]

    for fname in tqdm(years_list, desc="读取特征数据"):
        df_feat = pd.read_pickle(os.path.join(params.feat_dir, fname)).fillna(0)
        df_feat = df_feat.loc[:, ~df_feat.columns.str.contains('ElapsedTime', case=False)]
        df_feat = df_feat.loc[:, ~df_feat.columns.str.contains('_12[0-3]', case=False)]
        df_feat = df_feat.clip(upper=100)
        df_feat.index = df_feat.index.set_levels(
            pd.to_datetime(df_feat.index.levels[0]).strftime("%Y-%m-%d"), level=0
        )
        common_idx = df_feat.index.intersection(df_label.index)
        logger.info(f"处理文件: {fname}, 有效样本数: {len(common_idx)}, 丢失labels数: {len(df_feat) - len(common_idx)}")

        arr = df_feat.loc[common_idx].values
        expected_dim = params.time_steps * params.num_nodes * params.num_samples
        assert arr.shape[1] == expected_dim, f"特征维度不匹配: {arr.shape[1]} != {expected_dim}"

        X_part = arr.reshape(-1, params.time_steps, params.num_nodes, params.num_samples)  # [*, 15, 12, 8]
        y_part = df_label.loc[common_idx, 'label_5d'].values
        time_stamps_part = [idx[0] for idx in common_idx]

        X_list.append(X_part)
        y_list.append(y_part)
        time_stamps_list.extend(time_stamps_part)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    time_stamps = np.array(time_stamps_list)
    
    order = np.argsort(time_stamps.astype('datetime64[D]'))
    X, y, time_stamps = X[order], y[order], time_stamps[order]

    logger.info(f"数据加载完成，总样本数: {len(X)}, 形状: {X.shape}")
    logger.info(f"时间范围（全体数据）: {time_stamps[0]} 至 {time_stamps[-1]}")
    return X, y, time_stamps

def split_time_series_data(X, y, time_stamps):
    """按时间分割数据"""
    # 检查是否为测试模式
    if getattr(params, 'test_mode', False):
        # 测试模式：简单分割
        total_samples = len(X)
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.85)
        
        X_train, y_train, ts_train = X[:train_end], y[:train_end], time_stamps[:train_end]
        X_val, y_val, ts_val = X[train_end:val_end], y[train_end:val_end], time_stamps[train_end:val_end]
        X_test, y_test, ts_test = X[val_end:], y[val_end:], time_stamps[val_end:]
        
        print(f"测试模式数据分割: 训练集={len(X_train)}, 验证集={len(X_val)}, 测试集={len(X_test)}")
        return (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test)
    
    ts = pd.to_datetime(time_stamps)
    years = ts.year
    train_idx = np.where((years >= 2011) & (years <= 2018))[0]
    val_idx = np.where(years == 2019)[0]
    test_idx = np.where(years == 2020)[0]
    
    X_train, y_train, ts_train = X[train_idx], y[train_idx], time_stamps[train_idx]
    X_val, y_val, ts_val = X[val_idx], y[val_idx], time_stamps[val_idx]
    X_test, y_test, ts_test = X[test_idx], y[test_idx], time_stamps[test_idx]
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info("\n===== 数据集日期范围 =====")
    if len(ts_train): 
        logger.info(f"训练集: {ts_train.min()} 至 {ts_train.max()}（样本数: {len(ts_train)}）")
    if len(ts_val):   
        logger.info(f"验证集: {ts_val.min()} 至 {ts_val.max()}（样本数: {len(ts_val)}）")
    if len(ts_test):  
        logger.info(f"测试集: {ts_test.min()} 至 {ts_test.max()}（样本数: {len(ts_test)}）")
    
    return (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test)

# ---------- 量化数据集类 ----------
class StockDataset(Dataset):
    """量化数据集类"""
    def __init__(self, X, y):
        # X: [N, 15, 12, 8]
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    def __len__(self): 
        return len(self.X)
    
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

# ---------- 数据加载器工厂函数 ----------
def from_train_list(list_file: str, audio_root: str, params, is_distributed=False, max_duration: float = 4, normalize_audio: bool = True):
    """
    创建训练数据加载器（量化版本）
    
    Args:
        list_file: 数据列表文件路径（不使用，保持接口兼容）
        audio_root: 音频文件根目录（不使用，保持接口兼容）
        params: 包含所有配置参数的对象
        is_distributed: 是否使用分布式训练
        max_duration: 音频最大时长（不使用，保持接口兼容）
        normalize_audio: 是否进行音频归一化（不使用，保持接口兼容）
    """
    # 加载量化数据
    X, y, time_stamps = load_data()
    (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
    
    # 创建数据集
    train_dataset = StockDataset(X_train, y_train)
    
    # 创建数据加载器
    return DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=not is_distributed,
        sampler=DistributedSampler(train_dataset) if is_distributed else None,
        pin_memory=True,
        persistent_workers=True,
        num_workers=min(8, cpu_count()),
        drop_last=True
    )

def from_test_list(test_file: str, audio_root: str, params, batch_size: int = 16, normalize_audio: bool = True):
    """
    创建测试数据加载器（量化版本）
    
    Args:
        test_file: 测试文件路径（不使用，保持接口兼容）
        audio_root: 音频文件根目录（不使用，保持接口兼容）
        params: 包含所有配置参数的对象
        batch_size: 批次大小
        normalize_audio: 是否进行音频归一化（不使用，保持接口兼容）
    """
    # 加载量化数据
    X, y, time_stamps = load_data()
    (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
    
    # 创建测试数据集
    test_dataset = StockDataset(X_test, y_test)
    
    # 创建数据加载器
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=min(8, cpu_count()),
        drop_last=False
    )

# ---------- 快速测试 ----------
if __name__ == "__main__":
    from tqdm import tqdm
    
    # 启用测试模式
    params.test_mode = True
    
    # 测试数据加载
    try:
        X, y, time_stamps = load_data()
        print(f"数据加载成功: X.shape={X.shape}, y.shape={y.shape}")
        
        # 测试数据集
        dataset = StockDataset(X[:100], y[:100])
        print(f"数据集大小: {len(dataset)}")
        
        # 测试数据加载器
        loader = from_train_list("dummy", "dummy", params)
        print(f"数据加载器创建成功，批次数量: {len(loader)}")
        
        # 测试一个批次
        for batch in loader:
            X_batch, y_batch = batch
            print(f"批次形状: X={X_batch.shape}, y={y_batch.shape}")
            break
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()