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
    
    # 加载标签数据
    label_df = pd.read_csv(params.label_csv_path)
    label_df['datetime'] = pd.to_datetime(label_df['datetime'])
    
    # 处理股票代码，移除b'前缀和'后缀
    label_df['instrument'] = label_df['instrument'].str.replace("b'", "").str.replace("'", "")
    
    logger.info(f"标签数据加载完成，形状: {label_df.shape}")
    logger.info(f"时间范围: {label_df['datetime'].min()} 至 {label_df['datetime'].max()}")
    logger.info(f"股票数量: {label_df['instrument'].nunique()}")
    
    # 加载特征数据
    logger.info("开始加载特征数据")
    
    # 读取特征CSV文件
    feat_df = pd.read_csv(params.feat_path)
    
    # 第一列是时间戳，转换为datetime
    feat_df.iloc[:, 0] = pd.to_datetime(feat_df.iloc[:, 0])
    
    # 处理股票代码，移除b'前缀和'后缀
    feat_df['instrument'] = feat_df['instrument'].str.replace("b'", "").str.replace("'", "")
    
    logger.info(f"特征数据加载完成，形状: {feat_df.shape}")
    logger.info(f"时间范围: {feat_df.iloc[:, 0].min()} 至 {feat_df.iloc[:, 0].max()}")
    logger.info(f"股票数量: {feat_df['instrument'].nunique()}")
    
    # 找到共同的股票代码
    label_stocks = set(label_df['instrument'].unique())
    feat_stocks = set(feat_df['instrument'].unique())
    common_stocks = label_stocks.intersection(feat_stocks)
    logger.info(f"共同股票数量: {len(common_stocks)}")
    
    if len(common_stocks) == 0:
        raise ValueError("标签数据和特征数据没有共同的股票代码")
    
    # 筛选共同股票的数据
    label_filtered = label_df[label_df['instrument'].isin(common_stocks)].copy()
    feat_filtered = feat_df[feat_df['instrument'].isin(common_stocks)].copy()
    
    # 处理特征数据，移除异常值
    feat_numeric = feat_filtered.select_dtypes(include=[np.number])
    feat_numeric = feat_numeric.clip(upper=100)  # 限制上限
    feat_numeric = feat_numeric.fillna(0)
    
    # 选择前num_stocks个股票
    num_stocks = min(len(common_stocks), params.num_nodes)
    selected_stocks = list(common_stocks)[:num_stocks]
    
    # 获取唯一的时间戳
    label_dates = label_filtered['datetime'].unique()
    feat_dates = feat_filtered.iloc[:, 0].unique()
    
    logger.info(f"标签数据时间点: {len(label_dates)}")
    logger.info(f"特征数据时间点: {len(feat_dates)}")
    
    # 初始化结果数组
    time_steps = min(len(label_dates), params.time_steps)
    X_list = []
    y_list = []
    time_stamps_list = []
    
    # 按时间组织数据
    for i, date in enumerate(label_dates[:time_steps]):
        # 获取当前时间点的标签数据
        current_label = label_filtered[label_filtered['datetime'] == date]
        
        # 为每个股票创建特征向量和标签
        stock_features = []
        stock_labels = []
        
        for stock in selected_stocks:
            # 获取该股票的标签
            stock_label_data = current_label[current_label['instrument'] == stock]
            if len(stock_label_data) > 0:
                stock_label = stock_label_data['label_5d'].iloc[0]
            else:
                stock_label = 0.0
            
            # 获取该股票的特征（如果存在）
            stock_feat_data = feat_filtered[feat_filtered['instrument'] == stock]
            if len(stock_feat_data) > 0:
                # 获取数值特征
                stock_feat = stock_feat_data.select_dtypes(include=[np.number]).iloc[0].values
            else:
                # 如果特征不存在，使用零填充
                stock_feat = np.zeros(feat_numeric.shape[1])
            
            # 确保特征数量正确
            if len(stock_feat) < params.num_samples:
                # 如果特征不够，用零填充
                padded_feat = np.zeros(params.num_samples)
                padded_feat[:len(stock_feat)] = stock_feat
                stock_feat = padded_feat
            elif len(stock_feat) > params.num_samples:
                # 如果特征太多，截取前num_samples个
                stock_feat = stock_feat[:params.num_samples]
            
            stock_features.append(stock_feat)
            stock_labels.append(stock_label)
        
        # 确保股票数量正确
        while len(stock_features) < params.num_nodes:
            stock_features.append(np.zeros(params.num_samples))
            stock_labels.append(0.0)
        
        # 转换为numpy数组
        X_time = np.array(stock_features[:params.num_nodes], dtype=np.float32)
        y_time = np.array(stock_labels[:params.num_nodes], dtype=np.float32)
        
        X_list.append(X_time)
        y_list.append(y_time)
        time_stamps_list.append(date.strftime('%Y-%m-%d'))
    
    # 转换为最终格式
    X = np.array(X_list)  # [time_steps, num_nodes, num_samples]
    y = np.array(y_list)  # [time_steps, num_nodes]
    time_stamps = np.array(time_stamps_list)
    
    # 重新组织为 [samples, time_steps, num_nodes, num_samples] 格式
    # 这里需要创建滑动窗口，每个样本包含完整的时间序列
    if len(X) < params.time_steps:
        # 如果数据不够，直接使用现有数据并填充
        X_final = np.zeros((1, params.time_steps, params.num_nodes, params.num_samples))
        y_final = np.zeros(1)  # 改为1维
        X_final[0, :len(X), :, :] = X
        y_final[0] = y[-1, 0] if len(y) > 0 else 0.0  # 使用最后一个时间点的第一个股票标签
        time_stamps_final = np.array([time_stamps[-1] if len(time_stamps) > 0 else "2020-01-01"])
    else:
        # 创建滑动窗口样本
        num_samples = len(X) - params.time_steps + 1
        X_final = np.zeros((num_samples, params.time_steps, params.num_nodes, params.num_samples))
        y_final = np.zeros(num_samples)  # 改为1维
        time_stamps_final = []
        
        for i in range(num_samples):
            X_final[i] = X[i:i+params.time_steps]  # [time_steps, num_nodes, num_samples]
            y_final[i] = y[i+params.time_steps-1, 0]  # 使用窗口最后一天的第一个股票标签
            time_stamps_final.append(time_stamps[i+params.time_steps-1])
        
        time_stamps_final = np.array(time_stamps_final)
    
    logger.info(f"数据加载完成，最终形状: X={X_final.shape}, y={y_final.shape}")
    logger.info(f"时间范围: {time_stamps_final[0]} 至 {time_stamps_final[-1]}")
    
    return X_final, y_final, time_stamps_final

def split_time_series_data(X, y, time_stamps, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    分割时间序列数据为训练、验证和测试集
    
    Args:
        X: 特征数据 [samples, time_steps, num_nodes, num_samples]
        y: 标签数据 [samples, 1]
        time_stamps: 时间戳数组
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    Returns:
        (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test)
    """
    total_samples = len(X)
    
    # 如果样本数量太少，使用简单的分割
    if total_samples <= 3:
        # 对于很少的样本，全部用于训练
        return (X, X[:1], X[:1]), (y, y[:1], y[:1]), (time_stamps, time_stamps[:1], time_stamps[:1])
    
    # 计算分割点
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    # 分割数据
    X_train, y_train, ts_train = X[:train_end], y[:train_end], time_stamps[:train_end]
    X_val, y_val, ts_val = X[train_end:val_end], y[train_end:val_end], time_stamps[train_end:val_end]
    X_test, y_test, ts_test = X[val_end:], y[val_end:], time_stamps[val_end:]
    
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