# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
from collections import defaultdict
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


def get_accuracy(output, label):
    """分类准确率（保留用于兼容性）"""
    _, prediction = torch.max(output, 1)    # argmax
    correct = (prediction == label).sum().item()
    accuracy = correct / prediction.size(0)
    return accuracy


def get_prediction(output, label):
    """分类预测（保留用于兼容性）"""
    prob = nn.functional.softmax(output, dim=1)[:, 1]
    prob = prob.view(prob.size(0), 1)
    label = label.view(label.size(0), 1)
    datas = torch.cat((prob, label.float()), dim=1)
    return datas


def calculate_metrics_for_train(label, output):
    """计算训练时的指标（回归版本）"""
    # Ensure inputs are 1D
    label = label.squeeze()
    output = output.squeeze()
    
    # 转换为numpy数组
    y_true = label.cpu().detach().numpy().flatten()
    y_pred = output.cpu().detach().numpy().flatten()
    
    # 计算回归指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0, 0)
    
    # 计算准确率（基于预测值与真实值的接近程度）
    # 这里使用一个简单的阈值方法：如果预测值与真实值的相对误差小于10%，认为准确
    relative_error = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)
    accuracy = np.mean(relative_error < 0.1)  # 10%相对误差阈值
    
    return rmse, r2, accuracy, pearson


def compute_class_losses(pred, label, loss_func):
    """计算损失（回归版本）"""
    # Overall loss
    loss = loss_func(pred, label)
    
    # 对于回归任务，我们不需要分类别损失
    # 但为了保持接口兼容性，返回相同的结构
    loss_dict = {
        'overall': loss,
        'real_loss': loss,  # 回归任务中所有样本使用相同损失
        'fake_loss': loss,
    }
    
    return loss_dict


# ------------ compute average metrics of batches ---------------------
class Metrics_batch():
    def __init__(self):
        self.rmse_list = []
        self.r2_list = []
        self.pearson_list = []
        self.accuracy_list = []
        self.losses = []

    def update(self, label, output):
        # 计算回归指标
        y_true = label.cpu().detach().numpy().flatten()
        y_pred = output.cpu().detach().numpy().flatten()
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        pearson, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0, 0)
        
        # 计算准确率（基于相对误差）
        relative_error = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)
        accuracy = np.mean(relative_error < 0.1)
        
        self.rmse_list.append(rmse)
        self.r2_list.append(r2)
        self.pearson_list.append(pearson)
        self.accuracy_list.append(accuracy)
        
        return accuracy, r2, rmse, pearson

    def get_mean_metrics(self):
        mean_rmse = np.mean(self.rmse_list) if self.rmse_list else 0
        mean_r2 = np.mean(self.r2_list) if self.r2_list else 0
        mean_pearson = np.mean(self.pearson_list) if self.pearson_list else 0
        mean_accuracy = np.mean(self.accuracy_list) if self.accuracy_list else 0
        
        return {
            'rmse': mean_rmse, 
            'r2': mean_r2, 
            'pearson': mean_pearson, 
            'accuracy': mean_accuracy
        }

    def clear(self):
        self.rmse_list.clear()
        self.r2_list.clear()
        self.pearson_list.clear()
        self.accuracy_list.clear()
        self.losses.clear()


# ------------ compute average metrics of all data ---------------------
class Metrics_all():
    def __init__(self):
        self.preds = []
        self.labels = []

    def store(self, label, output):
        # 对于回归任务，直接存储预测值和标签
        self.labels.append(label.squeeze().cpu().numpy())
        self.preds.append(output.squeeze().cpu().numpy())

    def get_metrics(self):
        y_pred = np.concatenate(self.preds)
        y_true = np.concatenate(self.labels)
        
        # 计算回归指标
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        pearson, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0, 0)
        
        # 计算准确率
        relative_error = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)
        accuracy = np.mean(relative_error < 0.1)
        
        return {'accuracy': accuracy, 'rmse': rmse, 'r2': r2, 'pearson': pearson}

    def clear(self):
        self.preds.clear()
        self.labels.clear()


# only used to record a series of scalar value
class Recorder:
    def __init__(self):
        self.sum = 0
        self.num = 0

    def update(self, item, num=1):
        if item is not None:
            self.sum += item * num
            self.num += num

    def average(self):
        if self.num == 0:
            return None
        return self.sum/self.num

    def clear(self):
        self.sum = 0
        self.num = 0