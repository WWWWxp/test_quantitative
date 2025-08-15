#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU inference for Quantitative Analysis (GRU Model).
Uses torch.nn.DataParallel – simply export CUDA_VISIBLE_DEVICES
to control how many GPUs are used.
"""
import os, csv, argparse, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import from_test_list          # 自己的数据集工具
from params  import params                  # 配置
from model   import create_model           # 模型
warnings.filterwarnings("ignore")


# ---------- utils ----------
def compute_metrics(y_true, y_pred):
    """计算回归指标"""
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import pearsonr
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0, 0)
    
    return rmse, r2, pearson


# ---------- inference wrapper ----------
class Inference:
    def __init__(self, device, model_path, model):
        self.device     = device
        self.model_path = model_path
        self.model      = model.to(device).eval()

    # 兼容 DataParallel / 单卡
    def _load_state_dict(self, state):
        target = self.model.module if hasattr(self.model, "module") else self.model
        target.load_state_dict(state["model"], strict=True)

    def restore(self):
        ckpt = torch.load(self.model_path, map_location=self.device)
        self._load_state_dict(ckpt)

    @torch.no_grad()
    def predict(self, X):
        predictions = self.model(X)             # (B, 1)
        return predictions.cpu().numpy().ravel()


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_file",   default="./data/dummy.txt")
    parser.add_argument("--audio_root", default="./data/dummy")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_file",default="./result.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    # ---------- device & model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = create_model(params).eval()
    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        print(f"[Info] Detected {torch.cuda.device_count()} GPUs ➜ DataParallel enabled")
        base_model = torch.nn.DataParallel(base_model)

    infer = Inference(device, args.model_path, base_model)
    infer.restore()
    print(f"[Info] Model restored from {args.model_path}")

    # ---------- dataset ----------
    loader = from_test_list(args.wav_file, args.audio_root, params,
                            batch_size=args.batch_size)

    all_preds, all_targets = [], []

    # ---------- inference loop ----------
    for batch in tqdm(loader, desc="Inferencing"):
        if batch is None:              # collate 可能返回 None
            continue
        X, y = batch
        X = X.to(device)        # (B, T, N, F)
        preds = infer.predict(X)                 # (B,)
        all_preds.extend(preds)
        all_targets.extend(y.numpy().flatten())

    # ---------- 计算指标 ----------
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    rmse, r2, pearson = compute_metrics(all_targets, all_preds)
    
    print(f"[Results] RMSE: {rmse:.4f}, R2: {r2:.4f}, Pearson: {pearson:.4f}")

    # ---------- save results ----------
    results_df = pd.DataFrame({
        'target': all_targets,
        'prediction': all_preds,
        'error': all_targets - all_preds
    })
    results_df.to_csv(args.output_file, index=False)

    print(f"[Done] Results saved to {args.output_file}")
    print(f"Total samples: {len(all_preds)}")


if __name__ == "__main__":
    import pandas as pd
    main()
