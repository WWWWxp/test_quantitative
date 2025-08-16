# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sys
import random
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import warnings
import matplotlib.font_manager as fm

# ---------------- Font & logging ----------------
def setup_font():
    preferred_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'PingFang SC', 'Arial Unicode MS']
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for font in preferred_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            break
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans']
    plt.rcParams['axes.unicode_minus'] = False

def setup_logging():
    if logging.getLogger().hasHandlers(): return
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)
setup_font()
setup_logging()
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# CUDA 加速开关
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# ---------------- Config ----------------
feat_dir = "/home/hxcui/mnt/nas_q/projects/project_haoxin/feature/df_feature_13_window124_std15_20"
label_path = "/home/hxcui/mnt/nas_q/projects/project_haoxin/label/df_label_5d.pkl"

batch_size = 2048     # 若显存富余可试 3072/4096
epochs = 50
lr = 1e-3
patience = 10
OUTPUT_ROOT = "/home/hxcui/mnt/nas_q/projects/project_haoxin/output/gru_only_baseline"
num_workers = 8
time_window = "day"
random_seeds = [42, 123, 456, 789, 1000]

time_steps = 15
num_nodes = 12
num_samples = 8
flat_dim = num_nodes * num_samples  # 96

GRU_CONFIG = {
    "input_dim": flat_dim,   # 96
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.05,
    "fc_dims": [64],
    "use_batch_norm": False,
    "bidirectional": False,
    "pooling": "tail_mean",
    "tail_k": 8,
}



# data 600w
# 2048
# model para 


# ---------------- Dataset ----------------
class StockDataset(Dataset):
    def __init__(self, X, y):
        # X: [N, 15, 12, 8]
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ---------------- Model (GRU-only) ----------------
class TemporalModule(nn.Module):
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
        if self.batch_norm is not None: fused = self.batch_norm(fused)
        main_out = self.fc_block(fused)
        if p == "tail_mean":
            k = self.config.get("tail_k", 16)
            skip_out = self.linear_skip(x[:, -k:, :].mean(1))
        else:
            skip_out = self.linear_skip(x[:, -1, :])
        return main_out + skip_out

class GRUOnlyModel(nn.Module):
    """把每个时间步的 12×8 展平为 96 维，直接送入 GRU。"""
    def __init__(self, config):
        super().__init__()
        self.temporal = TemporalModule(config)

    def forward(self, x):  # x: [B, 15, 12, 8]
        B, T, N, F = x.shape
        x = x.view(B, T, N * F).contiguous()   # [B, T, 96]
        return self.temporal(x)

# ---------------- Data loading ----------------
def create_adjacency_matrix():
    # 占位（与 GNN+GRU 版本接口保持一致），本脚本不用。
    return np.eye(12, dtype=np.float32)

def load_data():
    logger.info("开始加载标签数据")
    with open(label_path, "rb") as f:
        df_label = pickle.load(f).fillna(0)
    df_label.index = df_label.index.set_levels(
        df_label.index.levels[0].strftime("%Y-%m-%d"), level=0
    )
    X_list, y_list, time_stamps_list = [], [], []
    years_list = sorted([f for f in os.listdir(feat_dir) if f.endswith(".pickle")])[:10]

    for fname in tqdm(years_list, desc="读取特征数据"):
        df_feat = pd.read_pickle(os.path.join(feat_dir, fname)).fillna(0)
        df_feat = df_feat.loc[:, ~df_feat.columns.str.contains('ElapsedTime', case=False)]
        df_feat = df_feat.loc[:, ~df_feat.columns.str.contains('_12[0-3]', case=False)]
        df_feat = df_feat.clip(upper=100)
        df_feat.index = df_feat.index.set_levels(
            pd.to_datetime(df_feat.index.levels[0]).strftime("%Y-%m-%d"), level=0
        )
        common_idx = df_feat.index.intersection(df_label.index)
        logger.info(f"处理文件: {fname}, 有效样本数: {len(common_idx)}, 丢失labels数: {len(df_feat) - len(common_idx)}")

        arr = df_feat.loc[common_idx].values
        expected_dim = time_steps * num_nodes * num_samples
        assert arr.shape[1] == expected_dim, f"特征维度不匹配: {arr.shape[1]} != {expected_dim}"

        X_part = arr.reshape(-1, time_steps, num_nodes, num_samples)  # [*, 15, 12, 8]
        y_part = df_label.loc[common_idx, 'label_5d'].values
        time_stamps_part = [idx[0] for idx in common_idx]

        X_list.append(X_part); y_list.append(y_part); time_stamps_list.extend(time_stamps_part)

    X = np.vstack(X_list); y = np.concatenate(y_list); time_stamps = np.array(time_stamps_list)
    order = np.argsort(time_stamps.astype('datetime64[D]'))
    X, y, time_stamps = X[order], y[order], time_stamps[order]

    logger.info(f"数据加载完成，总样本数: {len(X)}, 形状: {X.shape}")
    logger.info(f"时间范围（全体数据）: {time_stamps[0]} 至 {time_stamps[-1]}")
    return X, y, time_stamps

def split_time_series_data(X, y, time_stamps):
    ts = pd.to_datetime(time_stamps); years = ts.year
    train_idx = np.where((years >= 2011) & (years <= 2018))[0]
    val_idx   = np.where(years == 2019)[0]
    test_idx  = np.where(years == 2020)[0]
    X_train, y_train, ts_train = X[train_idx], y[train_idx], time_stamps[train_idx]
    X_val,   y_val,   ts_val   = X[val_idx], y[val_idx], time_stamps[val_idx]
    X_test,  y_test,  ts_test  = X[test_idx], y[test_idx], time_stamps[test_idx]
    logger.info("\n===== 数据集日期范围 =====")
    if len(ts_train): logger.info(f"训练集: {ts_train.min()} 至 {ts_train.max()}（样本数: {len(ts_train)}）")
    if len(ts_val):   logger.info(f"验证集: {ts_val.min()} 至 {ts_val.max()}（样本数: {len(ts_val)}）")
    if len(ts_test):  logger.info(f"测试集: {ts_test.min()} 至 {ts_test.max()}（样本数: {len(ts_test)}）")
    return (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test)

# ---------------- Metrics & plots ----------------
def group_by_time_window(time_stamps, preds, targets, window="day"):
    df = pd.DataFrame({"date": pd.to_datetime(time_stamps),
                       "pred": preds.flatten(),
                       "target": targets.flatten()})
    df["window"] = df["date"].dt.strftime("%Y-%m-%d" if window == "day" else "%Y-%W")
    groups = []
    for w, g in df.groupby("window"):
        if len(g) >= 10:
            groups.append({"window": w, "preds": g["pred"].values, "targets": g["target"].values, "count": len(g)})
    logger.info(f"按{window}分组完成，有效窗口数: {len(groups)}")
    return groups

def plot_pred_vs_true(y_true, y_pred, out_path, title="", rmse=None, max_points=200_000, seed=123):
    y_true = np.asarray(y_true).reshape(-1); y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) == 0: return
    if len(y_true) > max_points:
        rng = np.random.default_rng(seed); idx = rng.choice(len(y_true), size=max_points, replace=False)
        xs, ys = y_true[idx], y_pred[idx]
    else:
        xs, ys = y_true, y_pred
    if rmse is None: rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    try: a, b = np.polyfit(y_true, y_pred, 1)
    except Exception: a, b = 1.0, 0.0
    vmin = float(min(xs.min(), ys.min())); vmax = float(max(xs.max(), ys.max())); pad = 0.05 * (vmax - vmin + 1e-8)
    lo, hi = vmin - pad, vmax + pad
    xline = np.linspace(lo, hi, 256); y_diag = xline; y_low = xline - rmse; y_high = xline + rmse; y_cal = a*xline + b
    plt.figure(figsize=(6.2, 6.2), dpi=140)
    plt.scatter(xs, ys, s=2, alpha=0.35, linewidths=0)
    plt.plot(xline, y_diag, linestyle='-', linewidth=1.0, label='y = x')
    plt.fill_between(xline, y_low, y_high, alpha=0.15, label=f'±RMSE ({rmse:.4f})')
    plt.plot(xline, y_cal, linestyle='--', linewidth=1.0, label=f'Calib: y = {a:.2f}x + {b:.2f}')
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("True"); plt.ylabel("Pred"); plt.title(title)
    plt.legend(loc='best', frameon=True); plt.grid(alpha=0.25, linestyle='--')
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def calculate_window_ic(groups):
    rows = []
    for g in groups:
        x, y = g["preds"], g["targets"]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8: continue
        p, _ = pearsonr(x, y); s, _ = spearmanr(x, y)
        rows.append({"window": g["window"], "pearson_ic": p, "spearman_ic": s, "sample_count": g["count"]})
    return pd.DataFrame(rows)

def calculate_icr(ic_series):
    if len(ic_series) < 2: return 0.0
    mean_ic, std_ic = np.mean(ic_series), np.std(ic_series)
    return mean_ic / std_ic if std_ic > 1e-6 else 0.0

# ---------------- Train one experiment ----------------
def run_single_experiment(seed, X, y, time_stamps):
    logger.info(f"\n===== 开始实验 {seed}（随机种子: {seed}） =====")
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = True

    exp_out = os.path.join(OUTPUT_ROOT, f"exp_seed_{seed}")
    exp_model_dir = os.path.join(exp_out, "model"); os.makedirs(exp_model_dir, exist_ok=True)

    (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
    logger.info(f"Train X shape:{X_train.shape}|finite={np.isfinite(X_train).all()}")
    logger.info(f"Val   X shape:{X_val.shape}|finite={np.isfinite(X_val).all()}")

    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
                              num_workers=max(8, num_workers), pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader   = DataLoader(StockDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False,
                              num_workers=max(4, num_workers//2), pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)
    test_loader  = DataLoader(StockDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False,
                              num_workers=max(4, num_workers//2), pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)

    if torch.cuda.is_available():
        device_idx = 1 if torch.cuda.device_count() > 1 else 0
        device = torch.device(f"cuda:{device_idx}")
        dev_name = torch.cuda.get_device_name(device_idx)
    else:
        device = torch.device("cpu"); dev_name = "CPU"
    logger.info(f"使用设备: {device} (名称: {dev_name})")

    model = GRUOnlyModel(GRU_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.5)

    steps_per_epoch = len(train_loader)
    epochs_sched = min(20, epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=steps_per_epoch, epochs=epochs_sched,
        pct_start=0.1, div_factor=10.0, final_div_factor=10.0
    )
    scaler = GradScaler(enabled=True)

    model_path = os.path.join(exp_model_dir, "gru_only_model.pth")
    best_val_loss = float('inf'); early_stop = 0
    train_losses, val_losses = [], []

    logger.info("开始训练...")
    global_step = 0
    for epoch in range(epochs):
        model.train(); total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (seed {seed})")
        for Xb, yb in pbar:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(Xb)
                loss = criterion(out, yb)
            if not torch.isfinite(loss):
                logger.warning(f"Loss is inf/NaN at step {global_step}"); scaler.update(); continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            if epoch < epochs_sched: scheduler.step()
            total_loss += loss.item(); global_step += 1

        avg_train = total_loss / max(1, len(train_loader))
        train_losses.append(avg_train)

        model.eval(); vloss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model(Xb)
                    vloss += F.mse_loss(out, yb).item()
        avg_val = vloss / max(1, len(val_loader))
        val_losses.append(avg_val)
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss - 1e-6:
            best_val_loss = avg_val; torch.save(model.state_dict(), model_path); early_stop = 0
        else:
            early_stop += 1
            if early_stop >= patience:
                logger.info("触发Early Stopping"); break

    # ---- 评估 ----
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def collect_preds(loader):
        preds, trues = [], []
        with torch.no_grad():
            for Xb, yb in loader:
                Xb = Xb.to(device, non_blocking=True)
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model(Xb).cpu().float().numpy()
                preds.append(out); trues.append(yb.numpy())
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        logger.info(f"检查预测结果:{np.isfinite(preds).all()} | 目标:{np.isfinite(trues).all()}")
        return preds, trues

    os.makedirs(exp_out, exist_ok=True)
    val_preds, val_targets = collect_preds(val_loader)
    test_preds, test_targets = collect_preds(test_loader)

    pd.DataFrame(val_preds).to_csv(os.path.join(exp_out, "val_preds.csv"), index=False)
    pd.DataFrame(val_targets).to_csv(os.path.join(exp_out, "val_targets.csv"), index=False)
    pd.DataFrame(test_preds).to_csv(os.path.join(exp_out, "test_preds.csv"), index=False)
    pd.DataFrame(test_targets).to_csv(os.path.join(exp_out, "test_targets.csv"), index=False)

    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    val_r2 = r2_score(val_targets, val_preds)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    test_r2 = r2_score(test_targets, test_preds)

    baseline_rmse = np.sqrt(np.mean((val_targets - val_targets.mean()) ** 2))
    logger.info(f"[诊断] Val preds std: {np.std(val_preds):.6f} | y std: {np.std(val_targets):.6f} | Baseline RMSE: {baseline_rmse:.6f}")

    a, b = np.polyfit(val_preds.flatten(), val_targets.flatten(), 1)
    val_preds_cal = val_preds * a + b
    test_preds_cal = test_preds * a + b
    val_rmse_cal = np.sqrt(mean_squared_error(val_targets, val_preds_cal))
    val_r2_cal = r2_score(val_targets, val_preds_cal)
    test_rmse_cal = np.sqrt(mean_squared_error(test_targets, test_preds_cal))
    test_r2_cal = r2_score(test_targets, test_preds_cal)
    logger.info(f"[Cal] Val RMSE: {val_rmse_cal:.4f} | Val R2: {val_r2_cal:.4f}")

    plot_pred_vs_true(val_targets, val_preds,
                      os.path.join(exp_out, "val_pred_vs_true.png"),
                      title=f"Val | RMSE={val_rmse:.4f}")
    plot_pred_vs_true(val_targets, val_preds_cal,
                      os.path.join(exp_out, "val_pred_vs_true_cal.png"),
                      title=f"Val | Calibrated RMSE={val_rmse_cal:.4f}")

    val_groups = group_by_time_window(ts_val, val_preds, val_targets)
    test_groups = group_by_time_window(ts_test, test_preds, test_targets)
    val_ic_df = calculate_window_ic(val_groups)
    test_ic_df = calculate_window_ic(test_groups)

    val_avg_pearson = float(val_ic_df['pearson_ic'].mean()) if len(val_ic_df) else float("nan")
    test_avg_pearson = float(test_ic_df['pearson_ic'].mean()) if len(test_ic_df) else float("nan")
    val_icr = float(calculate_icr(val_ic_df['pearson_ic'])) if len(val_ic_df) else 0.0
    test_icr = float(calculate_icr(test_ic_df['pearson_ic'])) if len(test_ic_df) else 0.0

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Val Loss')
    plt.title(f"Learning Curve (seed {seed})"); plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend()
    plt.savefig(os.path.join(exp_out, "learning_curve.png")); plt.close()

    val_ic_df.to_csv(os.path.join(exp_out, "val_ic.csv"), index=False)
    test_ic_df.to_csv(os.path.join(exp_out, "test_ic.csv"), index=False)

    logger.info(f"\n===== 实验 {seed} 评估结果 =====")
    logger.info(f"验证集 - RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}, 平均Pearson IC: {val_avg_pearson if not np.isnan(val_avg_pearson) else 'nan'}, ICR: {val_icr:.4f}")
    logger.info(f"测试集 - RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, 平均Pearson IC: {test_avg_pearson if not np.isnan(test_avg_pearson) else 'nan'}, ICR: {test_icr:.4f}")

    return {
        "seed": seed,
        "val_rmse": val_rmse, "val_r2": val_r2,
        "val_avg_pearson": val_avg_pearson, "val_icr": val_icr,
        "test_rmse": test_rmse, "test_r2": test_r2,
        "test_avg_pearson": test_avg_pearson, "test_icr": test_icr
    }

# ---------------- Runner ----------------
def run_multi_experiments():
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda:1"); logger.info(f"使用指定GPU: {device} (名称: {torch.cuda.get_device_name(1)})")
    elif torch.cuda.is_available():
        device = torch.device("cuda"); logger.info(f"只有一个GPU可用，使用默认GPU: {device}")
    else:
        device = torch.device("cpu"); logger.info("无可用GPU，使用CPU进行训练")

    _ = create_adjacency_matrix()  # 与GNN版接口保持一致（未使用）
    X, y, time_stamps = load_data()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    results = [run_single_experiment(seed, X, y, time_stamps) for seed in random_seeds]
    df = pd.DataFrame(results); avg = df.mean(numeric_only=True).to_dict()

    logger.info("\n===== 实验平均结果（GRU-only） =====")
    logger.info(f"验证集平均 - RMSE: {avg.get('val_rmse', np.nan):.4f}, R2: {avg.get('val_r2', np.nan):.4f}, "
                f"平均Pearson IC: {avg.get('val_avg_pearson', np.nan):.4f}, ICR: {avg.get('val_icr', np.nan):.4f}")
    logger.info(f"测试集平均 - RMSE: {avg.get('test_rmse', np.nan):.4f}, R2: {avg.get('test_r2', np.nan):.4f}, "
                f"平均Pearson IC: {avg.get('test_avg_pearson', np.nan):.4f}, ICR: {avg.get('test_icr', np.nan):.4f}")

    df.to_csv(os.path.join(OUTPUT_ROOT, "all_experiments_results.csv"), index=False)
    pd.DataFrame([avg]).to_csv(os.path.join(OUTPUT_ROOT, "average_results.csv"), index=False)
    logger.info(f"所有实验结果已保存至 {OUTPUT_ROOT}")

if __name__ == "__main__":
    run_multi_experiments()
