
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import signal
import atexit

from tqdm import tqdm
import logging
import time
import matplotlib.pyplot as plt

from dataset import from_train_list, load_data, split_time_series_data, StockDataset
from model import create_model
from params import params
from metrics import calculate_metrics_for_train, compute_class_losses, Metrics_batch, Recorder

# 设置matplotlib中文字体
import matplotlib.font_manager as fm
def setup_font():
    """设置合适的字体以支持中文显示"""
    preferred_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'PingFang SC', 'Arial Unicode MS']
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    found_font = None
    for font in preferred_fonts:
        if font in available_fonts:
            found_font = font
            break

    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font]
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans']

    plt.rcParams['axes.unicode_minus'] = False

setup_font()

def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)

def summarize_model(net):
  total = sum(p.numel() for p in net.parameters())
  trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
  param_bytes = sum(p.numel() * p.element_size() for p in net.parameters())
  buffer_bytes = sum(b.numel() * b.element_size() for b in net.buffers())
  size_mb = (param_bytes + buffer_bytes) / (1024 ** 2)

  print(f"[Model Summary]")
  print(f"  • Total parameters    : {total:,}  ({total/1e6:.2f} M)")
  print(f"  • Trainable parameters: {trainable:,}  ({trainable/1e6:.2f} M)")
  print(f"  • Model size (MB)     : {size_mb:.2f}")

# ---------- 量化评估指标 ----------
def group_by_time_window(time_stamps, preds, targets, window="day"):
    """按时间窗口分组"""
    df = pd.DataFrame({
        "date": pd.to_datetime(time_stamps),
        "pred": preds.flatten(),
        "target": targets.flatten()
    })
    df["window"] = df["date"].dt.strftime("%Y-%m-%d" if window == "day" else "%Y-%W")

    groups = []
    for window_key, group in df.groupby("window"):
        if len(group) >= 10:
            groups.append({
                "window": window_key,
                "preds": group["pred"].values,
                "targets": group["target"].values,
                "count": len(group)
            })
    return groups

def calculate_window_ic(groups):
    """计算窗口IC"""
    ic_results = []
    for group in groups:
        x = group["preds"]
        y = group["targets"]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        pearson_ic, _ = pearsonr(x, y)
        spearman_ic, _ = spearmanr(x, y)
        ic_results.append({
            "window": group["window"],
            "pearson_ic": pearson_ic,
            "spearman_ic": spearman_ic,
            "sample_count": group["count"]
        })
    return pd.DataFrame(ic_results)

def calculate_icr(ic_series):
    """计算ICR"""
    if len(ic_series) < 2:
        return 0.0
    mean_ic = np.mean(ic_series)
    std_ic = np.std(ic_series)
    return mean_ic / std_ic if std_ic > 1e-6 else 0.0

def plot_pred_vs_true(y_true, y_pred, out_path, title="", rmse=None, max_points=200_000, seed=123):
    """绘制预测vs真实值散点图"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    n = len(y_true)
    if n == 0:
        return

    # 大样本自动子采样，减小png体积
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        xs = y_true[idx]
        ys = y_pred[idx]
    else:
        xs, ys = y_true, y_pred

    # 计算RMSE（若未传入）
    if rmse is None:
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # 拟合校准线 pred ≈ a * true + b（用于可视化）
    try:
        a, b = np.polyfit(y_true, y_pred, 1)
    except Exception:
        a, b = 1.0, 0.0

    vmin = float(min(xs.min(), ys.min()))
    vmax = float(max(xs.max(), ys.max()))
    pad = 0.05 * (vmax - vmin + 1e-8)
    lo, hi = vmin - pad, vmax + pad

    xline = np.linspace(lo, hi, 256)
    y_diag = xline  # y = x
    y_low  = xline - rmse
    y_high = xline + rmse
    y_cal  = a * xline + b

    plt.figure(figsize=(6.2, 6.2), dpi=140)
    plt.scatter(xs, ys, s=2, alpha=0.35, linewidths=0)
    plt.plot(xline, y_diag, linestyle='-', linewidth=1.0, label='y = x')
    plt.fill_between(xline, y_low, y_high, alpha=0.15, label=f'±RMSE ({rmse:.4f})')
    plt.plot(xline, y_cal, linestyle='--', linewidth=1.0, label=f'Calib: y = {a:.2f}x + {b:.2f}')

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    plt.legend(loc='best', frameon=True)
    plt.grid(alpha=0.25, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

class Learner:
  def __init__(self, model_dir, model, dataset, dataset_unsupervised, optimizer, params, dev_dataset=None, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = os.path.join(model_dir,'models')
    self.log_file = os.path.join(model_dir,'train_log')
    self.model = model
    summarize_model(self.model)
    self.dataset = dataset
    self.dataset_unsupervised = dataset_unsupervised
    self.dev_dataset = dev_dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.amp.autocast('cuda', enabled=kwargs.get('fp16', False))
    self.scaler = torch.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.current_epoch = 0
    self.is_master = True
    self.summary_writer = None
    os.makedirs(self.model_dir, exist_ok=True)
    
    # 初始化指标记录器
    self.loss_func = nn.SmoothL1Loss(beta=0.5)  # 使用SmoothL1Loss适配回归任务
    self.train_metrics = Metrics_batch()
    self.loss_recorder = Recorder()
    self.real_loss_recorder = Recorder()
    self.fake_loss_recorder = Recorder()
    
    # 训练统计信息
    self.total_steps_per_epoch = len(dataset) if dataset else 0
    self.total_epochs = getattr(params, 'epochs', 10)
    
    # 注册清理函数
    atexit.register(self.cleanup)
    signal.signal(signal.SIGINT, self.signal_handler)
    signal.signal(signal.SIGTERM, self.signal_handler)

  def cleanup(self):
    """清理资源"""
    if self.summary_writer:
      self.summary_writer.close()
    torch.cuda.empty_cache()

  def signal_handler(self, signum, frame):
    """信号处理器"""
    print(f"Received signal {signum}, cleaning up...")
    self.cleanup()
    sys.exit(0)

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'epoch': getattr(self, 'current_epoch', 0),
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict, strict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'], strict=strict)
    else:
      self.model.load_state_dict(state_dict['model'], strict=strict)

    optimizer_state_dict = state_dict['optimizer']
    current_state_dict = self.optimizer.state_dict()
    for group in optimizer_state_dict['param_groups']:
      for current_group in current_state_dict['param_groups']:
        if group['params'] == current_group['params']:
          current_group.update(group)
        else: 
          print(group)
    self.optimizer.load_state_dict(current_state_dict)
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']
    self.current_epoch = state_dict.get('epoch', 0)

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def list_all_checkpoints(self):
    """列出所有可用的checkpoint"""
    if not os.path.exists(self.model_dir):
      print("No model directory found")
      return []
    
    checkpoint_files = []
    for file in os.listdir(self.model_dir):
      if file.endswith('.pt') and not os.path.islink(os.path.join(self.model_dir, file)):
        checkpoint_files.append(file)
    
    if checkpoint_files:
      print("Available checkpoints:")
      for file in sorted(checkpoint_files):
        file_path = os.path.join(self.model_dir, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        mod_time = time.ctime(os.path.getmtime(file_path))
        print(f"  - {file} ({file_size:.1f}MB, {mod_time})")
    else:
      print("No checkpoints found")
    
    return checkpoint_files

  def find_latest_checkpoint(self):
    """查找最新的checkpoint文件，优先按epoch数字排序"""
    if not os.path.exists(self.model_dir):
      return None
    
    # 查找所有checkpoint文件
    checkpoint_files = []
    epoch_checkpoints = []
    other_checkpoints = []
    
    for file in os.listdir(self.model_dir):
      if file.endswith('.pt') and not os.path.islink(os.path.join(self.model_dir, file)):
        checkpoint_files.append(file)
        
        # 检查是否是epoch格式的checkpoint
        if file.startswith('epoch_') and file.endswith('.pt'):
          try:
            # 提取epoch数字，格式如: epoch_5-12345.pt
            parts = file.replace('.pt', '').split('-')
            epoch_part = parts[0]  # epoch_5
            epoch_num = int(epoch_part.split('_')[1])  # 5
            step_num = int(parts[1]) if len(parts) > 1 else 0  # 12345
            epoch_checkpoints.append((epoch_num, step_num, file))
          except (ValueError, IndexError):
            other_checkpoints.append(file)
        else:
          other_checkpoints.append(file)
    
    if not checkpoint_files:
      return None
    
    # 优先选择最新的epoch checkpoint
    if epoch_checkpoints:
      # 按epoch数字排序，然后按step数字排序
      epoch_checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)
      latest_file = epoch_checkpoints[0][2]
      latest_epoch = epoch_checkpoints[0][0]
      latest_step = epoch_checkpoints[0][1]
      print(f"Found latest epoch checkpoint: {latest_file} (Epoch {latest_epoch}, Step {latest_step})")
    else:
      # 如果没有epoch checkpoint，按修改时间排序
      other_checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.model_dir, x)), reverse=True)
      latest_file = other_checkpoints[0]
      print(f"Found latest checkpoint: {latest_file}")
    
    return latest_file.replace('.pt', '')

  def restore_from_checkpoint(self, filename=None):
    """恢复checkpoint，如果filename为None则自动查找最新的"""
    if filename is None:
      filename = self.find_latest_checkpoint()
      if filename is None:
        print("No checkpoint found, starting from scratch")
        return False
    
    checkpoint_path = f'{self.model_dir}/{filename}.pt'
    try:
      print(f"Loading checkpoint from: {checkpoint_path}")
      checkpoint = torch.load(checkpoint_path, weights_only=True)
      self.load_state_dict(checkpoint, strict=False)
      
      # 打印恢复信息
      restored_epoch = getattr(self, 'current_epoch', 0)
      restored_step = getattr(self, 'step', 0)
      print(f"Successfully restored from checkpoint:")
      print(f"  - Epoch: {restored_epoch}")
      print(f"  - Step: {restored_step}")
      
      return True
    except FileNotFoundError:
      print(f"Checkpoint file not found: {checkpoint_path}")
      return False
    except Exception as e:
      print(f"Error loading checkpoint: {e}")
      return False

  def train(self, max_steps=None):
    device = next(self.model.parameters()).device
    # config logging
    logging.basicConfig(format='%(asctime)s %(filename)s:%(lineno)s %(levelname)s:%(message)s',
            filename=self.log_file, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("params:{}".format(str(self.params)))
    
    # 从params读取epochs设置
    max_epochs = getattr(self.params, 'epochs', 10)
    
    # 确定开始的epoch（如果从checkpoint恢复）
    start_epoch = self.current_epoch
    if start_epoch > 0:
      print(f"Resuming training from epoch {start_epoch + 1}/{max_epochs}")
      logger.info(f"Resuming training from epoch {start_epoch + 1}/{max_epochs}")
      if start_epoch >= max_epochs:
        print(f"Training already completed! Current epoch {start_epoch} >= max epochs {max_epochs}")
        return
    else:
      print(f"Starting training for {max_epochs} epochs...")
      logger.info(f"Training for {max_epochs} epochs")
    
    start = time.time()
    
    for epoch in range(start_epoch, max_epochs):
      self.current_epoch = epoch + 1
      print(f"\n=== Starting Epoch {self.current_epoch}/{max_epochs} ===")
      print(f"Dataset size: {len(self.dataset)} batches")
      logger.info(f"Starting Epoch {self.current_epoch}/{max_epochs}")
      
      epoch_start_time = time.time()
      
      for features in tqdm(self.dataset, desc=f'Epoch {self.current_epoch}/{max_epochs}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

        loss  = self.train_step(features, self.step)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 10 == 0 or self.step % len(self.dataset) == 0:
            # 获取当前batch的指标
            current_metrics = self._get_current_metrics()
            self._write_summary(self.step, features, loss, current_metrics)
            
            # 打印详细指标
            metrics_str = f"step:{self.step}, loss:{loss:.4f}"
            if current_metrics:
              metrics_str += f", rmse:{current_metrics.get('rmse', 0):.4f}"
              metrics_str += f", r2:{current_metrics.get('r2', 0):.4f}"
              metrics_str += f", pearson:{current_metrics.get('pearson', 0):.4f}"
            
            print(f'-----{metrics_str}-----')
            logger.info(metrics_str)
            start = time.time()
            
          
          # 每100步记录平均指标
          if self.step % 100 == 0 and self.step > 0:
            avg_metrics = self.train_metrics.get_mean_metrics()
            avg_loss = self.loss_recorder.average()
            
            print(f'-----Average Metrics (last 100 steps)-----')
            print(f'Avg Loss: {avg_loss:.4f}')
            print(f'Avg RMSE: {avg_metrics["rmse"]:.4f}, R2: {avg_metrics["r2"]:.4f}, Pearson: {avg_metrics["pearson"]:.4f}')
            
            # 记录到tensorboard
            if self.summary_writer:
              self.summary_writer.add_scalar('train_avg/loss', avg_loss, self.step)
              self.summary_writer.add_scalar('train_avg/rmse', avg_metrics["rmse"], self.step)
              self.summary_writer.add_scalar('train_avg/r2', avg_metrics["r2"], self.step)
              self.summary_writer.add_scalar('train_avg/pearson', avg_metrics["pearson"], self.step)
              self.summary_writer.flush()
            
            # 清空记录器
            self.train_metrics.clear()
            self.loss_recorder.clear()
          
          # 每1000步进行一次dev评估
          if self.step % 1000 == 0 and self.step > 0:
            print(f"Running dev evaluation at step {self.step}...")
            rmse, r2, pearson = self.evaluate_dev()
            if rmse is not None:
              print(f'-----Dev Results at step {self.step}: RMSE={rmse:.4f}, R2={r2:.4f}, Pearson={pearson:.4f}-----')
              logger.info(f'Dev evaluation at step {self.step}: RMSE={rmse:.4f}, R2={r2:.4f}, Pearson={pearson:.4f}')
              
              # 记录到tensorboard
              if self.summary_writer:
                self.summary_writer.add_scalar('dev/rmse', rmse, self.step)
                self.summary_writer.add_scalar('dev/r2', r2, self.step)
                self.summary_writer.add_scalar('dev/pearson', pearson, self.step)
                self.summary_writer.flush()
          
        self.step += 1
      
      # 每个epoch结束后的操作
      epoch_time = time.time() - epoch_start_time
      steps_this_epoch = self.step - (epoch * self.total_steps_per_epoch)
      
      print(f"\n=== Epoch {self.current_epoch}/{max_epochs} Summary ===")
      print(f"Time: {epoch_time/60:.2f} minutes")
      print(f"Steps: {steps_this_epoch}")
      print(f"Total steps so far: {self.step}")
      
      logger.info(f"Epoch {self.current_epoch}/{max_epochs} completed in {epoch_time/60:.2f} minutes, {steps_this_epoch} steps")
      
      # 每个epoch结束后保存checkpoint
      if self.is_master:
        try:
          self.save_to_checkpoint(f'epoch_{self.current_epoch}')
          print(f"Checkpoint saved for epoch {self.current_epoch}")
        except Exception as e:
          print(f"Failed to save checkpoint for epoch {self.current_epoch}: {e}")
      
      # 每个epoch结束后进行dev评估
      if self.is_master and self.dev_dataset is not None:
        print(f"Running dev evaluation after epoch {self.current_epoch}...")
        rmse, r2, pearson = self.evaluate_dev()
        if rmse is not None:
          print(f'=== Epoch {self.current_epoch} Dev Results: RMSE={rmse:.4f}, R2={r2:.4f}, Pearson={pearson:.4f} ===')
          logger.info(f'Epoch {self.current_epoch} Dev Results: RMSE={rmse:.4f}, R2={r2:.4f}, Pearson={pearson:.4f}')
          
          # 记录到tensorboard
          if self.summary_writer:
            self.summary_writer.add_scalar('epoch_dev/rmse', rmse, self.current_epoch)
            self.summary_writer.add_scalar('epoch_dev/r2', r2, self.current_epoch)
            self.summary_writer.add_scalar('epoch_dev/pearson', pearson, self.current_epoch)
            self.summary_writer.add_scalar('epoch_info/epoch_time_minutes', epoch_time/60, self.current_epoch)
            self.summary_writer.flush()
    
    print(f"\n=== Training completed after {max_epochs} epochs ===")
    logger.info(f"Training completed after {max_epochs} epochs")

  def train_step(self, features, step):
    self.model.train()
    for param in self.model.parameters():
        param.grad = None

    # 量化数据格式：X和y
    X, y = features
    device = X.device
    
    with self.autocast:
      output = self.model(X)
      loss = self.loss_func(output, y)
    
    # 计算训练指标
    with torch.no_grad():
      # 计算回归指标
      y_np = y.cpu().numpy().flatten()
      pred_np = output.cpu().numpy().flatten()
      
      rmse = np.sqrt(mean_squared_error(y_np, pred_np))
      r2 = r2_score(y_np, pred_np)
      pearson, _ = pearsonr(y_np, pred_np) if len(y_np) > 1 else (0, 0)
      
      # 更新指标记录器
      self.train_metrics.update(y, output)
      
      # 记录损失
      self.loss_recorder.update(loss.item())
      
      # 存储当前batch的指标用于显示
      self.current_batch_metrics = {
        'rmse': rmse,
        'r2': r2,
        'pearson': pearson
      }

    if self.is_master and step % 50 == 0:
        with torch.no_grad():
            pred_std = output.std().item()
            target_std = y.std().item()
            if self.summary_writer:
                self.summary_writer.add_scalar('train/pred_std', pred_std, step)
                self.summary_writer.add_scalar('train/target_std', target_std, step)
                self.summary_writer.add_histogram('train/pred_hist', output, step)
            else:
                print(f'[step {step}] pred_std={pred_std:.3f}  target_std={target_std:.3f}')
                
    if torch.isnan(loss) or torch.isinf(loss):
      print("Loss is NaN or Inf")
      print(f"loss: {loss}")

    assert not torch.isnan(loss), "NaN detected in loss!"
    assert output.shape[0] == y.shape[0], f"Batch mismatch: output {output.shape}, y {y.shape}"
    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    
    return loss

  def _get_current_metrics(self):
    """获取当前batch的指标"""
    return getattr(self, 'current_batch_metrics', None)

  def _write_summary(self, step, features, loss, metrics=None):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_scalar('train/loss', loss, step)
    
    # 记录当前batch的指标
    if metrics:
      writer.add_scalar('train_batch/rmse', metrics.get('rmse', 0), step)
      writer.add_scalar('train_batch/r2', metrics.get('r2', 0), step)
      writer.add_scalar('train_batch/pearson', metrics.get('pearson', 0), step)
    
    writer.flush()
    self.summary_writer = writer

  def predict(self, features):
    with torch.no_grad():
      X, y = features
      output = self.model(X)
      return output.cpu().numpy().ravel()

  def evaluate_dev(self):
    """在开发集上进行评估，计算RMSE、R2和Pearson相关系数"""
    if self.dev_dataset is None:
      print("No dev dataset provided, skipping evaluation")
      return None, None, None
    
    self.model.eval()
    all_preds = []
    all_targets = []
    
    device = next(self.model.parameters()).device
    
    with torch.no_grad():
      for features in tqdm(self.dev_dataset, desc="Evaluating on dev set"):
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        
        X, y = features
        output = self.model(X)
        
        all_preds.extend(output.cpu().numpy().flatten())
        all_targets.extend(y.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    pearson, _ = pearsonr(all_targets, all_preds) if len(all_targets) > 1 else (0, 0)
    
    return rmse, r2, pearson

def _train_impl(replica_id, model, dataset, dataset_unsupervised, args, params, dev_dataset=None):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=1e-4)
  learner = Learner(args.model_dir, model, dataset, dataset_unsupervised, opt, params, dev_dataset=dev_dataset, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  
  # 自动恢复最新的checkpoint
  if learner.is_master:
    learner.restore_from_checkpoint()  # 自动查找最新checkpoint
  
  # 使用max_steps或epochs，优先使用max_steps
  max_steps = getattr(args, 'max_steps', None)
  learner.train(max_steps=max_steps)

def train_distributed_torchrun(replica_id, args, params):
  dataset = from_train_list(args.train_list[0], args.audio_root, params, is_distributed=True)
  
  # 创建dev数据集
  dev_dataset = None
  if hasattr(args, 'dev_list') and args.dev_list:
    dev_dataset = from_train_list(args.dev_list, args.audio_root, params, is_distributed=False)
  
  # 自动获取全局设备信息
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  
  # 初始化模型
  model = create_model(params).to(device)
  
  # 初始化DDP
  model = DistributedDataParallel(
    model,
    device_ids=[replica_id],
    output_device=replica_id,
    find_unused_parameters=True  # 根据实际情况调整
  )
  _train_impl(replica_id, model, dataset, None, args, params, dev_dataset=dev_dataset)

def test_learner_training_step():
    """测试Learner的训练步骤"""
    print("测试Learner训练步骤...")
    
    try:
        from params import params
        from dataset import load_data, split_time_series_data, StockDataset
        from torch.utils.data import DataLoader
        
        # 启用测试模式
        params.test_mode = True
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型和优化器
        model = create_model(params).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=1e-4)
        
        # 创建小数据集
        X, y, time_stamps = load_data()
        (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
        train_dataset = StockDataset(X_train[:100], y_train[:100])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # 创建Learner实例
        learner = Learner('./test_output', model, train_loader, None, optimizer, params)
        
        print("✓ Learner创建成功")
        
        # 测试一个训练步骤
        for batch in train_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(X_batch)
            loss = learner.loss_func(output, y_batch)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            print(f"✓ 训练步骤成功")
            print(f"  损失值: {loss.item():.4f}")
            print(f"  输出形状: {output.shape}")
            print(f"  目标形状: {y_batch.shape}")
            break
        
        return True
    except Exception as e:
        print(f"✗ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_learner_evaluation():
    """测试Learner的评估功能"""
    print("测试Learner评估功能...")
    
    try:
        from params import params
        from dataset import load_data, split_time_series_data, StockDataset
        from torch.utils.data import DataLoader
        
        # 启用测试模式
        params.test_mode = True
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        model = create_model(params).to(device)
        
        # 创建小数据集
        X, y, time_stamps = load_data()
        (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
        val_dataset = StockDataset(X_val[:50], y_val[:50])
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # 创建Learner实例
        learner = Learner('./test_output', model, None, None, None, params, dev_dataset=val_loader)
        
        print("✓ Learner评估实例创建成功")
        
        # 测试评估
        rmse, r2, pearson = learner.evaluate_dev()
        
        if rmse is not None:
            print(f"✓ 评估功能正常")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R2: {r2:.4f}")
            print(f"  Pearson: {pearson:.4f}")
        else:
            print("✗ 评估功能异常")
        
        return True
    except Exception as e:
        print(f"✗ 评估功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_learner_metrics():
    """测试Learner的指标计算"""
    print("测试Learner指标计算...")
    
    try:
        from params import params
        from dataset import load_data, split_time_series_data, StockDataset
        from torch.utils.data import DataLoader
        
        # 启用测试模式
        params.test_mode = True
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型和优化器
        model = create_model(params).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=1e-4)
        
        # 创建小数据集
        X, y, time_stamps = load_data()
        (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
        train_dataset = StockDataset(X_train[:100], y_train[:100])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # 创建Learner实例
        learner = Learner('./test_output', model, train_loader, None, optimizer, params)
        
        print("✓ Learner指标实例创建成功")
        
        # 测试指标计算
        for batch in train_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向传播
            output = model(X_batch)
            
            # 计算指标
            y_np = y_batch.cpu().numpy().flatten()
            pred_np = output.cpu().numpy().flatten()
            
            from sklearn.metrics import mean_squared_error, r2_score
            from scipy.stats import pearsonr
            
            rmse = np.sqrt(mean_squared_error(y_np, pred_np))
            r2 = r2_score(y_np, pred_np)
            pearson, _ = pearsonr(y_np, pred_np) if len(y_np) > 1 else (0, 0)
            
            print(f"✓ 指标计算成功")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R2: {r2:.4f}")
            print(f"  Pearson: {pearson:.4f}")
            break
        
        return True
    except Exception as e:
        print(f"✗ 指标计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__=='__main__':
  try:
    from params import params
    from dataset import from_train_list, load_data, split_time_series_data
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    print("=" * 60)
    print("Learner系统测试")
    print("=" * 60)
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    device = torch.device('cuda')
    print(f"使用设备: {device}")
    print(f"训练配置: {params.epochs} epochs, batch_size={params.batch_size}, lr={params.learning_rate}")

    # 启用测试模式
    params.test_mode = True

    # 加载量化数据
    X, y, time_stamps = load_data()
    (X_train, X_val, X_test), (y_train, y_val, y_test), (ts_train, ts_val, ts_test) = split_time_series_data(X, y, time_stamps)
    
    # 创建数据集
    train_dataset = StockDataset(X_train[:100], y_train[:100])
    dev_dataset = StockDataset(X_val[:50], y_val[:50])
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    model = create_model(params)
    opt = torch.optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=1e-4)
    learner = Learner('./test_output', model, train_loader, None, opt, params, dev_dataset=dev_loader)

    print("✓ 基础组件创建成功")

    # 测试单个batch
    print("\n=== 测试单个batch ===")
    for features in train_loader:
      X_batch, y_batch = features
      print(f"X shape: {X_batch.shape}")
      print(f"y shape: {y_batch.shape}")
      
      loss = learner.train_step(features, 10000)
      current_metrics = learner._get_current_metrics()
      learner._write_summary(10000, features, loss, current_metrics)
      
      print(f"Loss: {loss:.4f}")
      if current_metrics:
        print(f"Metrics: RMSE={current_metrics['rmse']:.4f}, R2={current_metrics['r2']:.4f}, Pearson={current_metrics['pearson']:.4f}")
      break
    
    # 测试评估功能
    print("\n=== 测试评估功能 ===")
    rmse, r2, pearson = learner.evaluate_dev()
    if rmse is not None:
        print(f"评估结果: RMSE={rmse:.4f}, R2={r2:.4f}, Pearson={pearson:.4f}")
    else:
        print("评估功能测试失败")
    
    print("\n" + "=" * 60)
    print("Learner系统测试完成！")
    print("=" * 60)
      
  except KeyboardInterrupt:
    print("训练被用户中断")
  except Exception as e:
    print(f"训练过程中出现错误: {e}")
    import traceback
    traceback.print_exc()
  finally:
    # 确保清理资源
    torch.cuda.empty_cache()
    print("资源清理完成")