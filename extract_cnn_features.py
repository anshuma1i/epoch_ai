import argparse
import os
import copy
import json
import logging
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from shapely import wkb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. Configuration & Setup
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASSES = [
    'Birds of Prey', 'Clutter', 'Cormorants', 'Ducks', 'Geese',
    'Gulls', 'Pigeons', 'Songbirds', 'Waders'
]

def setup_args():
    parser = argparse.ArgumentParser(description='Extract 1D CNN features for radar tracks.')
    parser.add_argument('--train-csv', type=str, default='dataset/train.csv')
    parser.add_argument('--test-csv', type=str, default='dataset/test.csv')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs per fold.')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Lowered Learning rate.')
    parser.add_argument('--n-splits', type=int, default=10, help='Number of CV splits.')
    parser.add_argument('--max-len', type=int, default=128, help='Max sequence length for truncation.')
    parser.add_argument('--embedding-dim', type=int, default=32, help='Embedding dim size.')
    parser.add_argument('--rotate-aug', action='store_true', help='Apply random rotation augmentation to dx,dy.')
    parser.add_argument('--out-train', type=str, default='dataset/train_cnn_features.csv')
    parser.add_argument('--out-test', type=str, default='dataset/test_cnn_features.csv')
    parser.add_argument('--fast-dev-run', action='store_true', help='Run 1 epoch on 1 split with tiny data.')
    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. Data Parsing & Dataset
# ─────────────────────────────────────────────
def parse_trajectory(hex_str: str) -> List[Tuple]:
    if not isinstance(hex_str, str) or len(hex_str) == 0:
        return []
    try:
        geom = wkb.loads(bytes.fromhex(hex_str) if not hex_str.startswith('\x01') else hex_str, hex=True)
        if geom.geom_type == 'LineString':
            return list(geom.coords)
        elif geom.geom_type == 'Point':
            return [geom.coords[0]]
        elif geom.geom_type in ('MultiPoint', 'GeometryCollection'):
            return [g.coords[0] for g in geom.geoms]
    except Exception:
        pass
    return []

def extract_sequence(row: pd.Series, max_len: int) -> np.ndarray:
    coords = parse_trajectory(row['trajectory'])
    n = len(coords)
    if n == 0:
        return np.zeros((max_len, 6), dtype=np.float32)
        
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    alts = [c[2] for c in coords] if len(coords[0]) > 2 else [0.0] * n
    rcs  = [c[3] for c in coords] if len(coords[0]) > 3 else [0.0] * n

    times = row.get('trajectory_time', '')
    t_list = []
    if isinstance(times, str) and times.strip():
        try:
            t_list = [float(x) for x in times.strip('[]').split(',')]
        except ValueError:
            pass
    elif isinstance(times, (list, np.ndarray)):
        t_list = list(times)
        
    if len(t_list) != n:
        t_list = [0.0] * n

    dx = np.diff(lons, prepend=lons[0]) * 71000
    dy = np.diff(lats, prepend=lats[0]) * 111000
    dz = np.diff(alts, prepend=alts[0])
    dt = np.diff(t_list, prepend=0.0)
    dt = np.where(dt == 0, 1.0, dt)
    
    speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt
    seq = np.column_stack([dx, dy, dz, dt, speed, rcs]).astype(np.float32)
    
    # Strictly pad or truncate to max_len
    L = seq.shape[0]
    if L > max_len:
        seq = seq[:max_len]
    elif L < max_len:
        pad = np.zeros((max_len - L, 6), dtype=np.float32)
        seq = np.vstack([seq, pad])
        
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    return seq

class RadarTrajectoryDataset:
    def __init__(self, df: pd.DataFrame, scaler=None, fit_scaler=False, max_len=128):
        self.df = df
        self.max_len = max_len
        self.track_ids = df.index.values
        
        if 'bird_group' in df.columns:
            self.y_str = df['bird_group'].values
            self.y_idx = df['bird_group'].map({c: i for i, c in enumerate(CLASSES)}).values
        else:
            self.y_str = np.array(['Gulls'] * len(df))
            self.y_idx = np.zeros(len(df), dtype=np.int64)
            
        logger.info(f"Parsing {len(df)} trajectories...")
        raw_sequences = [extract_sequence(row, self.max_len) for _, row in df.iterrows()]
        
        if fit_scaler:
            self.scaler = StandardScaler()
            flat_data = np.vstack(raw_sequences)
            self.scaler.fit(flat_data)
        else:
            self.scaler = scaler
            
        self.sequences = [self.scaler.transform(seq).astype(np.float32) for seq in raw_sequences]
        
    def __len__(self): return len(self.df)

class SimpleSeqDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, ids):
        self.x = x_tensor
        self.y = y_tensor
        self.ids = ids
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx], self.ids[idx]


# ─────────────────────────────────────────────
# 3. Model Architecture
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TrajectoryCNN(nn.Module):
    def __init__(self, in_channels=6, embedding_dim=32, num_classes=9, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Linear(128 * 2, embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        f = self.features(x)
        p_avg = self.avg_pool(f).squeeze(-1)
        p_max = self.max_pool(f).squeeze(-1)
        
        p = torch.cat([p_avg, p_max], dim=1)
        p = self.dropout(p)
        
        emb = self.embed(p)
        logits = self.head(emb)
        return emb, logits


# ─────────────────────────────────────────────
# 4. Training Plot Utilities
# ─────────────────────────────────────────────
def save_loss_plot(history, out_path, title):
    plt.figure(figsize=(10, 5))
    for fold_idx in range(len(history)):
        train_loss = history[fold_idx]['train_loss']
        val_loss = history[fold_idx]['val_loss']
        epochs = range(1, len(train_loss) + 1)
        
        plt.plot(epochs, train_loss, color='blue', alpha=0.3, label='Train' if fold_idx==0 else "")
        plt.plot(epochs, val_loss, color='orange', alpha=0.3, label='Val' if fold_idx==0 else "")
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()


# ─────────────────────────────────────────────
# 5. Main Training Loop
# ─────────────────────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    train_df = pd.read_csv(args.train_csv).set_index("track_id")
    test_df = pd.read_csv(args.test_csv).set_index("track_id")
    
    if args.fast_dev_run:
        train_df = train_df.iloc[:500]
        test_df = test_df.iloc[:100]
        args.epochs = 2
        args.n_splits = 2
        
    logger.info("Initializing Standard Scaler on full train data...")
    full_train_dataset = RadarTrajectoryDataset(train_df, fit_scaler=True, max_len=args.max_len)
    test_dataset = RadarTrajectoryDataset(test_df, scaler=full_train_dataset.scaler, fit_scaler=False, max_len=args.max_len)
    
    X_test_seq = np.array(test_dataset.sequences)
    tensor_x_test = torch.tensor(X_test_seq, dtype=torch.float32).transpose(1, 2)
    test_loader = DataLoader(
        SimpleSeqDataset(tensor_x_test, torch.zeros(len(test_dataset), dtype=torch.int64), test_dataset.track_ids),
        batch_size=args.batch_size, shuffle=False
    )

    oof_emb = np.zeros((len(train_df), args.embedding_dim), dtype=np.float32)
    test_emb = np.zeros((len(test_df), args.embedding_dim), dtype=np.float32)

    fold_indices = {t_id: i for i, t_id in enumerate(train_df.index)}
    history = []

    groups = train_df['primary_observation_id']
    y_full = train_df['bird_group']
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    split = list(cv.split(train_df, y_full, groups))

    for i, (train_idx, val_idx) in enumerate(split):
        logger.info(f"\n{'='*40}\nFold {i+1}/{args.n_splits}\n{'='*40}")
        
        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]
        
        ds_train = RadarTrajectoryDataset(train_fold_df, scaler=full_train_dataset.scaler, max_len=args.max_len)
        ds_val = RadarTrajectoryDataset(val_fold_df, scaler=full_train_dataset.scaler, max_len=args.max_len)
        
        # Validation setup
        X_va_seq = np.array(ds_val.sequences)
        tensor_x_val = torch.tensor(X_va_seq, dtype=torch.float32).transpose(1, 2)
        tensor_y_val = torch.tensor(ds_val.y_idx, dtype=torch.int64)
        loader_val = DataLoader(SimpleSeqDataset(tensor_x_val, tensor_y_val, ds_val.track_ids), batch_size=args.batch_size, shuffle=False)
        
        # Train Resampling (Undersample Gulls, Oversample Rest)
        X_tr_seq = np.array(ds_train.sequences)
        N, L, C = X_tr_seq.shape
        X_tr_flat = X_tr_seq.reshape(N, L * C)
        y_tr_str = ds_train.y_str
        
        counts = pd.Series(y_tr_str).value_counts()
        # Increased Gull undersampling limit to give model more data to learn from
        target_cnt = 600
        
        # 1. Undersample majority classes down to 600
        dict_under = {}
        for c, cnt in counts.items():
            if cnt > target_cnt:
                dict_under[c] = target_cnt
                
        logger.info(f"RandomUnderSampler strategy: {dict_under}")
        rus = RandomUnderSampler(sampling_strategy=dict_under, random_state=42)
        X_res, y_res = rus.fit_resample(X_tr_flat, y_tr_str)
        
        # 3. Final sequences and labels
        seq_res = X_res.reshape(-1, L, C)
        tensor_x_tr = torch.tensor(seq_res, dtype=torch.float32).transpose(1, 2)
        
        label_map = {c: idx for idx, c in enumerate(CLASSES)}
        y_int = np.array([label_map[y] for y in y_res])
        tensor_y_tr = torch.tensor(y_int, dtype=torch.int64)
        
        # Calculate Class Weights to handle the remaining imbalance
        unique_classes = np.unique(y_int)
        weights = compute_class_weight('balanced', classes=unique_classes, y=y_int)
        alpha_tensor = torch.zeros(len(CLASSES)).to(device)
        for idx, val in zip(unique_classes, weights):
            alpha_tensor[idx] = val

        loader_train = DataLoader(
            SimpleSeqDataset(tensor_x_tr, tensor_y_tr, np.arange(len(tensor_y_tr))), 
            batch_size=args.batch_size, shuffle=True
        )

        logger.info(f"Training 9-Class CNN. Dataset size: {len(y_int)}")
        model = TrajectoryCNN(in_channels=6, embedding_dim=args.embedding_dim, num_classes=len(CLASSES), dropout=0.5).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        
        # Use Focal Loss with alpha weights
        criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)
        
        hist = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for seqs, y_batch, _ in loader_train:
                seqs, y_batch = seqs.to(device), y_batch.to(device)
                
                # Input Jittering: Add tiny gaussian noise
                if random.random() > 0.5:
                    seqs = seqs + torch.randn_like(seqs) * 0.01
                
                # Rotation Augmentation: Rotate the 2D plane (dx, dy)
                if args.rotate_aug and random.random() > 0.5:
                    angle = random.uniform(0, 2 * np.pi)
                    c, s = np.cos(angle), np.sin(angle)
                    
                    # Channel 0: dx, Channel 1: dy
                    dx = seqs[:, 0, :].clone()
                    dy = seqs[:, 1, :].clone()
                    seqs[:, 0, :] = dx * c - dy * s
                    seqs[:, 1, :] = dx * s + dy * c

                optimizer.zero_grad()
                _, logits = model(seqs)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * seqs.size(0)
            train_loss /= len(tensor_x_tr)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seqs, y_batch, _ in loader_val:
                    seqs, y_batch = seqs.to(device), y_batch.to(device)
                    _, logits = model(seqs)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item() * seqs.size(0)
            val_loss /= len(tensor_x_val)
            
            scheduler.step()
            
            hist['train_loss'].append(train_loss)
            hist['val_loss'].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch == args.epochs - 1 or args.fast_dev_run or patience_counter >= args.patience:
                logger.info(f"  Epoch {epoch+1}/{args.epochs} - Tr Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")
                if patience_counter >= args.patience:
                    logger.info(f"  Early stopping triggered at epoch {epoch+1}")
                    break

        history.append(hist)
        model.load_state_dict(best_model_state)
        
        # Extract OOF and Test embeddings
        model.eval()
        with torch.no_grad():
            for seqs, _, t_ids in loader_val:
                emb, _ = model(seqs.to(device))
                for j, e in enumerate(emb.cpu().numpy()):
                    t_id_val = t_ids[j]
                    if isinstance(t_id_val, torch.Tensor):
                        t_id_val = t_id_val.item()
                    oof_emb[fold_indices[t_id_val]] = e

            for seqs, _, t_ids in test_loader:
                emb, _ = model(seqs.to(device))
                for j, e in enumerate(emb.cpu().numpy()):
                    t_id_val = t_ids[j]
                    if isinstance(t_id_val, torch.Tensor):
                        t_id_val = t_id_val.item()
                    idx_test = test_df.index.get_loc(t_id_val)
                    test_emb[idx_test] += e
                    
        if args.fast_dev_run and i == 1:
            break

    n_folds_completed = args.n_splits if not args.fast_dev_run else 2
    test_emb /= n_folds_completed

    logger.info("Constructing output DataFrames...")
    train_feat_df = pd.DataFrame(index=train_df.index)
    test_feat_df = pd.DataFrame(index=test_df.index)
    
    for i in range(args.embedding_dim):
        train_feat_df[f'cnn_emb_{i}'] = oof_emb[:, i]
        test_feat_df[f'cnn_emb_{i}'] = test_emb[:, i]

    train_feat_df.to_csv(args.out_train)
    test_feat_df.to_csv(args.out_test)
    logger.info(f"Saved {args.out_train} and {args.out_test}")
    
    logger.info("Saving training loss plots...")
    save_loss_plot(history, 'dataset/cnn_loss.png', 'Single-Stage 9-Class CNN Training Curve')

if __name__ == '__main__':
    args = setup_args()
    main(args)
