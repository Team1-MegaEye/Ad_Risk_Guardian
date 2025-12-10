# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 10:19:18 2025

@author: deoha
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 09:58:38 2025

@author: deoha
"""

# =============================================================
# 0. Environment Setup
# =============================================================

import os
import pickle
from pathlib import Path
import random
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# =============================================================
# 1. Path Configs (EDIT THIS FOR YOUR FOLDER!!)
# =============================================================


# ★★★ 스파이더(로컬)용 예시 경로 (원하는 폴더로 수정하세요) ★★★
DATA_ROOT = Path(r"C:\user\Processed_data")  # 입력 경로 설정

TRAIN_ROOT = DATA_ROOT / "train"
TEST_ROOT  = DATA_ROOT / "test"
USE_SEQ_NORM = True        # seq_norm.pt 우선 사용
SEQ_LEN = 10               # 시퀀스 길이
IMG_SIZE = 256             # EfficientNet 입력 이미지 크기
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42

MODEL_SAVE_PATH = DATA_ROOT / "efficientnet_lstm_best_freeze.pth"
EXCEL_LOG_PATH = DATA_ROOT / "training_log_efficientnet_lstm_freeze_n.xlsx"

print("DATA_ROOT =", DATA_ROOT)

# =============================================================
# 2. Utility Functions
# =============================================================

def list_frame_paths(seq_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = [p for p in seq_dir.iterdir() if p.suffix.lower() in exts]
    frames.sort()
    return frames


def load_sequence(seq_dir: Path,
                  use_seq_norm: bool = True,
                  seq_len: int = SEQ_LEN) -> torch.Tensor:

    seq_norm_path = seq_dir / "seq_norm.pt"

    # ---- Case 1: 정규화된 텐서가 존재하면 그걸 사용 ----
    if use_seq_norm and seq_norm_path.exists():
        seq = torch.load(seq_norm_path, map_location="cpu")

    # ---- Case 2: jpg 프레임 로딩 후 normalize ----
    else:
        frame_paths = list_frame_paths(seq_dir)
        if not frame_paths:
            raise RuntimeError(f"No frames in {seq_dir}")

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            t = TF.to_tensor(img)
            t = normalize(t)
            frames.append(t)

        seq = torch.stack(frames, dim=0)

    # ---- Length correction ----
    T, C, H, W = seq.shape

    if T >= seq_len:
        seq = seq[:seq_len]
    else:
        last = seq[-1:].expand(seq_len - T, C, H, W)
        seq = torch.cat([seq, last], dim=0)

    # ---- Resize to 112x112 ----
    seq = F_torch.interpolate(seq, size=(IMG_SIZE, IMG_SIZE),
                              mode="bilinear", align_corners=False)

    return seq

# =============================================================
# 3. Case1 Dataset Class
# =============================================================

class Case1SequenceDataset(Dataset):
    def __init__(self,
                 seq_dirs: List[Path],
                 labels: List[int],
                 use_seq_norm: bool = True,
                 seq_len: int = SEQ_LEN):

        self.seq_dirs = seq_dirs
        self.labels = labels
        self.use_seq_norm = use_seq_norm
        self.seq_len = seq_len

    def __len__(self):
        return len(self.seq_dirs)

    def __getitem__(self, idx):
        seq_dir = self.seq_dirs[idx]
        label = self.labels[idx]
        seq = load_sequence(seq_dir,
                            use_seq_norm=self.use_seq_norm,
                            seq_len=self.seq_len)
        # (T, C, H, W) → (1, T, C, H, W)는 DataLoader에서 batch_dim 추가되므로 여기서는 그대로
        return seq, label

# =============================================================
# 4. EfficientNetB0 + LSTM
# =============================================================

class EfficientNetB0LSTM(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 latent_dim: int = 1280,
                 lstm_layers: int = 1,
                 hidden_dim: int = 2048,
                 bidirectional: bool = False,
                 dropout: float = 0.4):
        super().__init__()

        # torchvision 버전에 따라 pretrained 인자 경고가 뜰 수 있음
        # 최신 버전에서는 아래처럼 사용하는 것을 권장:
        #   weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        #   base = models.efficientnet_b0(weights=weights)
        base = models.efficientnet_b0(pretrained=True)

        # EfficientNet의 features 부분만 사용 (classifier 제거)
        self.feature_extractor = base.features  # (B*T, 1280, H/32, W/32)

        # ---------------------------------------------------------
        # CNN freezing + 마지막 블록만 unfreeze
        # ---------------------------------------------------------
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # # ★ 마지막 블록만 다시 학습 가능하게 설정
        # if isinstance(self.feature_extractor, nn.Sequential):
        #     last_block = self.feature_extractor[-1]
        # else:
        #     # 혹시 Sequential이 아닐 경우 대비
        #     last_block = list(self.feature_extractor.children())[-1]

        # for param in last_block.parameters():
        #     param.requires_grad = True

        # ---------------------------------------------------------
        # LSTM
        # ---------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=latent_dim,      # EfficientNet-B0 output = 1280-d
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # ---------------------------------------------------------
        # FC & Dropout
        # ---------------------------------------------------------
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # → (B*T, 1280, 1, 1)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        # CNN reshape
        x = x.view(B * T, C, H, W)

        # CNN forward
        fmap = self.feature_extractor(x)  # (B*T, 1280, h, w)

        # Adaptive pooling
        x = self.avgpool(fmap)            # (B*T, 1280, 1, 1)
        x = x.view(B, T, -1)              # (B, T, 1280)

        # LSTM
        x_lstm, _ = self.lstm(x)          # (B, T, hidden_dim)

        # Sequence average
        x = torch.mean(x_lstm, dim=1)     # (B, hidden_dim)

        # FC
        x = self.fc(x)
        x = self.dropout(x)

        return x

# =============================================================
# 5. Collect train sequences
# =============================================================

def collect_sequences(root: Path):
    real_root = root / "real"
    fake_root = root / "fake"

    real_dirs = [d for d in real_root.iterdir() if d.is_dir()] if real_root.exists() else []
    fake_dirs = [d for d in fake_root.iterdir() if d.is_dir()] if fake_root.exists() else []

    real_dirs.sort()
    fake_dirs.sort()

    seq_dirs = real_dirs + fake_dirs
    labels   = [1] * len(real_dirs) + [0] * len(fake_dirs)

    print(f"[INFO] {root.name}/real: {len(real_dirs)}")
    print(f"[INFO] {root.name}/fake: {len(fake_dirs)}")
    print(f"[INFO] total: {len(seq_dirs)}")

    return seq_dirs, labels

# =============================================================
# 6. Train / Eval
# =============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for seqs, labels in tqdm(loader, desc="Train", leave=False):
        # seqs: (B, T, C, H, W)
        seqs = seqs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        labels_float = labels.float().unsqueeze(1)  # (B,) → (B,1)
        outputs = model(seqs)                       # (B,1)
        loss = criterion(outputs, labels_float)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * seqs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).long()

        correct += (preds == labels_float.long()).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for seqs, labels in tqdm(loader, desc="Valid", leave=False):
            seqs = seqs.to(device)
            labels = labels.to(device)

            labels_float = labels.float().unsqueeze(1)   # (B,) → (B,1)

            outputs = model(seqs)
            loss = criterion(outputs, labels_float)

            running_loss += loss.item() * seqs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).long()   # (B,1)

            correct += (preds == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().squeeze(1).tolist())

    return running_loss / total, correct / total, np.array(y_true), np.array(y_pred)

# =============================================================
# 7. Train Loop (Main)
# =============================================================

def main():
    # Reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # ---------------------
    # 1. Collect dataset
    # ---------------------
    train_dirs, train_labels = collect_sequences(TRAIN_ROOT)
    val_dirs,   val_labels   = collect_sequences(TEST_ROOT)

    print(f"[INFO] Train: {len(train_dirs)} | Val(Test): {len(val_dirs)}")

    train_dataset = Case1SequenceDataset(train_dirs, train_labels,
                                         use_seq_norm=USE_SEQ_NORM, seq_len=SEQ_LEN)
    val_dataset   = Case1SequenceDataset(val_dirs,   val_labels,
                                         use_seq_norm=USE_SEQ_NORM, seq_len=SEQ_LEN)

    # ⚠ Spyder/Windows에서 num_workers>0 이슈가 있으면 0으로 바꾸세요.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ---------------------
    # 2. Model / Loss / Optimizer
    # ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device:", device)

    model = EfficientNetB0LSTM().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # freeze 안 된 것만
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs,  val_accs  = [], []

    last_y_true, last_y_pred = None, None

    # ---------------------
    # 3. Training Loop
    # ---------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")

        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = eval_one_epoch(model, val_loader,
                                                           criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        last_y_true, last_y_pred = y_true, y_pred

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"[Epoch {epoch}] Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f">>>> Best model updated! val_acc={best_val_acc:.4f}")

    print("\n[Training Finished]")
    print("Best Validation Accuracy =", best_val_acc)

    # ---------------------
    # 4. Plot Loss Curve
    # ---------------------
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Train / Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # ---------------------
    # 5. Confusion Matrix & Report
    # ---------------------
    cm = confusion_matrix(last_y_true, last_y_pred)
    print("Confusion Matrix:\n", cm)

    report_str = classification_report(last_y_true, last_y_pred, digits=4)
    print("\nClassification Report:\n", report_str)

    # ---------------------
    # 6. Excel 로그 저장
    # ---------------------
    history_df = pd.DataFrame({
        "epoch":      list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "train_acc":  train_accs,
        "val_loss":   val_losses,
        "val_acc":    val_accs,
    })

    cm_df = pd.DataFrame(
        cm,
        index=["true_fake(0)", "true_real(1)"],
        columns=["pred_fake(0)", "pred_real(1)"]
    )

    report_dict = classification_report(
        last_y_true,
        last_y_pred,
        output_dict=True,
        digits=4
    )
    report_df = pd.DataFrame(report_dict).T

    excel_path_str = str(EXCEL_LOG_PATH)
    os.makedirs(EXCEL_LOG_PATH.parent, exist_ok=True)
    with pd.ExcelWriter(excel_path_str, engine="openpyxl") as writer:
        history_df.to_excel(writer, sheet_name="history", index=False)
        cm_df.to_excel(writer, sheet_name="confusion_matrix")
        report_df.to_excel(writer, sheet_name="classification_report")

    print(f"\n[LOG] Training log saved to: {excel_path_str}")

# =============================================================
# Entry Point (Colab & Spyder 공통)
# =============================================================
if __name__ == "__main__":
    main()
