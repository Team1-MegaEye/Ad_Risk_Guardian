# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 01:42:42 2025

@author: deoha
"""

"""
ResNeXt50 + LSTM 기반 딥페이크 시퀀스 이진 분류 스크립트

- 입력: (B, T, C, H, W) 형태의 얼굴 시퀀스 (T: 프레임 수)
- Backbone: ResNeXt50-32x4d (ImageNet pretrain, 기본 freeze)
- Temporal 모듈: LSTM (sequence-level representation 학습)
- 출력: 시퀀스 단위 real(1) / fake(0) 이진 분류

구성
----
1) Path 및 하이퍼파라미터 설정
2) 시퀀스 로딩 유틸 함수 (seq_norm 텐서 / 이미지 프레임)
3) Dataset 클래스
4) ResNeXt + LSTM 모델 정의
5) 시퀀스 경로 및 레이블 수집 함수
6) 1 epoch 학습/검증 함수
7) 메인 루프 (학습, Early Stopping, 로그/리포트/엑셀 저장)
"""

import os
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

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# =============================================================
# 1. Path Configs (EDIT THIS FOR YOUR FOLDER!!)
# =============================================================

DATA_ROOT = Path(r"C:\user\Processed_data")  # 입력 경로 설정

TRAIN_ROOT = DATA_ROOT / "train"
TEST_ROOT  = DATA_ROOT / "test"
USE_SEQ_NORM = True        # seq_norm.pt 우선 사용
SEQ_LEN = 10               # 시퀀스 길이
IMG_SIZE = 256           # ResNeXt 입력 이미지 크기
BATCH_SIZE = 16 
NUM_EPOCHS = 30  
LR = 1e-4  
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42

# Early Stopping 설정 (검증 정확도 기준)
PATIENCE   = 3            # 개선 없는 epoch가 PATIENCE 이상이면 종료
MIN_DELTA  = 1e-4            # val_acc가 이 값 이상 증가해야 '개선'으로 간주

MODEL_SAVE_PATH = DATA_ROOT / "resnext_lstm_best.pth"
EXCEL_LOG_PATH = DATA_ROOT / "training_log_resnext_lstm.xlsx"

print("DATA_ROOT =", DATA_ROOT)


# =============================================================
# 2. Utility Functions
# =============================================================

def list_frame_paths(seq_dir: Path) -> List[Path]:
    """
    시퀀스 디렉터리 내의 이미지 프레임 경로를 정렬된 리스트로 반환.

    Args:
        seq_dir (Path): 프레임 이미지(.jpg, .png 등)가 저장된 폴더 경로

    Returns:
        List[Path]: 정렬된 이미지 경로 리스트
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = [p for p in seq_dir.iterdir() if p.suffix.lower() in exts]
    frames.sort()
    return frames


def load_sequence(
    seq_dir: Path,
    use_seq_norm: bool = True,
    seq_len: int = SEQ_LEN
) -> torch.Tensor:
    """
    한 시퀀스 디렉터리에서 (T, C, H, W) 텐서를 로드하고,
    길이 보정 및 리사이즈를 수행.

    1) seq_norm 텐서가 있으면 우선 사용
    2) 없으면 이미지 프레임을 로드 후 normalize

    Args:
        seq_dir (Path): 시퀀스 폴더 경로
        use_seq_norm (bool): True이면 seq_norm 텐서를 우선 사용
        seq_len (int): 최종적으로 맞추고자 하는 시퀀스 길이 T

    Returns:
        torch.Tensor: shape (T, 3, IMG_SIZE, IMG_SIZE)
    """
    # ※ 프로젝트에서 저장해둔 seq_norm 파일명 (필요 시 수정)
    seq_norm_path = seq_dir / "seq_norm.pt"

    # ---- Case 1: 정규화된 텐서가 존재하면 그걸 사용 ----
    if use_seq_norm and seq_norm_path.exists():
        seq = torch.load(seq_norm_path, map_location="cpu")

    # ---- Case 2: jpg 프레임 로딩 후 normalize ----
    else:
        frame_paths = list_frame_paths(seq_dir)
        if not frame_paths:
            raise RuntimeError(f"No frames in {seq_dir}")

        # ImageNet 통계 (ResNeXt 사전학습과 동일하게 맞추기)
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            t = TF.to_tensor(img)   # [0,1] 범위 (C,H,W)
            t = normalize(t)
            frames.append(t)

        seq = torch.stack(frames, dim=0)  # (T, C, H, W)

    # ---- Length correction (T → seq_len) ----
    T, C, H, W = seq.shape

    if T >= seq_len:
        # 앞에서부터 seq_len 프레임 사용
        seq = seq[:seq_len]
    else:
        # 부족하면 마지막 프레임을 복제해서 채우기
        last = seq[-1:].expand(seq_len - T, C, H, W)
        seq = torch.cat([seq, last], dim=0)

    # ---- Resize to (IMG_SIZE, IMG_SIZE) ----
    seq = F_torch.interpolate(
        seq,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False
    )

    return seq


# =============================================================
# 3. Case1 Dataset Class
# =============================================================


class Case1SequenceDataset(Dataset):
    """
    시퀀스 단위(real/fake) 이진 분류를 위한 Dataset.

    - 각 시퀀스는 개별 폴더에 저장되어 있으며,
      load_sequence()로 (T, C, H, W) 텐서로 로딩.
    """

    def __init__(
        self,
        seq_dirs: List[Path],
        labels: List[int],
        use_seq_norm: bool = True,
        seq_len: int = SEQ_LEN,
    ):
        """
        Args:
            seq_dirs (List[Path]): 시퀀스 폴더 경로 리스트
            labels (List[int]): 각 시퀀스의 라벨 (real=1, fake=0)
            use_seq_norm (bool): seq_norm 텐서 사용 여부
            seq_len (int): 시퀀스 길이
        """
        self.seq_dirs = seq_dirs
        self.labels = labels
        self.use_seq_norm = use_seq_norm
        self.seq_len = seq_len

    def __len__(self):
        return len(self.seq_dirs)

    def __getitem__(self, idx):
        seq_dir = self.seq_dirs[idx]
        label = self.labels[idx]

        seq = load_sequence(
            seq_dir,
            use_seq_norm=self.use_seq_norm,
            seq_len=self.seq_len
        )
        # 반환 형태: (T, C, H, W), scalar label
        return seq, label

    

# =============================================================
# 4. Model: ResNeXt50 + LSTM
# =============================================================

class ResNeXtLSTM(nn.Module):
    """
    ResNeXt50 백본 + LSTM을 이용한 시퀀스 이진 분류 모델.

    - 각 프레임을 CNN으로 임베딩(latent_dim)으로 투영
    - LSTM으로 temporal feature를 집계
    - 시퀀스 평균 풀링 후 FC로 이진 로짓(logit) 출력
    """

    def __init__(
        self,
        num_classes: int = 1,
        latent_dim: int = 2048,   # ResNeXt50 conv 출력 채널 수
        lstm_layers: int = 1,
        hidden_dim: int = 1024,   # LSTM hidden dim
        bidirectional: bool = False,
        dropout: float = 0.5,
        cnn_finetune: bool = False,  # True이면 layer4만 fine-tuning
    ):
        super().__init__()

        # 1) ResNeXt50 backbone (ImageNet pretrain)
        base = models.resnext50_32x4d(pretrained=True)
        # 마지막 FC를 제외한 feature extractor
        self.feature_extractor = nn.Sequential(*list(base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 2) CNN 파라미터 동결 (기본값: 전부 freeze)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 3) cnn_finetune=True이면 마지막 block(layer4)만 미세조정 허용
        if cnn_finetune:
            for name, param in self.feature_extractor.named_parameters():
                if "layer4" in name:
                    param.requires_grad = True

        # 4) LSTM
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # 5) FC
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, T, C, H, W)

        Returns:
            torch.Tensor: (B, 1) 시퀀스 단위 로짓
        """
        B, T, C, H, W = x.shape

        # (B*T, C, H, W)
        x = x.view(B * T, C, H, W)

        # CNN feature 추출
        fmap = self.feature_extractor(x)  # (B*T, 2048, h, w)
        x = self.avgpool(fmap)            # (B*T, 2048, 1, 1)
        x = x.view(B, T, -1)              # (B, T, 2048)

        # LSTM
        x_lstm, _ = self.lstm(x)          # (B, T, hidden_dim)

        # 시퀀스 평균 풀링 (T dimension)
        x = torch.mean(x_lstm, dim=1)     # (B, hidden_dim)

        # FC
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)                    # (B, 1) 로짓

        return x


# =============================================================
# 5. Collect train sequences
# =============================================================

def collect_sequences(root: Path):
    """
    root/real, root/fake 폴더에서 각각 시퀀스 디렉터리를 수집하고
    라벨(real=1, fake=0)을 함께 반환.

    Args:
        root (Path): train 또는 test(root) 디렉터리

    Returns:
        seq_dirs (List[Path]): 시퀀스 폴더 경로 리스트
        labels (List[int]): 각 시퀀스의 라벨 리스트
    """
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

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """
    한 epoch 동안 train_loader 전체에 대해 학습 수행.

    Returns:
        epoch_loss (float): 평균 학습 loss
        epoch_acc  (float): 학습 accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for seqs, labels in tqdm(loader, desc="Train", leave=False):
        seqs = seqs.to(device)      # (B, T, C, H, W)
        labels = labels.to(device)  # (B,)

        optimizer.zero_grad()

        # BCEWithLogitsLoss용 (B,) → (B,1)
        labels_float = labels.float().unsqueeze(1)

        outputs = model(seqs)  # (B,1), 로짓
        loss = criterion(outputs, labels_float)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * seqs.size(0)

        # 시그모이드 후 0.5 threshold로 예측
        preds = (torch.sigmoid(outputs) > 0.5).long()  # (B,1)

        correct += (preds == labels.unsqueeze(1)).sum().item()
        total   += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total

    return epoch_loss, epoch_acc

def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """
    검증/테스트 데이터에 대한 1 epoch 평가.

    Returns:
        epoch_loss (float): 평균 검증 loss
        epoch_acc  (float): 검증 accuracy
        y_true (np.ndarray): 정답 라벨 배열
        y_pred (np.ndarray): 예측 라벨 배열
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for seqs, labels in tqdm(loader, desc="Valid", leave=False):
            seqs = seqs.to(device)
            labels = labels.to(device)   # (B,)

            labels_float = labels.float().unsqueeze(1)  # (B,1)

            outputs = model(seqs)        # (B,1)
            loss = criterion(outputs, labels_float)

            running_loss += loss.item() * seqs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).long()  # (B,1)

            correct += (preds == labels.unsqueeze(1)).sum().item()
            total   += labels.size(0)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().squeeze(1).tolist())

    epoch_loss = running_loss / total
    epoch_acc  = correct / total

    return epoch_loss, epoch_acc, np.array(y_true), np.array(y_pred)


# =============================================================
# 7. Train Loop (Main) + Early Stopping(val_acc 기준)
# =============================================================

def main():
    # ---------------------
    # 0. Reproducibility
    # ---------------------
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

    train_dataset = Case1SequenceDataset(
        train_dirs,
        train_labels,
        use_seq_norm=USE_SEQ_NORM,
        seq_len=SEQ_LEN
    )
    val_dataset = Case1SequenceDataset(
        val_dirs,
        val_labels,
        use_seq_norm=USE_SEQ_NORM,
        seq_len=SEQ_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ---------------------
    # 2. Model / Loss / Optimizer
    # ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # CNN은 기본 freeze, LSTM hidden_dim만 축소(1024→512)
    model = ResNeXtLSTM(
        hidden_dim=512,
        cnn_finetune=False  # 필요 시 True로 설정하여 layer4만 미세조정
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ---------------------
    # 3. Training Loop + Early Stopping
    # ---------------------
    best_val_acc = 0.0
    epochs_no_improve = 0  # Early Stopping 카운터

    # 기록용 리스트
    train_losses, val_losses = [], []
    train_accs,  val_accs   = [], []

    # 마지막 epoch의 예측/정답 (리포트용)
    last_y_true, last_y_pred = None, None

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, y_true, y_pred = eval_one_epoch(
            model, val_loader, criterion, device
        )

        # 기록
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        last_y_true, last_y_pred = y_true, y_pred

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"[Epoch {epoch}] Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # ---------------------
        # Early Stopping (검증 정확도 기준)
        # ---------------------
        # val_acc가 MIN_DELTA 이상 개선되었는지 확인
        if val_acc - best_val_acc > MIN_DELTA:
            best_val_acc = val_acc
            epochs_no_improve = 0

            # 베스트 모델 가중치 저장
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f">>>> [BEST] val_acc={best_val_acc:.4f} (model saved)")
        else:
            epochs_no_improve += 1
            print(f"[EarlyStopping] No improvement for {epochs_no_improve} epoch(s) "
                  f"(best_val_acc={best_val_acc:.4f})")

            if epochs_no_improve >= PATIENCE:
                print(f"[EarlyStopping] Stop training at epoch {epoch}. "
                      f"(patience={PATIENCE})")
                break

    print("\n[Training Finished]")
    print("Best Validation Accuracy =", best_val_acc)

    # ---------------------
    # 4. Plot Loss Curve
    # ---------------------
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------------------
    # 5. Confusion Matrix & Classification Report
    #     (마지막 epoch 기준, Early Stopping 종료 시점)
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

    # Confusion Matrix
    cm_df = pd.DataFrame(
        cm,
        index=["true_fake(0)", "true_real(1)"],
        columns=["pred_fake(0)", "pred_real(1)"]
    )

    # Classification Report (딕셔너리 형태 → DataFrame)
    report_dict = classification_report(
        last_y_true,
        last_y_pred,
        output_dict=True,
        digits=4
    )
    report_df = pd.DataFrame(report_dict).T

    excel_path_str = str(EXCEL_LOG_PATH)
    with pd.ExcelWriter(excel_path_str, engine="openpyxl") as writer:
        history_df.to_excel(writer, sheet_name="history", index=False)
        cm_df.to_excel(writer, sheet_name="confusion_matrix")
        report_df.to_excel(writer, sheet_name="classification_report")

    print(f"\n[LOG] Training log saved to: {excel_path_str}")


if __name__ == "__main__":
    main()