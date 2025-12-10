# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 08:13:05 2025

@author: deoha
"""

import os
import random
from pathlib import Path
from typing import List

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
# 1. Path Configs (로컬 경로로 수정해서 사용)
# ======================================== 
DATA_ROOT = Path(r"C:\user\Processed_data")  # 입력 경로설정

TRAIN_ROOT = DATA_ROOT / "train"
TEST_ROOT  = DATA_ROOT / "test"
USE_SEQ_NORM = True        # seq_norm.pt 우선 사용
SEQ_LEN = 10               # 시퀀스 길이
IMG_SIZE = 256             # 입력 이미지 크기 (MobileNetV3에 맞게 224~256 권장)
BATCH_SIZE =16
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42

MODEL_SAVE_PATH = DATA_ROOT / "mobilenetv3_conv_lstm_best.pth"
EXCEL_LOG_PATH = DATA_ROOT / "training_log_conv_lstm.xlsx"

print("DATA_ROOT =", DATA_ROOT)

# =============================================================
# 2. Utility Functions (시퀀스 로딩 관련)
# =============================================================

def list_frame_paths(seq_dir: Path) -> List[Path]:
    """
    주어진 시퀀스 폴더(seq_dir) 내에서
    이미지 프레임(jpg, png, bmp 등) 경로를 정렬된 리스트로 반환.
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
    하나의 시퀀스 폴더에서 프레임 텐서를 로드하는 함수.

    1) seq_norm.pt 존재 시:
       - 사전에 정규화/전처리된 텐서(형태: (T, C, H, W))를 직접 로드
    2) 없을 경우:
       - jpg/png 프레임들을 PIL로 읽어서 ToTensor + ImageNet Normalize 수행
    3) 길이 보정:
       - T >= seq_len : 앞에서 seq_len 만큼 자름
       - T <  seq_len : 마지막 프레임을 복제해서 길이를 seq_len까지 늘림
    4) 최종적으로 (seq_len, C, IMG_SIZE, IMG_SIZE) 형태로 리턴
    """
    seq_norm_path = seq_dir / "seq_norm.pt"

    # ---- Case 1: 정규화된 텐서가 존재하면 그걸 사용 ----
    if use_seq_norm and seq_norm_path.exists():
        # 저장된 텐서는 (T, C, H, W) 형태라고 가정
        seq = torch.load(seq_norm_path, map_location="cpu")

    # ---- Case 2: jpg 프레임 로딩 후 normalize ----
    else:
        frame_paths = list_frame_paths(seq_dir)
        if not frame_paths:
            raise RuntimeError(f"No frames in {seq_dir}")

        # ImageNet 통계 기반 정규화
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            t = TF.to_tensor(img)  # (C, H, W), 0~1
            t = normalize(t)
            frames.append(t)

        seq = torch.stack(frames, dim=0)  # (T, C, H, W)

    # ---- Length correction (시퀀스 길이 보정) ----
    T, C, H, W = seq.shape

    if T >= seq_len:
        # 앞에서 seq_len 프레임만 사용
        seq = seq[:seq_len]
    else:
        # 마지막 프레임을 복제해서 길이를 맞춤
        last = seq[-1:].expand(seq_len - T, C, H, W)
        seq = torch.cat([seq, last], dim=0)

    # ---- Resize (모든 프레임을 IMG_SIZE로 리사이즈) ----
    seq = F_torch.interpolate(
        seq,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False
    )

    # (seq_len, C, IMG_SIZE, IMG_SIZE)
    return seq

# =============================================================
# 3. Dataset Class (시퀀스 단위 Dataset)
# =============================================================

class Case1SequenceDataset(Dataset):
    """
    시퀀스 폴더 목록 + 라벨 목록을 받아
    (T, C, H, W) 텐서와 이진 라벨(0/1)을 반환하는 Dataset.
    """
    def __init__(
        self,
        seq_dirs: List[Path],
        labels: List[int],
        use_seq_norm: bool = True,
        seq_len: int = SEQ_LEN
    ):
        self.seq_dirs = seq_dirs
        self.labels = labels
        self.use_seq_norm = use_seq_norm
        self.seq_len = seq_len

    def __len__(self):
        return len(self.seq_dirs)

    def __getitem__(self, idx):
        seq_dir = self.seq_dirs[idx]
        label = self.labels[idx]

        # (T, C, H, W) 시퀀스 텐서 로딩
        seq = load_sequence(
            seq_dir,
            use_seq_norm=self.use_seq_norm,
            seq_len=self.seq_len
        )
        return seq, label

# =============================================================
# 4. ConvLSTM Cell 정의
# =============================================================

class ConvLSTMCell(nn.Module):
    """
    단일 ConvLSTM 셀 구현.

    - 입력: x (B, input_dim, H, W)
    - 상태: h_cur, c_cur (B, hidden_dim, H, W)
    - 출력: h_next, c_next (동일한 spatial size 유지)
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 입력과 이전 hidden을 채널 방향으로 concat 후 Conv
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, h_cur, c_cur):
        # x:     (B, input_dim, H, W)
        # h_cur: (B, hidden_dim, H, W)
        # c_cur: (B, hidden_dim, H, W)
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)

        # 게이트 분리
        cc_i, cc_f, cc_o, cc_g = torch.split(
            conv_out, self.hidden_dim, dim=1
        )
        i = torch.sigmoid(cc_i)      # input gate
        f = torch.sigmoid(cc_f)      # forget gate
        o = torch.sigmoid(cc_o)      # output gate
        g = torch.tanh(cc_g)         # candidate cell

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# =============================================================
# 5. MobileNetV3 + ConvLSTM 기반 시퀀스 분류 모델
# =============================================================

class MobileNetV3ConvLSTM(nn.Module):
    """
    MobileNetV3-Large feature extractor + ConvLSTM 기반
    시퀀스 이진 분류 모델.

    - 입력:  x (B, T, C, H, W)
    - 출력: logits (B, num_classes)
      * num_classes=1인 경우, BCEWithLogitsLoss와 함께 사용 (이진 분류)
    """
    def __init__(
        self,
        num_classes: int = 1,
        latent_dim: int = 960,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.4,
        freeze_cnn: bool = True
    ):
        """
        Args
        ----
        num_classes : 출력 클래스 수 (이진 분류 → 1)
        latent_dim  : MobileNetV3 feature 채널 수 (Large 기준 960)
        hidden_dim  : ConvLSTM hidden 채널 수
        kernel_size : ConvLSTM 커널 크기 (3 권장)
        dropout     : FC 직전 dropout 비율
        freeze_cnn  : True면 feature extractor 파라미터 동결
        """
        super().__init__()

        # 1) ImageNet-pretrained MobileNetV3-Large backbone
        base = models.mobilenet_v3_large(pretrained=True)
        # feature extractor 부분만 사용 (classifier 제외)
        self.feature_extractor = base.features  # (B, 960, h, w)

        # 2) backbone 동결 여부
        if freeze_cnn:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # 3) ConvLSTM
        self.hidden_dim = hidden_dim
        self.conv_lstm = ConvLSTMCell(latent_dim, hidden_dim, kernel_size)

        # 4) Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # (B, hidden_dim, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args
        ----
        x : (B, T, C, H, W) 형태의 시퀀스 이미지 텐서

        Returns
        -------
        logits : (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # (B, T, C, H, W) -> (B*T, C, H, W)로 펼쳐서 CNN에 통과
        x = x.view(B * T, C, H, W)
        fmap = self.feature_extractor(x)  # (B*T, 960, h, w)

        # 다시 (B, T, C_f, H_f, W_f)로 reshape
        _, C_f, H_f, W_f = fmap.shape
        fmap = fmap.view(B, T, C_f, H_f, W_f)

        # 초기 hidden, cell (0으로 초기화)
        h = torch.zeros(B, self.hidden_dim, H_f, W_f, device=x.device)
        c = torch.zeros(B, self.hidden_dim, H_f, W_f, device=x.device)

        # 시퀀스 방향으로 ConvLSTM 순차 적용
        for t in range(T):
            h, c = self.conv_lstm(fmap[:, t], h, c)

        # 마지막 시간 스텝의 hidden 상태를 사용하여 분류
        x = self.avgpool(h)         # (B, hidden_dim, 1, 1)
        x = x.view(B, -1)           # (B, hidden_dim)
        x = self.dropout(self.fc(x))  # (B, num_classes)

        return x

# =============================================================
# 6. 시퀀스 폴더 수집 함수
# =============================================================

def collect_sequences(root: Path):
    """
    주어진 root(예: TRAIN_ROOT 또는 TEST_ROOT) 아래에서
    real / fake 폴더를 찾고, 각 시퀀스 폴더 경로와 라벨을 반환.

    Returns
    -------
    seq_dirs : List[Path]
        시퀀스 폴더 경로 리스트
    labels   : List[int]
        각 시퀀스의 라벨 (real=1, fake=0)
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
    print(f"[INFO] {root.name}/total: {len(seq_dirs)}")

    return seq_dirs, labels

# =============================================================
# 7. Train / Eval 한 epoch 수행 함수
# =============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    한 epoch 동안 train_loader를 순회하며 학습을 수행하고
    평균 손실과 정확도를 반환.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for seqs, labels in tqdm(loader, desc="Train", leave=False):
        seqs = seqs.to(device)      # (B, T, C, H, W)
        labels = labels.to(device)  # (B,)

        optimizer.zero_grad()

        labels_float = labels.float().unsqueeze(1)  # (B,) -> (B,1)
        outputs = model(seqs)                       # (B,1)
        loss = criterion(outputs, labels_float)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * seqs.size(0)

        probs = torch.sigmoid(outputs)             # (B,1)
        preds = (probs > 0.5).long()               # (B,1) → 0/1

        correct += (preds.squeeze(1) == labels).sum().item()
        total   += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    """
    한 epoch 동안 val_loader를 순회하며 평가를 수행하고,
    손실 / 정확도 / 예측값 / 확률 등을 반환.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    y_true, y_pred = [], []
    y_prob = []

    with torch.no_grad():
        for seqs, labels in tqdm(loader, desc="Valid", leave=False):
            seqs = seqs.to(device)
            labels = labels.to(device)

            labels_float = labels.float().unsqueeze(1)

            outputs = model(seqs)
            loss = criterion(outputs, labels_float)
            running_loss += loss.item() * seqs.size(0)

            probs = torch.sigmoid(outputs)   # (B,1), 0~1
            preds = (probs > 0.5).long()     # (B,1) → 0/1

            correct += (preds.squeeze(1) == labels).sum().item()
            total   += labels.size(0)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().squeeze(1).tolist())
            y_prob.extend(probs.cpu().squeeze(1).tolist())

    epoch_loss = running_loss / total
    epoch_acc  = correct / total

    return epoch_loss, epoch_acc, np.array(y_true), np.array(y_pred), np.array(y_prob)

# =============================================================
# 8. Train Loop (Main)
#    - EarlyStopping: "검증 정확도(val_acc)" 기준
#    - Validation 확률 분포 시각화 & Excel 로그 저장
# =============================================================

def main():
    # ---------------------
    # 0. Reproducibility (시드 고정)
    # ---------------------
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # ---------------------
    # 1. 데이터셋 수집
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
    val_dataset   = Case1SequenceDataset(
        val_dirs,
        val_labels,
        use_seq_norm=USE_SEQ_NORM,
        seq_len=SEQ_LEN
    )

    # Windows/Spyder 환경에서 num_workers>0 문제 발생 시 0으로 조정 필요
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader   = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ---------------------
    # 2. Model / Loss / Optimizer 정의
    # ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # freeze_cnn=True → ImageNet feature extractor 전체 동결
    model = MobileNetV3ConvLSTM(freeze_cnn=True).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # ---------------------
    # 3. EarlyStopping 설정 (검증 정확도 기준)
    # ---------------------
    #  - best_val_acc: 지금까지의 최고 검증 정확도
    #  - patience: 향상 없이 몇 epoch 연속되면 중단할지
    best_val_acc = 0.0
    patience = 5
    epochs_no_improve = 0

    # 기록용 리스트 (손실/정확도 history)
    train_losses, val_losses = [], []
    train_accs,  val_accs   = [], []

    # 마지막 epoch(or early stopping 직전 epoch)의 예측 값 저장
    last_y_true, last_y_pred, last_y_prob = None, None, None

    # ---------------------
    # 4. Training Loop
    # ---------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, y_true, y_pred, y_prob = eval_one_epoch(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        last_y_true, last_y_pred, last_y_prob = y_true, y_pred, y_prob

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"[Epoch {epoch}] Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # ---- EarlyStopping + best model 저장 (val_acc 기준) ----
        # 작은 향상도 인정 (1e-4)
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f">>>> Best model updated! val_acc={best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] No improvement in val_acc for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(
                    f"\n[EarlyStopping] Validation accuracy has not improved "
                    f"for {patience} consecutive epochs. Stop training."
                )
                break

    print("\n[Training Finished]")
    print(f"Best Validation Accuracy = {best_val_acc:.4f}")

    # ---------------------
    # 5. 손실 곡선 시각화
    # ---------------------
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Validation Loss")
    plt.grid(True)
    plt.show()

    # ---------------------
    # 6. Confusion Matrix & Report (마지막 eval 기준)
    # ---------------------
    if last_y_true is not None:
        cm = confusion_matrix(last_y_true, last_y_pred)
        print("Confusion Matrix:\n", cm)

        report_str = classification_report(
            last_y_true,
            last_y_pred,
            digits=4
        )
        print("\nClassification Report:\n", report_str)

        # ---------------------
        # 6-1. Validation 확률 분포 히스토그램
        #       - true_fake(0) / true_real(1) 별 예측 확률 분포
        # ---------------------
        plt.figure()
        plt.hist(
            last_y_prob[last_y_true == 0],
            bins=20,
            alpha=0.5,
            label="true_fake(0)"
        )
        plt.hist(
            last_y_prob[last_y_true == 1],
            bins=20,
            alpha=0.5,
            label="true_real(1)"
        )
        plt.xlabel("Predicted Probability (sigmoid output)")
        plt.ylabel("Count")
        plt.title("Validation Probability Distribution")
        plt.legend()
        plt.grid(True)
        plt.show()

        # ---------------------
        # 7. Excel 로그 저장
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
        with pd.ExcelWriter(excel_path_str, engine="openpyxl") as writer:
            history_df.to_excel(writer, sheet_name="history", index=False)
            cm_df.to_excel(writer, sheet_name="confusion_matrix")
            report_df.to_excel(writer, sheet_name="classification_report")

        print(f"\n[LOG] Training log saved to: {excel_path_str}")
    else:
        print("[WARN] last_y_true가 없습니다. eval이 실행되지 않은 것 같습니다.")


if __name__ == "__main__":
    main()