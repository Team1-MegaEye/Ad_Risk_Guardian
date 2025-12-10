
"""
MobileNetV3-Large + LSTM Deepfake Detector (PyTorch)
- Colab / Spyder (로컬) 공용 버전
- Sequence-level binary classification (real vs fake)
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
# 1. Path Configs (EDIT THIS FOR YOUR FOLDER!!)
# =============================================================

DATA_ROOT = Path(r"C:\user\Processed_data")  # 입력 경로 설정

TRAIN_ROOT = DATA_ROOT / "train"
TEST_ROOT  = DATA_ROOT / "test"

USE_SEQ_NORM = True        # seq_norm.pt 있으면 우선 사용
SEQ_LEN = 10               # 시퀀스 길이
IMG_SIZE = 256            # 입력 이미지 크기 
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42

# Early Stopping 설정 (Validation Accuracy 기준)
PATIENCE = 3  # 개선 없는 epoch가 PATIENCE번 연속 발생하면 중단

MODEL_SAVE_PATH = DATA_ROOT / "mobilenetv3_lstm_best.pth"
EXCEL_LOG_PATH = DATA_ROOT / "training_log_mobilenetv3_lstm.xlsx"

print("DATA_ROOT =", DATA_ROOT)


# =============================================================
# 2. Utility Functions
# =============================================================

def list_frame_paths(seq_dir: Path) -> List[Path]:
    """
    시퀀스 폴더 안의 이미지 프레임 경로를 정렬해서 반환.

    Args:
        seq_dir (Path): 한 시퀀스에 해당하는 디렉토리 경로

    Returns:
        List[Path]: 정렬된 프레임 이미지 경로 리스트
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
    하나의 시퀀스 폴더에서 (T, C, H, W) 텐서를 로드하고,
    길이를 seq_len으로 맞춘 뒤 IMG_SIZE x IMG_SIZE 로 리사이즈.

    우선순위:
    1) seq_norm.pt (정규화된 텐서) 가 존재하면 바로 로드
    2) 그렇지 않으면 프레임 이미지들을 직접 로드 후 normalize

    Args:
        seq_dir (Path): 시퀀스 디렉토리
        use_seq_norm (bool): seq_norm.pt 사용 여부
        seq_len (int): 사용할 시퀀스 길이

    Returns:
        torch.Tensor: (T, C, IMG_SIZE, IMG_SIZE)
    """
    seq_norm_path = seq_dir / "seq_norm.pt"

    # ---- Case 1: 정규화된 텐서가 존재하면 그걸 사용 ----
    if use_seq_norm and seq_norm_path.exists():
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
            t = TF.to_tensor(img)      # (C, H, W), 값 [0,1]
            t = normalize(t)
            frames.append(t)

        seq = torch.stack(frames, dim=0)  # (T, C, H, W)

    # ---- Length correction (길이 보정: T → seq_len) ----
    T, C, H, W = seq.shape
    if T >= seq_len:
        seq = seq[:seq_len]
    else:
        # 마지막 프레임을 반복해서 길이 맞추기
        last = seq[-1:].expand(seq_len - T, C, H, W)
        seq = torch.cat([seq, last], dim=0)

    # ---- Resize to IMG_SIZE x IMG_SIZE ----
    # 현재 shape: (T, C, H, W) → F.interpolate는 (N, C, H, W)를 기대
    seq = F_torch.interpolate(
        seq,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False
    )

    return seq  # (T, C, IMG_SIZE, IMG_SIZE)


# =============================================================
# 3. Dataset Class
# =============================================================

class Case1SequenceDataset(Dataset):
    """
    시퀀스 단위(real/fake) 분류용 Dataset.

    - seq_dirs: 각 샘플(시퀀스) 폴더 경로 리스트
    - labels: 해당 시퀀스의 레이블 (1: real, 0: fake)
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

        # (T, C, H, W)
        seq = load_sequence(
            seq_dir,
            use_seq_norm=self.use_seq_norm,
            seq_len=self.seq_len
        )

        # DataLoader가 batch 차원 B를 추가 → (B, T, C, H, W)
        return seq, label



# =============================================================
# 4. Model: MobileNetV3-Large + LSTM
# =============================================================

class MobileNetV3LSTM(nn.Module):
    """
    MobileNetV3-Large CNN + LSTM 기반 시퀀스 분류 모델.

    - CNN(Backbone): 프레임 단위 특징 추출
    - LSTM: 시간 축(T) 방향 시퀀스 모델링
    - 최종 FC: 이진 분류 (real vs fake)
    """

    def __init__(
        self,
        num_classes: int = 1,
        latent_dim: int = 960,      # MobileNetV3-Large 마지막 채널 수
        lstm_layers: int = 1,
        hidden_dim: int = 2048,
        bidirectional: bool = False,
        dropout: float = 0.4
    ):
        super().__init__()

        # 최신 버전에서는 weights=models.MobileNet_V3_Large_Weights.DEFAULT 권장
        base = models.mobilenet_v3_large(pretrained=True)

        # Conv feature extractor만 사용 (classifier 제거)
        self.feature_extractor = base.features  # (B, 960, H/32, W/32)

        # -------------------------------------------------
        # ① CNN 전체 먼저 freeze
        # -------------------------------------------------
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # -------------------------------------------------
        # ② 마지막 레이어(블록)만 unfreeze (fine-tuning)
        #    - 전체를 학습시키고 싶다면 아래 블록을 주석 처리
        # -------------------------------------------------
        for param in self.feature_extractor[-1].parameters():
            param.requires_grad = True

        # Adaptive Pooling: (B, 960, h, w) → (B, 960)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # LSTM: 프레임 시퀀스 단위로 temporal modeling
        self.lstm = nn.LSTM(
            input_size=latent_dim,     # 960
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)  # 이진 분류라 1 출력

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, T, C, H, W)

        Returns:
            torch.Tensor: (B, 1) - raw logits (BCEWithLogitsLoss 사용)
        """
        B, T, C, H, W = x.shape

        # CNN input reshape: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)
        fmap = self.feature_extractor(x)    # (B*T, 960, h, w)

        # Adaptive Pooling → (B*T, 960)
        x = self.avgpool(fmap)              # (B*T, 960, 1, 1)
        x = x.view(B, T, -1)                # (B, T, 960)

        # LSTM: (B, T, 960) → (B, T, hidden_dim)
        x_lstm, _ = self.lstm(x)

        # Temporal mean pooling (T 차원 평균)
        x = torch.mean(x_lstm, dim=1)       # (B, hidden_dim)

        # FC layer
        x = self.fc(self.dropout(self.relu(x)))  # (B, 1)

        return x


# =============================================================
# 5. Collect train/test sequences
# =============================================================

def collect_sequences(root: Path):
    """
    root 내부의 real, fake 폴더에서 시퀀스 디렉토리 수집.

    구조 예시:
        root/
          real/
            vid001/
            vid002/
            ...
          fake/
            vid101/
            vid102/
            ...

    Args:
        root (Path): train 또는 test(=validation) 루트 경로

    Returns:
        seq_dirs (List[Path]): 시퀀스 디렉토리 리스트
        labels   (List[int]): 각 시퀀스에 대한 레이블 (real:1, fake:0)
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
    print(f"[INFO] {root.name} total: {len(seq_dirs)}")

    return seq_dirs, labels


# =============================================================
# 6. Train / Eval Functions
# =============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    한 epoch 동안 train_loader 전체에 대해 학습을 수행.

    Returns:
        epoch_loss (float): 평균 train loss
        epoch_acc  (float): train accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for seqs, labels in tqdm(loader, desc="Train", leave=False):
        seqs = seqs.to(device)        # (B, T, C, H, W)
        labels = labels.to(device)    # (B,)

        optimizer.zero_grad()

        # BCEWithLogitsLoss → float + (B, 1)
        labels_float = labels.float().unsqueeze(1)  # (B, 1)
        outputs = model(seqs)                       # (B, 1)

        loss = criterion(outputs, labels_float)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * seqs.size(0)

        # 이진 분류 prediction (0/1)
        preds = (torch.sigmoid(outputs) > 0.5).long()  # (B, 1)

        # accuracy 계산
        correct += (preds.squeeze(1) == labels.long()).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    """
    한 epoch 동안 val_loader 전체에 대해 검증을 수행.

    Returns:
        epoch_loss (float): 평균 validation loss
        epoch_acc  (float): validation accuracy
        y_true (np.ndarray): 전체 GT 레이블
        y_pred (np.ndarray): 전체 예측 레이블 (0/1)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for seqs, labels in tqdm(loader, desc="Valid", leave=False):
            seqs = seqs.to(device)
            labels = labels.to(device)      # (B,)

            labels_float = labels.float().unsqueeze(1)   # (B, 1)
            outputs = model(seqs)                        # (B, 1)

            loss = criterion(outputs, labels_float)
            running_loss += loss.item() * seqs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).long()   # (B, 1)

            # accuracy 계산
            correct += (preds.squeeze(1) == labels.long()).sum().item()
            total += labels.size(0)

            # metric 저장
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().squeeze(1).tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, np.array(y_true), np.array(y_pred)


# =============================================================
# 7. Train Loop (Main + Early Stopping)
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
    val_dirs,   val_labels   = collect_sequences(TEST_ROOT)   # 여기서는 validation 용도

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

    # ⚠ Windows/Spyder에서는 num_workers=0 권장
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ---------------------
    # 2. Model / Loss / Optimizer
    # ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = MobileNetV3LSTM().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Early Stopping 변수
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs,  val_accs  = [], []

    last_y_true, last_y_pred = None, None  # 마지막 epoch의 label 저장용

    # ---------------------
    # 3. Training Loop (+ Early Stopping)
    # ---------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")

        # ---- Train ----
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # ---- Validation ----
        val_loss, val_acc, y_true, y_pred = eval_one_epoch(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        last_y_true, last_y_pred = y_true, y_pred

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"[Epoch {epoch}] Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # ---- Best model 저장 & Early Stopping 로직 ----
        if val_acc > best_val_acc:
            # 정확도 개선 → best 갱신
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0  # patience 리셋

            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f">>>> Best model updated! val_acc={best_val_acc:.4f} (epoch={epoch})")
        else:
            # 개선 없음 → patience 카운트 증가
            patience_counter += 1
            print(f"[EarlyStopping] No improvement. patience_counter={patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"[EarlyStopping] Validation accuracy did not improve for {PATIENCE} epochs.")
                print(f"               Stop training at epoch {epoch}. Best epoch: {best_epoch}, best val_acc: {best_val_acc:.4f}")
                break

    print("\n[Training Finished]")
    print(f"Best Validation Accuracy = {best_val_acc:.4f} (epoch={best_epoch})")

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
    plt.tight_layout()
    plt.show()

    # ---------------------
    # 5. Confusion Matrix & Report (마지막 epoch 기준)
    # ---------------------
    if last_y_true is not None and last_y_pred is not None:
        cm = confusion_matrix(last_y_true, last_y_pred)
        print("Confusion Matrix:\n", cm)

        report_str = classification_report(last_y_true, last_y_pred, digits=4)
        print("\nClassification Report:\n", report_str)
    else:
        cm = np.array([[0, 0], [0, 0]])
        report_str = "No evaluation results available."

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

    if last_y_true is not None and last_y_pred is not None:
        report_dict = classification_report(
            last_y_true,
            last_y_pred,
            output_dict=True,
            digits=4
        )
        report_df = pd.DataFrame(report_dict).T
    else:
        report_df = pd.DataFrame()

    excel_path_str = str(EXCEL_LOG_PATH)
    with pd.ExcelWriter(excel_path_str, engine="openpyxl") as writer:
        history_df.to_excel(writer, sheet_name="history", index=False)
        cm_df.to_excel(writer, sheet_name="confusion_matrix")
        if not report_df.empty:
            report_df.to_excel(writer, sheet_name="classification_report")

    print(f"\n[LOG] Training log saved to: {excel_path_str}")


# =============================================================
# 8. Entry Point
# =============================================================

if __name__ == "__main__":
    main()
