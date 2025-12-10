

# -*- coding: utf-8 -*-
"""
MobileNetV3-Large + BiLSTM 기반 딥페이크 탐지 (시퀀스 단위 이진 분류)

- Backbone: MobileNetV3-Large (ImageNet 사전학습, feature extractor로 사용)
- Temporal encoder: BiLSTM
- Task: sequence-level binary classification (real=1, fake=0)
- Environment: Windows + Spyder (num_workers=0, __main__ guard 적용)
- Extras:
    * Early Stopping (validation loss 기준)
    * Best model weight 저장 및 재평가
    * Validation 예측 확률 분포 시각화
    * 학습 로그(손실/정확도, 혼동행렬, 분류 보고서) Excel 저장

구성 개요
---------
1) [Config] 경로 및 하이퍼파라미터 설정
2) [Utils] 시퀀스 로딩/전처리 유틸 함수
3) [Dataset] 시퀀스 Dataset 클래스
4) [Model] MobileNetV3-Large + BiLSTM 모델 정의
5) [Data Collect] train/val 시퀀스 경로 및 라벨 수집
6) [Train/Eval] 한 epoch 학습/평가 함수
7) [Analysis] Validation 확률 분포 분석/시각화
8) [Main] 전체 학습 루프 + Early Stopping + 로그/시각화
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

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
import pandas as pd
from tqdm import tqdm  # 진행바

# =============================================================
# 1. Path Configs (로컬 환경에 맞게 수정!!)
# =============================================================

# ★★★ 반드시 본인 로컬 경로로 수정 ★★★
DATA_ROOT = Path(r"C:\Video_Dataset\Processed_data")  # 입력 경로 설정 

TRAIN_ROOT = DATA_ROOT / "train"
TEST_ROOT  = DATA_ROOT / "test"
USE_SEQ_NORM = True        # seq_norm.pt 우선 사용
SEQ_LEN = 10               # 시퀀스 길이
IMG_SIZE = 256             # MobileNetV3 입력 이미지 크기(256도 사용 가능)
BATCH_SIZE = 16
NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42

# 윈도우/Spyder에서는 보통 num_workers=0 이 안전
NUM_WORKERS = 0

# Early Stopping 설정 (val_loss 기준)
EARLY_STOPPING_PATIENCE = 3   # 개선 없으면 몇 epoch 후 멈출지
MIN_DELTA = 1e-4             # 개선으로 볼 최소 변화량

MODEL_SAVE_PATH = DATA_ROOT / "mobilenetv3_bilstm_best_Freeze.pth"
EXCEL_LOG_PATH = DATA_ROOT / "training_log_mobilenetv3_bilstm_Freeze.xlsx"

print("DATA_ROOT =", DATA_ROOT)


# =============================================================
# 2. Utility Functions
# =============================================================

def list_frame_paths(seq_dir: Path) -> List[Path]:
    """
    List all frame image paths in a sequence directory.

    Parameters
    ----------
    seq_dir : Path
        Directory containing frame images for one video sequence.

    Returns
    -------
    List[Path]
        Sorted list of image file paths (e.g., .jpg, .png).
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
    Load a video sequence as a 4D tensor (T, C, H, W).

    Priority:
        1) If seq_norm.pt exists and use_seq_norm=True, load it directly.
        2) Otherwise, load individual frame images and normalize with
           ImageNet mean/std.

    Also:
        - Adjust sequence length to `seq_len` by truncating or padding.
        - Resize frames to (IMG_SIZE, IMG_SIZE).

    Parameters
    ----------
    seq_dir : Path
        Directory containing sequence data (frames and/or seq_norm.pt).
    use_seq_norm : bool, optional
        Whether to prefer pre-computed normalized tensor (seq_norm.pt),
        by default True.
    seq_len : int, optional
        Target sequence length, by default SEQ_LEN.

    Returns
    -------
    torch.Tensor
        Tensor of shape (T, C, H, W) after length correction and resizing.
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

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            t = TF.to_tensor(img)   # (C, H, W), float32 in [0, 1]
            t = normalize(t)
            frames.append(t)

        seq = torch.stack(frames, dim=0)   # (T, C, H, W)

    # ---- Length correction ----
    T, C, H, W = seq.shape

    if T >= seq_len:
        # 앞에서 seq_len 개만 사용
        seq = seq[:seq_len]
    else:
        # 마지막 프레임을 복제하여 seq_len까지 패딩
        last = seq[-1:].expand(seq_len - T, C, H, W)
        seq = torch.cat([seq, last], dim=0)

    # ---- Resize to IMG_SIZE x IMG_SIZE ----
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
    Dataset for sequence-level deepfake detection.

    Each item is:
        - seq: tensor of shape (T, C, H, W)
        - label: int 0 or 1 (fake=0, real=1)
    """

    def __init__(
        self,
        seq_dirs: List[Path],
        labels: List[int],
        use_seq_norm: bool = True,
        seq_len: int = SEQ_LEN
    ):
        """
        Parameters
        ----------
        seq_dirs : List[Path]
            List of sequence directories (each corresponds to one sample).
        labels : List[int]
            List of integer labels (0: fake, 1: real).
        use_seq_norm : bool, optional
            Whether to use seq_norm.pt if available, by default True.
        seq_len : int, optional
            Target sequence length, by default SEQ_LEN.
        """
        self.seq_dirs = seq_dirs
        self.labels = labels
        self.use_seq_norm = use_seq_norm
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.seq_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq_dir = self.seq_dirs[idx]
        label = self.labels[idx]
        seq = load_sequence(
            seq_dir,
            use_seq_norm=self.use_seq_norm,
            seq_len=self.seq_len
        )
        # seq: (T, C, H, W)
        return seq, label


# =============================================================
# 4. MobileNetV3-Large + BiLSTM
# =============================================================

class MobileNetV3BiLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        latent_dim: int = 960,
        lstm_layers: int = 1,
        hidden_dim: int = 2048,
        bidirectional: bool = True,
        dropout: float = 0.4,
        # -------------------------
        # [Hyperparameter] Backbone fine-tuning 전략
        #   - "freeze_all"        : 백본 전체 동결 (feature extractor로만 사용)
        #   - "unfreeze_last"     : 마지막 block만 언프리즈해서 미세 조정
        # -------------------------
        backbone_mode: str = "freeze_all",
    ):
        super().__init__()
        
        # ------------------------------------------------------
        # Load MobileNetV3-Large backbone
        #
        # - pretrained=True  : Load ImageNet pre-trained weights
        # ------------------------------------------------------
        base = models.mobilenet_v3_large(pretrained=True)

        # ------------------------------------------------------
        # 1) Backbone (CNN feature extractor)
        #    - base.features 출력 shape: (B, 960, H/32, W/32)
        # ------------------------------------------------------
        self.feature_extractor = base.features

        # ------------------------------------------------------
        # [Hyperparameter: Backbone fine-tuning 전략]
        #
        #  (A) freeze_all
        #      - 백본 전체를 동결 (requires_grad = False)
        #      - LSTM + FC 헤드만 학습
        #      - 장점: 안정적, 작은 데이터셋에서도 과적합/붕괴 위험↓
        #
        #  (B) unfreeze_last
        #      - 마지막 block만 언프리즈
        #      - 상위 레벨 feature를 데이터셋에 맞게 미세 조정
        #      - 장점: 성능 향상 여지 ↑, 단 학습 불안정/과적합 가능성 존재
        #
        #  → 본 연구에서는 두 설정 모두 실험하여 결과 비교
        #    (실험 로그/결과는 보고서/논문에서 상세 기술)
        # ------------------------------------------------------
        if backbone_mode == "freeze_all":
            # (A) 백본 전체 동결
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        elif backbone_mode == "unfreeze_last":
            # (B) 백본 전체를 일단 동결한 후,
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

            # 마지막 block만 언프리즈 (실제 구조: Sequential([...]) 이므로 -1 인덱스)
            last_block = self.feature_extractor[-1]
            for param in last_block.parameters():
                param.requires_grad = True


        else:
            raise ValueError(f"Unknown backbone_mode: {backbone_mode}")

        # ------------------------------------------------------
        # 2) BiLSTM + Classifier Head
        # ------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # output: (B, 960, 1, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, C, H, W).

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (B, 1).
        """
        B, T, C, H, W = x.shape

        # CNN input reshape: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)

        # Feature extraction
        fmap = self.feature_extractor(x)    # (B*T, 960, h, w)

        # Adaptive pooling → (B*T, 960, 1, 1)
        x = self.avgpool(fmap)

        # Reshape back to (B, T, 960)
        x = x.view(B, T, -1)

        # LSTM: (B, T, 960) → (B, T, hidden_dim*dir)
        x_lstm, _ = self.lstm(x)

        # Temporal mean pooling over T
        x = torch.mean(x_lstm, dim=1)       # (B, hidden_dim*dir)

        # Final classifier head
        x = self.fc(self.dropout(self.relu(x)))  # (B, 1)

        return x

# =============================================================
# 5. Collect train sequences
# =============================================================

def collect_sequences(root: Path) -> Tuple[List[Path], List[int]]:
    """
    Collect sequence directories and labels from a root folder.

    Expected folder structure:
        root/
            real/
                seq_00001/
                ...
            fake/
                seq_00001/
                ...

    Parameters
    ----------
    root : Path
        Root directory containing `real` and `fake` subfolders.

    Returns
    -------
    seq_dirs : List[Path]
        List of sequence directories.
    labels : List[int]
        Corresponding labels (1 for real, 0 for fake).
    """
    real_root = root / "real"
    fake_root = root / "fake"

    real_dirs = [d for d in real_root.iterdir() if d.is_dir()] if real_root.exists() else []
    fake_dirs = [d for d in fake_root.iterdir() if d.is_dir()] if fake_root.exists() else []

    real_dirs.sort()
    fake_dirs.sort()

    seq_dirs = real_dirs + fake_dirs
    labels = [1] * len(real_dirs) + [0] * len(fake_dirs)

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
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    loader : DataLoader
        Dataloader for training set.
    criterion : nn.Module
        Loss function (e.g., BCEWithLogitsLoss).
    optimizer : torch.optim.Optimizer
        Optimizer (e.g., Adam).
    device : torch.device
        Device to use ("cuda" or "cpu").

    Returns
    -------
    epoch_loss : float
        Average training loss over the epoch.
    epoch_acc : float
        Average training accuracy over the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for seqs, labels in tqdm(loader, desc="Train", leave=False):
        seqs = seqs.to(device)                     # (B, T, C, H, W)
        labels = labels.to(device)                 # (B,)

        optimizer.zero_grad()

        labels_float = labels.float().unsqueeze(1)  # (B, 1)
        outputs = model(seqs)                       # (B, 1)

        loss = criterion(outputs, labels_float)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * seqs.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).long()  # (B, 1)
        correct += (preds == labels.unsqueeze(1)).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc



def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    loader : DataLoader
        Dataloader for validation/test set.
    criterion : nn.Module
        Loss function (e.g., BCEWithLogitsLoss).
    device : torch.device
        Device to use ("cuda" or "cpu").

    Returns
    -------
    epoch_loss : float
        Average validation loss.
    epoch_acc : float
        Average validation accuracy.
    y_true : np.ndarray
        Ground-truth labels (shape: (N,)).
    y_pred : np.ndarray
        Predicted labels (shape: (N,)).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for seqs, labels in tqdm(loader, desc="Valid", leave=False):
            seqs = seqs.to(device)
            labels = labels.to(device)

            labels_float = labels.float().unsqueeze(1)  # (B, 1)
            outputs = model(seqs)                       # (B, 1)

            loss = criterion(outputs, labels_float)
            running_loss += loss.item() * seqs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).long()

            correct += (preds == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().squeeze(1).tolist())

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc, np.array(y_true), np.array(y_pred)


# =============================================================
# 6-1. Validation 확률 분포 시각화 함수
# =============================================================

def analyze_val_prob_distribution(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze and visualize probability distribution on validation set.

    For each sample:
        p = sigmoid(logit) (probability of REAL, label=1)

    Then:
        - Plot histograms of p for label=0 (fake) and label=1 (real).

    Parameters
    ----------
    model : nn.Module
        Trained model (best model recommended).
    loader : DataLoader
        Validation dataloader.
    device : torch.device
        Device to use ("cuda" or "cpu").

    Returns
    -------
    all_probs : np.ndarray
        Predicted probabilities for all validation samples.
    all_labels : np.ndarray
        Ground-truth labels for all validation samples.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for seqs, labels in loader:
            seqs = seqs.to(device)
            labels = labels.to(device)

            logits = model(seqs)               # (B, 1)
            probs = torch.sigmoid(logits)      # (B, 1)

            all_probs.extend(probs.cpu().numpy().ravel().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # fake(0), real(1) 나눠서 그리기
    fake_mask = (all_labels == 0)
    real_mask = (all_labels == 1)

    plt.figure()
    plt.hist(all_probs[fake_mask], bins=20, alpha=0.5, label="fake (label=0)")
    plt.hist(all_probs[real_mask], bins=20, alpha=0.5, label="real (label=1)")
    plt.xlabel("Predicted probability of REAL (sigmoid output)")
    plt.ylabel("Count")
    plt.title("Validation probability distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return all_probs, all_labels


# =============================================================
# 7. Train Loop (Main)
# =============================================================

def main() -> None:
    """
    Main training pipeline.

    Steps
    -----
    1) Set random seeds for reproducibility.
    2) Collect train/validation sequences and labels.
    3) Create Dataset and DataLoader objects.
    4) Initialize model, loss function, optimizer.
    5) Run training loop with Early Stopping (val_loss 기준).
    6) Load best model and evaluate on validation set.
    7) Save training history, confusion matrix, classification report to Excel.
    8) Visualize validation probability distribution.
    """
    # ---------------------
    # 1. Reproducibility
    # ---------------------
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # ---------------------
    # 2. Collect dataset
    # ---------------------
    train_dirs, train_labels = collect_sequences(TRAIN_ROOT)
    val_dirs, val_labels = collect_sequences(TEST_ROOT)

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
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )



    # ---------------------
    # 3. Model / Loss / Optimizer
    # ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = MobileNetV3BiLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Early Stopping용 변수
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    # 기록용 리스트
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # ---------------------
    # 4. Training Loop
    # ---------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, y_true, y_pred = eval_one_epoch(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"[Epoch {epoch}] Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # ---------------------
        # Early Stopping (val_acc 기준)
        # ---------------------
        if val_acc > (best_val_acc + MIN_DELTA):
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
        
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(
                f">>>> Best model updated! "
                f"(epoch={best_epoch}, val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f})"
            )
        else:
            epochs_no_improve += 1
            print(
                f"[EarlyStopping] val_acc 개선 없음: "
                f"{epochs_no_improve}/{EARLY_STOPPING_PATIENCE}"
            )
        
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(
                    f"[EarlyStopping] patience={EARLY_STOPPING_PATIENCE} 도달, 학습 중단."
                )
                break


    print("\n[Training Finished]")
    print(f"Best Validation Loss = {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"Best Validation Acc  = {best_val_acc:.4f}")

    # ---------------------
    # 5. Plot Loss Curve
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
    # 6. Confusion Matrix & Report (best 모델 기준)
    # ---------------------
    # best 모델 로드
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)

    # best 모델 기준으로 다시 validation 평가
    _, _, y_true_best, y_pred_best = eval_one_epoch(
        model, val_loader, criterion, device
    )

    cm = confusion_matrix(y_true_best, y_pred_best)
    print("Confusion Matrix (best model):\n", cm)

    report_str = classification_report(y_true_best, y_pred_best, digits=4)
    print("\nClassification Report (best model):\n", report_str)

    # ---------------------
    # 7. Excel 로그 저장
    # ---------------------
    history_df = pd.DataFrame(
        {
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "train_acc": train_accs,
            "val_loss": val_losses,
            "val_acc": val_accs,
        }
    )

    cm_df = pd.DataFrame(
        cm,
        index=["true_fake(0)", "true_real(1)"],
        columns=["pred_fake(0)", "pred_real(1)"],
    )

    report_dict = classification_report(
        y_true_best,
        y_pred_best,
        output_dict=True,
        digits=4,
    )
    report_df = pd.DataFrame(report_dict).T

    excel_path_str = str(EXCEL_LOG_PATH)
    os.makedirs(EXCEL_LOG_PATH.parent, exist_ok=True)

    with pd.ExcelWriter(excel_path_str, engine="openpyxl") as writer:
        history_df.to_excel(writer, sheet_name="history", index=False)
        cm_df.to_excel(writer, sheet_name="confusion_matrix")
        report_df.to_excel(writer, sheet_name="classification_report")

    print(f"\n[LOG] Training log saved to: {excel_path_str}")

    # ---------------------
    # 8. Validation 확률 분포 시각화 (best 모델 기준)
    # ---------------------
    analyze_val_prob_distribution(model, val_loader, device)


# =============================================================
# 8. Entry Point (윈도우/Spyder 필수)
# =============================================================

if __name__ == "__main__":
    main()
