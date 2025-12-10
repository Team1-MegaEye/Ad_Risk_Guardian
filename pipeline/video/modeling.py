
# -*- coding: utf-8 -*-
"""
MobileNetV3-Large + LSTM 기반 딥페이크 탐지 (시퀀스 이진 분류)

구성 개요
---------
1) 데이터셋 구성
   - DATA_ROOT/train : 학습(Train)에 사용되는 시퀀스
   - DATA_ROOT/test  : 검증(Validation)에 사용되는 시퀀스
       * 실제 폴더명은 'test' 이지만, 이 스크립트에서는 "Validation" 용도로만 사용
       * 완전히 독립된 최종 Test 셋이 있다면, 별도 스크립트/코드로 평가하는 것을 권장

2) 모델 구조
   - Backbone : ImageNet 사전학습 MobileNetV3-Large (features 전체 freeze)
   - Head     : LSTM + FC(1)
   - Loss     : BCEWithLogitsLoss (이진 분류)

3) 학습/검증 전략
   - Train : DATA_ROOT/train
   - Validation : DATA_ROOT/test (VAL_ROOT, 최종 테스트셋 아님)
   - BalancedBatchSampler 로 train 배치 내
       숫자 시작 seq : 'unknown_' 시작 seq 비율을 numeric_ratio:(1-numeric_ratio) 로 유지
   - Early Stopping : Validation Accuracy 기준

4) Optuna 튜닝
   - 튜닝 대상 : (lr / batch_size / seq_len)
   - sqlite DB에 study 저장 → 재실행 시 이어서 튜닝 가능
   - best trial 하이퍼파라미터로 다시 학습 (train + validation)
     후 모델 가중치(.pth), 학습 로그(.xlsx) 저장
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
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models, transforms
from torchvision.transforms import functional as TF

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import optuna


# =============================================================
# 1. 경로 및 기본 설정 (Path & Global Config)
# =============================================================

# ★★ [중요] 데이터 루트
#   - DATA_ROOT/train : "학습(Train)"에만 사용
#   - DATA_ROOT/test  : "검증(Validation)" 용도로 사용 (이 스크립트에서의 VAL_ROOT)
DATA_ROOT = Path(
    r"C:\user\Processed_data"
)  # 입력 경로 설정

# 학습용(Train) 시퀀스가 들어 있는 폴더
TRAIN_ROOT = DATA_ROOT / "train"

# 검증용(Validation) 시퀀스가 들어 있는 폴더
#   ※ 실제 폴더명은 'test' 이지만, 여기서는 epoch마다 성능을 확인하는
#      "검증(Validation) 셋" 역할을 하므로 변수명은 VAL_ROOT 로 사용
VAL_ROOT = DATA_ROOT / "test"

# seq_norm.pt 사용 여부 (있으면 우선 사용, 없으면 JPG 프레임 로딩)
USE_SEQ_NORM = True

# Optuna 튜닝 기본값 (trial에서 덮어씌움)
DEFAULT_SEQ_LEN    = 10   # 기본 시퀀스 길이
DEFAULT_IMG_SIZE   = 256  # 입력 이미지 크기 (H=W)
DEFAULT_BATCH_SIZE = 16   # 기본 배치 크기
DEFAULT_LR         = 1e-4

NUM_EPOCHS   = 30
WEIGHT_DECAY = 1e-5
RANDOM_SEED  = 42

# 숫자 시작 seq : 'unknown_' 시작 seq 비율 (예: 0.3 → 3:7)
NUMERIC_RATIO = 0.3

# Early Stopping 설정 (Validation Accuracy 기준)
EARLY_STOP_PATIENCE  = 3
EARLY_STOP_MIN_DELTA = 0.0

# Optuna 관련 설정
N_TRIALS   = 27
STUDY_NAME = "mobilenetv3_case3_optuna_freeze_acc"

# Optuna study 를 저장할 sqlite DB 경로
STUDY_DB_PATH = (DATA_ROOT / "optuna_mobilenetv3_case3_tuning_freeze_acc.db").resolve()
STUDY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
storage_url = "sqlite:///" + str(STUDY_DB_PATH).replace("\\", "/")

# 최적 모델 및 엑셀 로그 저장 경로
MODEL_SAVE_PATH = DATA_ROOT / "mobilenetv3_lstm_best_freeze_optuna_acc.pth"  # 가중치 파일 저장하기
EXCEL_LOG_PATH  = DATA_ROOT / "training_log_mobilenetv3_lstm_freeze_optuna_acc.xlsx"

print("DATA_ROOT =", DATA_ROOT)


# =============================================================
# 2. 시퀀스 로딩 유틸 함수 (Sequence Loading Utilities)
# =============================================================

def list_frame_paths(seq_dir: Path) -> List[Path]:
    """Return sorted frame paths in a sequence directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = [p for p in seq_dir.iterdir() if p.suffix.lower() in exts]
    frames.sort()
    return frames


def load_sequence(
    seq_dir: Path,
    use_seq_norm: bool = True,
    seq_len: int = DEFAULT_SEQ_LEN,
    img_size: int = DEFAULT_IMG_SIZE,
) -> torch.Tensor:
    """
    Load one sequence directory as a tensor (T, C, H, W), then:
      - Adjust temporal length to seq_len (crop or pad by last frame).
      - Resize spatial size to (img_size, img_size).

    Loading strategy:
      1) If seq_norm.pt exists and use_seq_norm=True → load that tensor.
      2) Otherwise, load all frame images and normalize by ImageNet mean/std.

    Args:
        seq_dir: Directory of a sequence.
        use_seq_norm: Whether to prefer precomputed seq_norm.pt.
        seq_len: Desired temporal length.
        img_size: Desired spatial size (H, W).

    Returns:
        Tensor of shape (seq_len, C, img_size, img_size).
    """
    seq_norm_path = seq_dir / "seq_norm.pt"

    # Case 1: seq_norm.pt 사용
    if use_seq_norm and seq_norm_path.exists():
        seq = torch.load(seq_norm_path, map_location="cpu")

    # Case 2: 개별 프레임 이미지 로딩
    else:
        frame_paths = list_frame_paths(seq_dir)
        if not frame_paths:
            raise RuntimeError(f"No frames in {seq_dir}")

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            t = TF.to_tensor(img)
            t = normalize(t)
            frames.append(t)

        seq = torch.stack(frames, dim=0)  # (T, C, H, W)

    # ---- 시퀀스 길이 보정 (pad or crop) ----
    T, C, H, W = seq.shape
    if T >= seq_len:
        seq = seq[:seq_len]
    else:
        last = seq[-1:].expand(seq_len - T, C, H, W)
        seq = torch.cat([seq, last], dim=0)

    # ---- 공간 크기 보정 ----
    seq = F_torch.interpolate(
        seq, size=(img_size, img_size),
        mode="bilinear", align_corners=False
    )

    return seq  # (seq_len, C, img_size, img_size)


# =============================================================
# 3. Dataset & Balanced Sampler
# =============================================================

class Case1SequenceDataset(Dataset):
    """
    Sequence dataset for both Train and Validation.

    Given a list of sequence directories and labels, returns:
      - sequence tensor: (T, C, H, W)
      - label: int (0 or 1)
    """
    def __init__(
        self,
        seq_dirs: List[Path],
        labels: List[int],
        use_seq_norm: bool = True,
        seq_len: int = DEFAULT_SEQ_LEN,
        img_size: int = DEFAULT_IMG_SIZE,
    ):
        self.seq_dirs = seq_dirs
        self.labels = labels
        self.use_seq_norm = use_seq_norm
        self.seq_len = seq_len
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.seq_dirs)

    def __getitem__(self, idx: int):
        seq_dir = self.seq_dirs[idx]
        label = self.labels[idx]
        seq = load_sequence(
            seq_dir,
            use_seq_norm=self.use_seq_norm,
            seq_len=self.seq_len,
            img_size=self.img_size,
        )
        return seq, label


class BalancedBatchSampler(Sampler):
    """
    Balanced batch sampler for training.

    It mixes:
      - sequences whose directory names start with numeric (0-9)
      - sequences whose directory names start with 'unknown'

    Ratio per batch is approximately numeric_ratio : (1 - numeric_ratio).

    This sampler should be used only for the training DataLoader.
    Validation DataLoader uses a simple sequential sampler.
    """
    def __init__(
        self,
        numeric_indices: List[int],
        unknown_indices: List[int],
        batch_size: int,
        numeric_ratio: float = 0.3,
    ):
        self.numeric_indices = numeric_indices
        self.unknown_indices = unknown_indices
        self.batch_size = batch_size

        # 한 배치에 들어갈 개수 계산
        self.n_numeric = max(1, int(round(batch_size * numeric_ratio)))
        self.n_unknown = batch_size - self.n_numeric

        # 만들 수 있는 배치 수 (둘 중 더 작은 쪽 기준)
        self.num_batches = min(
            len(self.numeric_indices) // self.n_numeric,
            len(self.unknown_indices) // self.n_unknown,
        )

    def __iter__(self):
        numeric = self.numeric_indices.copy()
        unknown = self.unknown_indices.copy()

        random.shuffle(numeric)
        random.shuffle(unknown)

        n_ptr = 0
        u_ptr = 0

        for _ in range(self.num_batches):
            batch = numeric[n_ptr:n_ptr + self.n_numeric] + \
                    unknown[u_ptr:u_ptr + self.n_unknown]
            random.shuffle(batch)
            n_ptr += self.n_numeric
            u_ptr += self.n_unknown
            yield batch

    def __len__(self) -> int:
        return self.num_batches


# =============================================================
# 4. MobileNetV3-Large + LSTM 모델 정의 (Model Definition)
# =============================================================

class MobileNetV3LSTM(nn.Module):
    """
    MobileNetV3-Large backbone + LSTM head for sequence-level binary classification.

    - Backbone: torchvision mobilenet_v3_large (features only, frozen by default)
    - Time pooling: frame-wise feature → LSTM → temporal average
    - Output: 1D logit per sequence (BCEWithLogitsLoss)
    """
    def __init__(
        self,
        num_classes: int = 1,
        latent_dim: int = 960,
        lstm_layers: int = 1,
        hidden_dim: int = 2048,
        bidirectional: bool = False,
        dropout: float = 0.4,
    ):
        super().__init__()

        base = models.mobilenet_v3_large(pretrained=True)
        self.feature_extractor = base.features

        # -----------------------------------------------------
        # CNN Backbone Freeze
        #   - 현재는 feature_extractor 전체를 freeze (이미지넷 표현만 사용)
        #   - 마지막 block만 학습하고 싶다면 아래 주석 예시 참고
        # -----------------------------------------------------
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 예시) 마지막 block만 unfreeze 하고 싶은 경우
        # last_block = list(self.feature_extractor.children())[-1]
        # for param in last_block.parameters():
        #     param.requires_grad = True

        self.avgpool = nn.AdaptiveAvgPool2d(1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, C, H, W).

        Returns:
            Logits tensor of shape (B, 1).
        """
        B, T, C, H, W = x.shape

        # (B*T, C, H, W) → CNN Backbone
        x = x.view(B * T, C, H, W)
        fmap = self.feature_extractor(x)

        # (B*T, C', H', W') → Global Avg Pool → (B, T, latent_dim)
        x = self.avgpool(fmap)
        x = x.view(B, T, -1)

        # LSTM 통과 후 time dimension 평균
        x_lstm, _ = self.lstm(x)
        x = torch.mean(x_lstm, dim=1)  # (B, hidden_dim)

        x = self.fc(self.dropout(self.relu(x)))  # (B, 1)
        return x


# =============================================================
# 5. Train / Validation 시퀀스 수집 (Sequence Collection)
# =============================================================

def collect_sequences(root: Path):
    """
    Collect sequence directories and labels under the given root.

    Structure:
        root/real/<seq_dirs...> → label 1
        root/fake/<seq_dirs...> → label 0

    Usage:
        - root == TRAIN_ROOT → training sequences
        - root == VAL_ROOT   → validation sequences
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
# 6. 1 Epoch 학습 / 검증 루프 (Train & Eval Loops)
# =============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """
    Train for one epoch on the given DataLoader.

    Args:
        model: Model to train.
        loader: Training DataLoader.
        criterion: Loss function (BCEWithLogitsLoss).
        optimizer: Optimizer (Adam).
        device: Torch device.

    Returns:
        epoch_loss: Average training loss.
        epoch_acc:  Average training accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for seqs, labels in tqdm(loader, desc="Train", leave=False):
        seqs = seqs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        labels_float = labels.float().unsqueeze(1)
        outputs = model(seqs)

        loss = criterion(outputs, labels_float)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * seqs.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds.squeeze(1) == labels.long()).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """
    Evaluate for one epoch on the given DataLoader (Validation).

    No gradient is computed in this step.

    Args:
        model: Model to evaluate.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        epoch_loss: Average validation loss.
        epoch_acc:  Average validation accuracy.
        y_true:     Ground-truth labels (numpy array).
        y_pred:     Predicted labels (numpy array).
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

            labels_float = labels.float().unsqueeze(1)
            outputs = model(seqs)
            loss = criterion(outputs, labels_float)

            running_loss += loss.item() * seqs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds.squeeze(1) == labels.long()).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().squeeze(1).tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(y_true), np.array(y_pred)


# =============================================================
# 7. 하나의 (lr, batch_size, seq_len) 조합으로 전체 학습 수행
#    (Train + Validation Pipeline)
# =============================================================

def run_experiment(
    lr: float,
    batch_size: int,
    seq_len: int,
    num_epochs: int,
    save_artifacts: bool = False,
    model_save_path: Optional[Path] = None,
    excel_log_path: Optional[Path] = None,
) -> float:
    """
    Full training + validation pipeline for a given hyperparameter set.

    Steps:
        1) Collect training sequences from TRAIN_ROOT.
        2) Collect validation sequences from VAL_ROOT.
        3) Build Dataset and DataLoader for both.
        4) Train num_epochs with:
             - train_one_epoch()
             - eval_one_epoch()
           and track best validation accuracy.
        5) Optionally save best model, confusion matrix, classification report,
           and history logs to Excel.

    Args:
        lr: Learning rate.
        batch_size: Batch size.
        seq_len: Temporal length of sequences.
        num_epochs: Maximum number of epochs.
        save_artifacts: Whether to save model and logs.
        model_save_path: Path to save best model weights.
        excel_log_path: Path to save training logs as Excel.

    Returns:
        best_val_acc: Best validation accuracy achieved.
    """
    print(f"\n==== Run experiment | lr={lr:.5e}, batch_size={batch_size}, seq_len={seq_len} ====")

    # 재현성 설정
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # -----------------------------
    # [1] Train / Validation 데이터 수집
    # -----------------------------
    train_dirs, train_labels = collect_sequences(TRAIN_ROOT)  # TRAIN_ROOT → 학습 세트
    val_dirs,   val_labels   = collect_sequences(VAL_ROOT)    # VAL_ROOT  → 검증 세트

    # Dataset 구성
    train_dataset = Case1SequenceDataset(
        train_dirs, train_labels,
        use_seq_norm=USE_SEQ_NORM,
        seq_len=seq_len,
        img_size=DEFAULT_IMG_SIZE,
    )
    val_dataset = Case1SequenceDataset(
        val_dirs, val_labels,
        use_seq_norm=USE_SEQ_NORM,
        seq_len=seq_len,
        img_size=DEFAULT_IMG_SIZE,
    )

    # -----------------------------
    # [2] Train Loader (Balanced Sampler) 구성
    # -----------------------------
    numeric_indices = [
        i for i, d in enumerate(train_dirs)
        if not d.name.startswith("unknown")
    ]
    
    unknown_indices = [
        i for i, d in enumerate(train_dirs)
        if d.name.startswith("unknown")
    ]

    print(f"[INFO] Train numeric dirs : {len(numeric_indices)}")
    print(f"[INFO] Train unknown dirs : {len(unknown_indices)}")

    balanced_sampler = BalancedBatchSampler(
        numeric_indices=numeric_indices,
        unknown_indices=unknown_indices,
        batch_size=batch_size,
        numeric_ratio=NUMERIC_RATIO,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=balanced_sampler,
        num_workers=0,
    )

    # -----------------------------
    # [3] Validation Loader 구성
    # -----------------------------
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # -----------------------------
    # [4] 모델 / 손실함수 / 옵티마이저 설정
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = MobileNetV3LSTM().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    # Early Stopping & Best Model Tracking (Validation 기준)
    best_val_acc = 0.0
    best_epoch = 0
    best_y_true, best_y_pred = None, None
    early_stop_counter = 0

    train_losses, val_losses = [], []
    train_accs,  val_accs  = [], []

    # -----------------------------
    # [5] Epoch 루프 (Train + Validation)
    # -----------------------------
    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{num_epochs} ==========")

        # --- (1) Train ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # --- (2) Validation ---
        val_loss, val_acc, y_true, y_pred = eval_one_epoch(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"[Epoch {epoch}] Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # -----------------------------
        # [6] best validation accuracy 갱신 + Early Stopping
        # -----------------------------
        if val_acc > best_val_acc + EARLY_STOP_MIN_DELTA:
            # 검증 성능이 향상된 경우 → best 갱신
            best_val_acc = val_acc
            best_epoch = epoch
            best_y_true = y_true
            best_y_pred = y_pred
            early_stop_counter = 0

            # best 모델 가중치 저장 (옵션)
            if save_artifacts and model_save_path is not None:
                torch.save(model.state_dict(), model_save_path)
                print(f">>>> Best model saved! epoch={best_epoch}, val_acc={best_val_acc:.4f}")
        else:
            # 검증 성능 향상이 없으면 patience 카운트 증가
            early_stop_counter += 1
            print(f"[EarlyStop] no improvement in val_acc for {early_stop_counter} epoch(s).")
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"[EarlyStop] Stop training at epoch {epoch}.")
                break

    print("\n[Training Finished]")
    print(f"Best Validation Accuracy = {best_val_acc:.4f} at epoch {best_epoch}")

    # ---------------------------------------------------------
    # [7] 결과 아티팩트 저장 (그래프/엑셀/리포트, best epoch 기준)
    # ---------------------------------------------------------
    if save_artifacts:
        # Loss curve
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.legend()
        plt.title("Train / Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.show()

        # Confusion Matrix & Classification Report (best epoch 기준, Validation 셋)
        if best_y_true is not None and best_y_pred is not None:
            cm = confusion_matrix(best_y_true, best_y_pred)
            print("\n[Best Model Evaluation on Validation Set]")
            print("Confusion Matrix:\n", cm)

            report_str = classification_report(best_y_true, best_y_pred, digits=4)
            print("\nClassification Report:\n", report_str)

            # history 로그
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
                columns=["pred_fake(0)", "pred_real(1)"],
            )

            report_dict = classification_report(
                best_y_true,
                best_y_pred,
                output_dict=True,
                digits=4,
            )
            report_df = pd.DataFrame(report_dict).T

            if excel_log_path is not None:
                excel_path_str = str(excel_log_path)
                with pd.ExcelWriter(excel_path_str, engine="openpyxl") as writer:
                    history_df.to_excel(writer, sheet_name="history", index=False)
                    cm_df.to_excel(writer, sheet_name="confusion_matrix")
                    report_df.to_excel(writer, sheet_name="classification_report")

                print(f"\n[LOG] Training log saved to: {excel_path_str}")

    return best_val_acc


# =============================================================
# 8. Optuna Objective & main 진입점 (Hyperparameter Tuning)
# =============================================================

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.

    For each trial, suggest (lr, batch_size, seq_len),
    run training + validation, and return best validation accuracy.
    """
    lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    seq_len = trial.suggest_categorical("seq_len", [8, 10, 12])

    best_val_acc = run_experiment(
        lr=lr,
        batch_size=batch_size,
        seq_len=seq_len,
        num_epochs=NUM_EPOCHS,
        save_artifacts=False,   # 튜닝 단계에서는 파일 저장 X (최종 단계에서만 저장)
        model_save_path=None,
        excel_log_path=None,
    )
    return best_val_acc


def main():
    """
    Main entry point.

    1) Load or create Optuna study (sqlite).
    2) Run remaining trials up to N_TRIALS.
    3) Re-train with best hyperparameters and save artifacts.
    """
    # ---------------------------------------------------------
    # 1) Optuna Study 로드 / 생성 (sqlite 기반, 이어서 튜닝 가능)
    # ---------------------------------------------------------
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
    )

    # 이미 완료된 trial 수
    n_complete = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])

    # 이번 실행에서 추가로 돌릴 trial 수
    remaining_trials = max(0, N_TRIALS - n_complete)

    print(f"\n[Optuna] Completed trials: {n_complete}")
    print(f"[Optuna] Remaining trials this run: {remaining_trials}")

    if remaining_trials > 0:
        # objective() 내부에서 항상 "Train + Validation" 을 돌리고,
        # 반환되는 값은 validation accuracy (VAL_ROOT 기준)
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print("[Optuna] All requested trials already completed. Skipping optimization.")

    # ---------------------------------------------------------
    # 2) 최적 하이퍼파라미터로 최종 학습 + 아티팩트 저장
    # ---------------------------------------------------------
    print("\n===== Optuna Tuning Finished (current DB 상태 기준) =====")
    print("Best trial number :", study.best_trial.number)
    print("Best val_acc      :", study.best_value)
    print("Best params       :", study.best_params)

    best_lr = study.best_params["lr"]
    best_batch_size = study.best_params["batch_size"]
    best_seq_len = study.best_params["seq_len"]

    # 최적 조합으로 다시 학습하면서:
    #  - TRAIN_ROOT : 학습
    #  - VAL_ROOT   : 검증(Validation)
    #  - best 모델 가중치 저장
    #  - Confusion Matrix / Classification Report (Validation 기준) 출력
    #  - history/metrics 엑셀 저장
    run_experiment(
        lr=best_lr,
        batch_size=best_batch_size,
        seq_len=best_seq_len,
        num_epochs=NUM_EPOCHS,
        save_artifacts=True,
        model_save_path=MODEL_SAVE_PATH,
        excel_log_path=EXCEL_LOG_PATH,
    )


if __name__ == "__main__":
    main()
