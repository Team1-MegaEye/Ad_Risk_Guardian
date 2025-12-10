# models.py
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file

from app.utils import split_sentences


# ============================================
# MobileNetV3 + LSTM 구조 (영상 딥페이크 탐지 모델)
# ============================================

IMG_SIZE = 256
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "weights" / "mobilenetv3_lstm_best_freeze_optuna_acc.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MobileNetV3LSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        latent_dim: int = 960,
        lstm_layers: int = 1,
        hidden_dim: int = 2048,
        bidirectional: bool = False,
        dropout: float = 0.4
    ):
        
        super().__init__()

        # MobileNetV3 Large 사전학습 모델 로드
        # features 부분만 사용 (고정된 feature extractor)
        base = models.mobilenet_v3_large(pretrained=True)
        self.feature_extractor = base.features
        
        # -----------------------------------
        # CNN backbone freeze (추론 시에도 그대로 사용)
        # -----------------------------------
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # 마지막 block만 unfreeze 할 수 있으나, inference에는 영향 없음

        # 영상 프레임 feature → 1×1 평균 풀링
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 시간적 연속성을 위한 LSTM
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Fully-connected classifier
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        """
        x: 영상 시퀀스 (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        # (B*T, C, H, W) 형태로 CNN 통과
        x = x.view(B * T, C, H, W)
        fmap = self.feature_extractor(x)

        # (B*T, latent_dim)
        x = self.avgpool(fmap)
        x = x.view(B, T, -1)

        # LSTM으로 temporal feature 처리
        x_lstm, _ = self.lstm(x)

        # 모든 time-step 평균
        x = torch.mean(x_lstm, dim=1)

        # 최종 binary logit 출력
        x = self.fc(self.dropout(self.relu(x)))

        return x


# ============================================
# 영상 모델 로드 함수
# ============================================
def load_trained_model(model_path: Path, device: torch.device):
    """저장된 가중치 파일(.pth)을 로드하여 모델 복원"""
    assert model_path.is_file(), f"모델 파일이 없습니다: {model_path}"

    model = MobileNetV3LSTM()
    ckpt = torch.load(model_path, map_location=device)

    # 다양한 저장 형태 지원
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] 영상 모델 로드 완료")
    return model


# ============================================
# 텍스트 과장광고 분류 모델 (KcELECTRA)
# ============================================

BASE_MODEL_NAME = "beomi/KcELECTRA-base-v2022"
CHECKPOINT_ROOT = "weights/checkpoint"

class TextExaggerationClassifier:
    """
    KcELECTRA 기반 단문 과장광고 분류기
    - 입력: 문장(str)
    - 출력: 정상(1)일 확률
    """

    def __init__(self,
                 checkpoint_root: str = CHECKPOINT_ROOT,
                 base_model_name: str = BASE_MODEL_NAME,
                 device: torch.device = device):

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

        # 사전 fine-tuned 된 모델 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_ROOT)

        self.model.to(device)
        self.model.eval()
        self.device = device

        print(f"[INFO] 텍스트 모델 로드 완료")

    @torch.no_grad()
    def predict_proba(self, texts):
        """
        입력 문장 리스트에 대해 정상 확률 반환
        """
        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.model(**enc)

        # binary classification → sigmoid 적용
        logits = outputs.logits.squeeze(-1)
        probs = torch.sigmoid(logits)

        return probs.cpu().numpy().tolist()

    def predict_label(self, texts, threshold: float = 0.5):
        """
        threshold 기준으로 정상/과장 라벨(1/0) 반환
        """
        probs = self.predict_proba(texts)
        labels = [1 if p >= threshold else 0 for p in probs]
        return labels


# ============================================
# Models 클래스: 영상 + 텍스트 모델 통합
# ============================================
class Models:
    def __init__(self):
        # 1) 영상 딥페이크 모델 로드
        self.model = load_trained_model(MODEL_PATH, device)
        
        # 2) 얼굴 검출용 MTCNN
        self.mtcnn = MTCNN(
            image_size=IMG_SIZE,
            margin=20,
            keep_all=False,
            device=device,
            post_process=False
        )
        
        # 3) 기본 이미지 전처리
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # 4) 디바이스
        self.device = device

        # 5) 텍스트 과장광고 분류 모델
        self.text_classifier = TextExaggerationClassifier()


    # ============================================
    # 영상 예측: 가짜 확률(p_fake) 반환
    # ============================================
    def predict_video(self, frame_dir: str, seq_len: int = 10):
        frame_dir = Path(frame_dir)
        
        # jpg/png 프레임 로드
        frame_paths = sorted(frame_dir.glob("*.jpg")) + \
                      sorted(frame_dir.glob("*.png"))

        if len(frame_paths) == 0:
            raise RuntimeError(f"No frames found in {frame_dir}")

        faces = []

        # 최대 seq_len 프레임만 사용
        for i, frame_path in enumerate(frame_paths[:seq_len]):
            img = Image.open(frame_path).convert("RGB")

            # 얼굴 검출 (없으면 resize fallback)
            with torch.no_grad():
                face = self.mtcnn(img)

            if face is None:
                # 얼굴이 없으면 전체 프레임을 resize하여 사용
                img_resized = img.resize((256, 256))
                face = self.to_tensor(img_resized)

                # 디버깅용 저장
                output_path = frame_dir / f"resized_{i:03d}.jpg"
                img_resized.save(output_path)
            else:
                # mtcnn 출력이 다양한 형태일 수 있음 → Tensor로 정규화
                if isinstance(face, np.ndarray):
                    face = torch.from_numpy(face)
                if isinstance(face, Image.Image):
                    face = self.to_tensor(face)
                if face.max() > 1:
                    face = face / 255.0
                if face.ndim == 4:
                    face = face[0]

                # crop된 얼굴 저장(디버깅)
                if isinstance(face, torch.Tensor):
                    face_img = transforms.ToPILImage()(face.cpu().clamp(0.0, 1.0))
                    output_path = frame_dir / f"face_cropped_{i:03d}.jpg"
                    face_img.save(output_path)

            # Normalize 적용
            face = self.normalize(face)
            faces.append(face)

        # 프레임이 부족하면 마지막 프레임 반복
        while len(faces) < seq_len:
            faces.append(faces[-1].clone())

        faces_tensor = torch.stack(faces, dim=0)          # (T, C, H, W)
        faces_tensor = faces_tensor.unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        # 추론
        with torch.no_grad():
            logits = self.model(faces_tensor)
            p_real = torch.sigmoid(logits).item()
            p_fake = 1 - p_real

        return round(p_fake, 4)


    # ============================================
    # 텍스트 예측: 문장별 과장 확률 등을 포함한 결과 반환
    # ============================================
    def predict_text(self, text: str):
        sentences = split_sentences(text)

        if len(sentences) == 0:
            return {
                "most_exaggerated_sentence": None,
                "exaggeration_prob": 0.0,
                "sentence_probs": []
            }

        # 문장별 정상 확률 계산
        p_normals = self.text_classifier.predict_proba(sentences)

        results = []
        for sent, p_normal in zip(sentences, p_normals):
            p_exag = 1 - p_normal
            results.append((sent, float(p_normal), float(p_exag)))

        # 과장 확률이 가장 높은 문장 선택
        most_exaggerated = max(results, key=lambda x: x[2])

        return {
            "most_exaggerated_sentence": most_exaggerated[0],
            "exaggeration_prob": most_exaggerated[2],
            "sentence_probs": results
        }
        
# 전역 인스턴스 (tasks 및 main에서 import하여 사용)
models = Models()
