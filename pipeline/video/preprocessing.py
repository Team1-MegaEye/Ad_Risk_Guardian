
# -*- coding: utf-8 -*-
"""
KoDF + FaceForensics++를 적용한 전처리 + 증강 + 분석 파이프라인

구성 개요
---------
1) [0] 사용자 환경 설정
   - 입력/출력 루트 경로(IN_ROOT, OUT_ROOT, AN_OUT)와 공통 옵션 설정
     (프레임 수, 이미지 크기, MTCNN 파라미터, 증강/밸런싱/분석 여부 등)

2) [1] 유틸 함수
   - ensure_dir: 디렉터리 생성
   - list_frame_paths: 시퀀스 디렉터리에서 프레임 경로 수집(앞 MAX_FRAMES장만 사용)
   - pil_from_path: Path → RGB PIL 이미지 로드
   - save_tensor_as_images: (T,3,H,W) 텐서를 0001.jpg, 0002.jpg … 형태로 저장
   - normalize_sequence: ImageNet 통계(IMAGENET_MEAN/STD) 기반 프레임별 정규화

3) [2] 얼굴 검출/정렬/크롭 함수
   - detect_align_crop_sequence:
     * 각 프레임에 대해 MTCNN으로 얼굴 검출 + 정렬 + 크롭 + 리사이즈 수행
     * post_process=False로 받아온 [0,255] 텐서를 직접 [0,1]로 스케일링
     * 검출 실패 시 직전 성공 프레임을 재사용, 최초 실패 구간은 중앙 크롭으로 대체
     * 항상 (T,3,H,W) 형태의 얼굴 시퀀스 텐서를 반환(또는 전부 실패 시 None)

4) [3] 시퀀스 증강 유틸 및 SequenceAugmentor
   - uniform / np_random: 증강에 사용하는 난수 래핑 함수
   - SequenceAugmentor:
     * 한 시퀀스 전체에 동일한 랜덤 변환을 적용(temporal consistency 유지)
     * 좌우 플립, 회전, 이동/스케일(affine), 밝기/대비/채도/색상(jitter),
       가우시안 블러, 컷아웃(cutout) 등을 조합하여 증강
     * apply()에서 (T,3,H,W) → 증강된 (T,3,H,W) + 사용된 파라미터 딕셔너리 반환

5) [4] 시퀀스 디렉터리 수집
   - collect_sequences:
     * split_root(예: IN_ROOT/train 또는 IN_ROOT/test) 하위의
       real/<video_id>/, fake/<video_id>/ 디렉터리 목록을 수집
     * real, fake 각각 정렬된 디렉터리 리스트를 반환

6) [5] 공통 전처리 (train/test 공통)
   - process_split_common:
     * real, fake 각각에 대해 video_id 디렉터리를 순회하면서
       list_frame_paths → detect_align_crop_sequence 수행
     * 크롭/정렬/리사이즈된 얼굴 시퀀스를 JPG 프레임(0001.jpg, …)으로 저장
     * SAVE_NORMALIZED=True인 경우, ImageNet 정규화 텐서를 seq_norm.pt로 함께 저장
     * 이 단계에서는 **증강/클래스 밸런싱은 수행하지 않음**

7) [6] Train 전용: 클래스 밸런싱 + 추가 증강
   - balance_train_with_augmentation:
     * Stage 1: 클래스 밸런싱
       - train/real, train/fake 시퀀스 개수를 확인하고,
         개수가 적은 쪽(minority 클래스)만 선택적으로 증강
       - 부족한 개수만큼 증강 시퀀스를 생성하여 real/fake를 1:1로 맞춤
       - 생성 폴더 이름: <원본이름>_balXXX, 증강 메타데이터는 aug_meta.json에 저장
     * Stage 2: multiplier 기반 추가 증강
       - multiplier > 1.0인 경우, 현재 real/fake 개수를 기준으로
         각 클래스를 multiplier 배수만큼 늘리도록 추가 증강
       - 원본 계열 디렉터리( *_aug, *_bal 제외)를 우선 source로 사용
       - 생성 폴더 이름: <원본이름>_augMXXX, seq_norm.pt 및 aug_meta.json 함께 저장

8) [7] 시퀀스를 feature 벡터로 변환
   - sequence_to_feature:
     * seq_norm.pt가 있으면 이를 로드, 없으면 JPG 프레임을 다시 읽어 정규화
     * 시간 축에 대해 평균을 내어 (3,H,W) 형태의 "평균 얼굴 이미지" 생성
     * target_hw×target_hw(기본 64×64)로 리사이즈 후 flatten → 1D feature 벡터 반환
     * PCA / t-SNE 분석에 사용되는 고정 길이 벡터 생성용 함수

9) [8] Train 데이터 분석 (PCA / t-SNE)
   - analyze_train_features:
     * OUT_ROOT/train/real, fake 하위의 시퀀스 폴더들에 대해
       sequence_to_feature로 feature를 추출
     * real/fake 라벨(y_label)과 original/augmented 플래그(y_aug)를 함께 기록
     * PCA 2차원 투영:
       - real/fake & original/augmented를 색/마커로 구분하여 2D 산점도 저장
       - pca_train_orig_vs_aug.png 파일로 저장
     * t-SNE 2차원 투영:
       - (선택적으로 PCA로 차원 축소 후) t-SNE를 수행하고 동일한 방식으로 시각화
       - tsne_train_orig_vs_aug.png 파일로 저장

10) [9] main()
    - 전체 파이프라인 실행 엔트리포인트
      1) 시드 고정(random / numpy / torch)
      2) 출력 폴더 구조 생성 (train/test, real/fake, 분석 출력 폴더)
      3) MTCNN 초기화 (image_size, margin, min_face_size, thresholds 등 설정)
      4) train:
         - process_split_common으로 얼굴 전처리 및 seq_norm.pt 저장
         - AUGMENT_TRAIN & BALANCE_TRAIN이 True이면
           balance_train_with_augmentation으로 클래스 밸런싱 + 추가 증강 수행
      5) test:
         - process_split_common만 수행 (증강/밸런싱 없음)
      6) RUN_ANALYSIS가 True이면
         - analyze_train_features를 호출하여 PCA / t-SNE 분석 및 시각화 수행
"""


import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random
import json

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from facenet_pytorch import MTCNN

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ============================================================
# [0] Global config (paths, options, constants)
#    - Colab / local 환경에 따라 경로만 수정해서 사용
# ============================================================

# 입력 루트: 비디오별로 프레임(예: N장) 추출이 끝난 폴더 구조
IN_ROOT = Path(r"C:\Video_Dataset\Input_data")  # 입력 경로 설정

# 출력 루트: 얼굴 크롭/정규화/증강 결과를 저장할 폴더
OUT_ROOT = Path(r"C:\Video_Dataset\Ouput_data")  # 출력 경로 설정

# 분석 결과(PCA / t-SNE) 저장 루트
AN_OUT = Path(r"C:\Video_Dataset\Ouput_data")  # 시각화 파일 출력 경로 설정

# 사용할 프레임 수 (앞에서부터 MAX_FRAMES장만 사용)
MAX_FRAMES = 10

# 얼굴 패치 크기 및 MTCNN 파라미터
IMAGE_SIZE = 256
MARGIN = 20
MIN_FACE_SIZE = 40
THRESHOLDS = [0.6, 0.7, 0.7]
KEEP_ALL_FACES = False  # 여러 얼굴이 있어도 가장 큰 얼굴만 사용

# Train split 옵션
AUGMENT_TRAIN = True     # train에 증강 적용 여부
BALANCE_TRAIN = True     # train에서 real/fake 1:1 밸런싱 여부
SAVE_NORMALIZED = True   # seq_norm.pt(정규화 텐서) 저장 여부
RUN_ANALYSIS = True      # 전처리 이후 PCA / t-SNE 분석 수행 여부

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# 이미지 확장자 / 정규화 통계
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 전역 시드 고정 (random / numpy)
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# [1] Common utilities
#    - directory utils, frame listing, PIL load, normalization
# ============================================================

def ensure_dir(p: Path) -> None:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    p : Path
        Target directory path. All parent directories will be created
        as needed (parents=True, exist_ok=True).
    """
    p.mkdir(parents=True, exist_ok=True)


def list_frame_paths(seq_dir: Path) -> List[Path]:
    """
    Collect and sort frame image paths under a given sequence directory.

    The function recursively searches for files with valid image extensions
    and returns at most the first ``MAX_FRAMES`` paths in sorted order.

    Parameters
    ----------
    seq_dir : Path
        Directory that contains extracted frames of a video (potentially
        with nested subdirectories).

    Returns
    -------
    List[Path]
        Sorted list of frame paths, truncated to at most ``MAX_FRAMES``
        elements. Returns an empty list if no valid image file is found.
    """
    frames = [
        p for p in seq_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    frames.sort()
    frames = frames[:MAX_FRAMES]
    return frames


def pil_from_path(p: Path) -> Image.Image:
    """
    Load an image file and convert it to an RGB PIL image.

    Parameters
    ----------
    p : Path
        Path to an image file.

    Returns
    -------
    PIL.Image.Image
        Loaded image in RGB mode.
    """
    return Image.open(p).convert("RGB")


def save_tensor_as_images(seq_tensor: torch.Tensor, out_dir: Path) -> None:
    """
    Save a sequence tensor as individual JPG images.

    The input sequence is assumed to be in (T, 3, H, W) format with values
    in [0, 1]. Each frame is saved as zero-padded filenames
    (e.g., ``0001.jpg``, ``0002.jpg``, ...).

    Parameters
    ----------
    seq_tensor : torch.Tensor
        Tensor of shape (T, 3, H, W) with float values in [0, 1].
    out_dir : Path
        Output directory where individual JPG frames will be stored.
        The directory is created if it does not exist.
    """
    ensure_dir(out_dir)
    T = seq_tensor.shape[0]
    for i in range(T):
        img = F.to_pil_image(seq_tensor[i].clamp(0, 1))
        img.save(out_dir / f"{i + 1:04d}.jpg", quality=95)


def normalize_sequence(seq_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply ImageNet normalization to each frame in a sequence.

    The normalization is performed frame-wise using global ImageNet mean
    and standard deviation values (IMAGENET_MEAN / IMAGENET_STD).

    Parameters
    ----------
    seq_tensor : torch.Tensor
        Tensor of shape (T, 3, H, W) with float values in [0, 1].

    Returns
    -------
    torch.Tensor
        Normalized tensor of identical shape and dtype as the input.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    seq_tensor = seq_tensor.clone()
    for i in range(seq_tensor.shape[0]):
        seq_tensor[i] = normalize(seq_tensor[i])
    return seq_tensor


# ============================================================
# [2] Face detection / alignment / cropping (MTCNN)
# ============================================================

@torch.no_grad()
def detect_align_crop_sequence(
    frame_paths: List[Path],
    mtcnn: MTCNN,
    image_size: int
) -> Optional[torch.Tensor]:
    """
    Detect, align, and crop faces for a sequence of frames using MTCNN.

    For each frame, this function applies MTCNN-based face detection and
    alignment, resizes the detected face patch to the specified size, and
    returns a stacked tensor over the temporal dimension. If detection fails
    for a frame:
        * If at least one previous frame was successfully detected, the last
          valid face patch is reused.
        * Otherwise, a center-cropped patch from the original image is used.

    Parameters
    ----------
    frame_paths : list of Path
        List of image paths that represent the video frames in temporal order.
    mtcnn : facenet_pytorch.MTCNN
        Initialized MTCNN model for face detection and alignment.
    image_size : int
        Target spatial size (height and width) of the face patch (e.g., 256).

    Returns
    -------
    Optional[torch.Tensor]
        Tensor of shape (T, 3, H, W) with float values in [0, 1],
        where T is the number of frames successfully processed (same
        as ``len(frame_paths)``). Returns ``None`` if all frames fail.
    """
    if len(frame_paths) == 0:
        return None

    images = [pil_from_path(p) for p in frame_paths]
    aligned = []

    last_valid = None
    for img in images:
        # post_process=False → returns tensor in [0, 255]
        single_aligned = mtcnn(img)
        if single_aligned is not None:
            last_valid = single_aligned.float() / 255.0
            aligned.append(last_valid)
        else:
            # use last valid patch if available
            if last_valid is not None:
                aligned.append(last_valid.clone())
            else:
                # fallback: resize + center crop from original
                fallback = F.resize(img, image_size)
                fallback = F.center_crop(fallback, image_size)
                aligned.append(F.to_tensor(fallback))  # already in [0, 1]

    if not aligned:
        return None

    seq_tensor = torch.stack(aligned, dim=0)  # (T, 3, H, W)
    return seq_tensor


# ============================================================
# [3] Random utility for augmentation (wrapped for clarity)
# ============================================================

def uniform(a: float, b: float) -> float:
    """
    Sample a float uniformly from the interval [a, b].

    This is a thin wrapper around :func:`random.uniform` used for clarity
    in the augmentation code.

    Parameters
    ----------
    a : float
        Lower bound of the sampling range.
    b : float
        Upper bound of the sampling range.

    Returns
    -------
    float
        Random float sampled from [a, b].
    """
    return random.uniform(a, b)


def np_random() -> float:
    """
    Sample a float uniformly from [0, 1).

    This is a thin wrapper around :func:`random.random` used for clarity
    in the augmentation code.

    Returns
    -------
    float
        Random float in [0, 1).
    """
    return random.random()


# ============================================================
# [4] Sequence-level augmentation (train only)
#    - Same random params applied to all frames (temporal consistency)
# ============================================================

class SequenceAugmentor:
    """
    Sequence-level augmentation for video frame tensors.

    This class applies a set of spatial and color augmentations to a sequence
    of frames while keeping the same random parameters across all frames
    within the sequence. This preserves temporal consistency, which is
    critical for video-based models.

    The following transformations are supported:
        * Horizontal flip
        * In-plane rotation
        * Affine translation / scaling
        * Brightness, contrast, saturation, and hue jitter
        * Optional Gaussian blur
        * Optional cutout applied to a common region across all frames
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        rot_deg: float = 8.0,
        jitter_brightness: float = 0.15,
        jitter_contrast: float = 0.15,
        jitter_saturation: float = 0.10,
        jitter_hue: float = 0.02,
        affine_translate: float = 0.02,
        affine_scale: float = 0.05,
        blur_prob: float = 0.3,
        blur_radius: Tuple[float, float] = (0.0, 1.2),
        cutout_prob: float = 0.5,
        cutout_scale: Tuple[float, float] = (0.1, 0.3),
        cutout_fill: float = 0.0,
    ) -> None:
        """
        Initialize augmentation hyperparameters.

        Parameters
        ----------
        hflip_prob : float, optional
            Probability to apply horizontal flipping, by default 0.5.
        rot_deg : float, optional
            Maximum rotation angle in degrees (symmetric range [-rot_deg, rot_deg]),
            by default 8.0.
        jitter_brightness : float, optional
            Brightness jitter factor, by default 0.15.
        jitter_contrast : float, optional
            Contrast jitter factor, by default 0.15.
        jitter_saturation : float, optional
            Saturation jitter factor, by default 0.10.
        jitter_hue : float, optional
            Hue jitter factor, by default 0.02.
        affine_translate : float, optional
            Maximum translation as a fraction of image size, by default 0.02.
        affine_scale : float, optional
            Maximum scaling factor deviation from 1.0, by default 0.05.
        blur_prob : float, optional
            Probability to apply Gaussian blur, by default 0.3.
        blur_radius : tuple of float, optional
            Range of Gaussian blur radius, by default (0.0, 1.2).
        cutout_prob : float, optional
            Probability to apply cutout, by default 0.5.
        cutout_scale : tuple of float, optional
            Range of cutout patch size as fraction of image size, by default (0.1, 0.3).
        cutout_fill : float, optional
            Fill value for the cutout region (in tensor value range), by default 0.0.
        """
        self.hflip_prob = hflip_prob
        self.rot_deg = rot_deg
        self.jitter_brightness = jitter_brightness
        self.jitter_contrast = jitter_contrast
        self.jitter_saturation = jitter_saturation
        self.jitter_hue = jitter_hue
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale
        self.blur_prob = blur_prob
        self.blur_radius = blur_radius
        self.cutout_prob = cutout_prob
        self.cutout_scale = cutout_scale
        self.cutout_fill = cutout_fill

    def _sample_params(self, W: int, H: int) -> Dict:
        """
        Sample a set of random augmentation parameters for one sequence.

        The sampled parameters are reused for all frames in the sequence
        to preserve temporal consistency.

        Parameters
        ----------
        W : int
            Width of the frame (in pixels).
        H : int
            Height of the frame (in pixels).

        Returns
        -------
        dict
            Dictionary of sampled augmentation parameters.
        """
        return {
            "flip": np_random() < self.hflip_prob,
            "angle": uniform(-self.rot_deg, self.rot_deg),
            "scale": 1.0 + uniform(-self.affine_scale, self.affine_scale),
            "translate": (
                uniform(-W * self.affine_translate, W * self.affine_translate),
                uniform(-H * self.affine_translate, H * self.affine_translate),
            ),
            "brightness": 1.0 + uniform(-self.jitter_brightness, self.jitter_brightness),
            "contrast": 1.0 + uniform(-self.jitter_contrast, self.jitter_contrast),
            "saturation": 1.0 + uniform(-self.jitter_saturation, self.jitter_saturation),
            "hue": uniform(-self.jitter_hue, self.jitter_hue),
            "use_blur": np_random() < self.blur_prob,
            "blur_radius": uniform(self.blur_radius[0], self.blur_radius[1]),
        }

    def apply(self, seq_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Apply sequence-level augmentation to a given frame tensor sequence.

        All frames within the sequence share the same sampled augmentation
        parameters. The function supports geometric transforms, color jitter,
        optional Gaussian blur, and optional cutout.

        Parameters
        ----------
        seq_tensor : torch.Tensor
            Input sequence tensor of shape (T, 3, H, W) with values in [0, 1].

        Returns
        -------
        Tuple[torch.Tensor, dict]
            - Augmented sequence tensor of shape (T, 3, H, W).
            - Dictionary of augmentation parameters used for this sequence.
        """
        T, C, H, W = seq_tensor.shape
        params = self._sample_params(W, H)
        out = []

        # 1) geometric + color transforms
        for t in range(T):
            img = F.to_pil_image(seq_tensor[t].clamp(0, 1))

            if params["flip"]:
                img = F.hflip(img)

            img = F.rotate(
                img,
                angle=params["angle"],
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            )
            img = F.affine(
                img,
                angle=0.0,
                translate=(
                    int(params["translate"][0]),
                    int(params["translate"][1]),
                ),
                scale=params["scale"],
                shear=(0.0, 0.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            )

            img = F.adjust_brightness(img, params["brightness"])
            img = F.adjust_contrast(img, params["contrast"])
            img = F.adjust_saturation(img, params["saturation"])
            img = F.adjust_hue(img, params["hue"])

            if params["use_blur"] and params["blur_radius"] > 1e-6:
                img = img.filter(ImageFilter.GaussianBlur(radius=params["blur_radius"]))

            out.append(F.to_tensor(img))

        # 2) Cutout (same region across all frames)
        if np.random.rand() < self.cutout_prob:
            mask_ratio = uniform(self.cutout_scale[0], self.cutout_scale[1])
            cut_w = int(W * mask_ratio)
            cut_h = int(H * mask_ratio)

            cx = random.randint(0, max(0, W - cut_w))
            cy = random.randint(0, max(0, H - cut_h))

            for t in range(T):
                img_np = out[t].clone().numpy()
                img_np[:, cy:cy + cut_h, cx:cx + cut_w] = self.cutout_fill
                out[t] = torch.tensor(img_np, dtype=out[t].dtype)

        return torch.stack(out, dim=0), params


# ============================================================
# [5] Collect sequence directories (real / fake)
# ============================================================

def collect_sequences(split_root: Path) -> Tuple[List[Path], List[Path]]:
    """
    Collect real/fake sequence directories under a given split root.

    The expected directory structure is::

        split_root/
            real/<video_id>/
            fake/<video_id>/

    Parameters
    ----------
    split_root : Path
        Path to a split root (e.g., ``IN_ROOT / "train"`` or
        ``IN_ROOT / "test"``).

    Returns
    -------
    tuple of list of Path
        Two lists: (real_dirs, fake_dirs), each containing paths to
        sequence directories. Lists are sorted lexicographically.
    """
    real_dir = split_root / "real"
    fake_dir = split_root / "fake"

    reals = [d for d in real_dir.iterdir() if d.is_dir()] if real_dir.exists() else []
    fakes = [d for d in fake_dir.iterdir() if d.is_dir()] if fake_dir.exists() else []

    reals.sort()
    fakes.sort()
    return reals, fakes


# ============================================================
# [6] Common preprocessing for train / test
#    - face align + crop + resize + [0,1] save
#    - optionally save normalized sequence tensor
# ============================================================

def process_split_common(
    in_split_root: Path,
    out_split_root: Path,
    mtcnn: MTCNN,
    image_size: int,
    save_norm: bool,
) -> None:
    """
    Perform common preprocessing for a given split (train or test).

    For each sequence under ``in_split_root/real`` and ``in_split_root/fake``,
    this function:
        1. Collects up to ``MAX_FRAMES`` frame paths.
        2. Runs face detection + alignment + cropping via MTCNN.
        3. Saves the cropped faces as individual JPG frames.
        4. Optionally saves an ImageNet-normalized tensor (``seq_norm.pt``).

    Parameters
    ----------
    in_split_root : Path
        Input split root (e.g., ``IN_ROOT / "train"`` or ``IN_ROOT / "test"``).
    out_split_root : Path
        Output split root (e.g., ``OUT_ROOT / "train"`` or ``OUT_ROOT / "test"``).
    mtcnn : facenet_pytorch.MTCNN
        Initialized MTCNN model for face detection and alignment.
    image_size : int
        Target spatial size of the face patch (height and width).
    save_norm : bool
        If True, save an additional ``seq_norm.pt`` file containing
        ImageNet-normalized tensors for each sequence.
    """
    real_dirs, fake_dirs = collect_sequences(in_split_root)

    for label, seq_dirs in [("real", real_dirs), ("fake", fake_dirs)]:
        for seq_dir in tqdm(seq_dirs, desc=f"[{in_split_root.name}] {label} (crop+resize)"):
            out_seq_dir = out_split_root / label / seq_dir.name

            # Skip if already processed (first frame exists)
            if (out_seq_dir / "0001.jpg").exists():
                continue

            frame_paths = list_frame_paths(seq_dir)
            if not frame_paths:
                print(f"[SKIP] {label} {seq_dir} : no frames found")
                continue

            seq = detect_align_crop_sequence(
                frame_paths,
                mtcnn=mtcnn,
                image_size=image_size,
            )
            if seq is None:
                print(f"[SKIP] {label} {seq_dir} : detect_align_crop_sequence() returned None")
                continue

            save_tensor_as_images(seq, out_seq_dir)

            if save_norm:
                norm = normalize_sequence(seq)
                torch.save(norm, out_seq_dir / "seq_norm.pt")


# ============================================================
# [7] Train-only: class balancing + multiplier-based augmentation
# ============================================================

def balance_train_with_augmentation(
    out_train_root: Path,
    seed: int = 42,
    multiplier: float = 2.0,
) -> None:
    """
    Apply class balancing and multiplier-based augmentation to the train split.

    This function is intended for **train** data only. It operates in two stages:

    Stage 1 (Class balancing)
        * Counts base (original) sequences for real and fake classes
          (excluding ``*_aug*`` and ``*_bal*`` directories).
        * If the class sizes differ, it augments the minority class by
          creating additional sequences until the base counts match.
        * Augmented sequences are saved as ``<basename>_balXXX`` and their
          metadata is stored in ``aug_meta.json``.
        * If any ``*_bal*`` directory already exists, Stage 1 is assumed to
          be completed and is skipped on subsequent runs.

    Stage 2 (Multiplier augmentation)
        * If ``multiplier > 1.0``, additionally augments each class so that
          the total number of sequences becomes approximately
          ``multiplier × current_count``.
        * Newly generated sequences are named ``<basename>_augMXXX`` and
          also include ``seq_norm.pt`` and ``aug_meta.json``.

    Parameters
    ----------
    out_train_root : Path
        Output train root directory (e.g., ``OUT_ROOT / "train"``).
    seed : int, optional
        Random seed for reproducibility of augmentation, by default 42.
    multiplier : float, optional
        Target multiplier for Stage 2 augmentation (e.g., 2.0 → double),
        by default 2.0.
    """
    random.seed(seed)
    np.random.seed(seed)

    real_root = out_train_root / "real"
    fake_root = out_train_root / "fake"

    if not real_root.exists() or not fake_root.exists():
        print("[WARN] Train output structure is invalid, skip balancing/augmentation.")
        return

    # 모든 디렉터리 수집
    real_all = [d for d in real_root.iterdir() if d.is_dir()]
    fake_all = [d for d in fake_root.iterdir() if d.is_dir()]

    # base 디렉터리 정의: *_aug, *_bal 이 아닌 것들 → 원본 시퀀스
    def is_base_dir(d: Path) -> bool:
        """
        Check whether a directory corresponds to a base (original) sequence.

        A directory is considered base if its name does not contain
        ``"_aug"`` or ``"_bal"``.
        """
        name = d.name
        return ("_aug" not in name) and ("_bal" not in name)

    real_base = [d for d in real_all if is_base_dir(d)]
    fake_base = [d for d in fake_all if is_base_dir(d)]

    n_real_base = len(real_base)
    n_fake_base = len(fake_base)

    if n_real_base == 0 or n_fake_base == 0:
        print(f"[WARN] Train: one base class is empty (real_base={n_real_base}, fake_base={n_fake_base}), skip augmentation.")
        return

    augmentor = SequenceAugmentor()

    # ---------- Stage 1: base 기준 클래스 밸런싱 (한 번만 실행) ----------
    has_balanced_once = any("_bal" in d.name for d in (real_all + fake_all))

    if not has_balanced_once and (n_real_base != n_fake_base):
        minority_label = "real" if n_real_base < n_fake_base else "fake"
        majority_n = max(n_real_base, n_fake_base)
        minority_dirs = real_base if minority_label == "real" else fake_base
        aug_needed = majority_n - len(minority_dirs)

        print(f"[INFO] Stage 1: balancing class '{minority_label}' with {aug_needed} extra sequences (base only).")

        created = 0
        idx = 0
        while created < aug_needed:
            base_dir = minority_dirs[idx % len(minority_dirs)]
            frames = list_frame_paths(base_dir)
            if not frames:
                idx += 1
                continue

            seq = torch.stack(
                [F.to_tensor(pil_from_path(fp)) for fp in frames],
                dim=0,
            )
            aug_seq, aug_params = augmentor.apply(seq)

            out_aug_dir = base_dir.parent / f"{base_dir.name}_bal{created + 1:03d}"
            if not out_aug_dir.exists():
                save_tensor_as_images(aug_seq, out_aug_dir)
                if SAVE_NORMALIZED:
                    torch.save(
                        normalize_sequence(aug_seq),
                        out_aug_dir / "seq_norm.pt",
                    )

                meta = {
                    "source_seq": base_dir.name,
                    "label": base_dir.parent.name,  # "real" or "fake"
                    "split": "train",
                    "mode": "balance",
                    "params": {**aug_params, "source_path": str(base_dir)},
                }
                with open(out_aug_dir / "aug_meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                created += 1

            idx += 1

        # Stage 1 이후 한 번 더 전체 디렉터리 재수집
        real_all = [d for d in real_root.iterdir() if d.is_dir()]
        fake_all = [d for d in fake_root.iterdir() if d.is_dir()]
    else:
        if has_balanced_once:
            print("[INFO] Stage 1: *_bal 디렉터리가 이미 존재하여 재실행을 생략합니다.")
        else:
            print("[INFO] Stage 1: base real/fake already balanced, skip.")

    n_real_all = len(real_all)
    n_fake_all = len(fake_all)
    print(f"[INFO] After Stage 1 → real={n_real_all}, fake={n_fake_all}")

    # ---------- Stage 2: multiplier-based augmentation ----------
    if multiplier <= 1.0:
        print("[INFO] multiplier <= 1.0, skip Stage 2 augmentation.")
        return

    # Stage 2는 현재 전체 개수를 기준으로 multiplier 배 확대
    n_real_current = len([d for d in real_root.iterdir() if d.is_dir()])
    n_fake_current = len([d for d in fake_root.iterdir() if d.is_dir()])

    print(f"[INFO] Stage 2: multiplier={multiplier}")
    print(f"       real: current {n_real_current}")
    print(f"       fake: current {n_fake_current}")

    add_real = int(round(n_real_current * (multiplier - 1.0)))
    add_fake = int(round(n_fake_current * (multiplier - 1.0)))

    print(f"       real: add {add_real}")
    print(f"       fake: add {add_fake}")

    # Stage 2 source: base 디렉터리 우선 사용 (없으면 전체 사용)
    base_real_dirs = [d for d in real_root.iterdir() if d.is_dir() and is_base_dir(d)]
    base_fake_dirs = [d for d in fake_root.iterdir() if d.is_dir() and is_base_dir(d)]

    if not base_real_dirs:
        base_real_dirs = [d for d in real_root.iterdir() if d.is_dir()]
    if not base_fake_dirs:
        base_fake_dirs = [d for d in fake_root.iterdir() if d.is_dir()]

    # --- augment real ---
    created = 0
    idx = 0
    while created < add_real and base_real_dirs:
        base_dir = base_real_dirs[idx % len(base_real_dirs)]
        frames = list_frame_paths(base_dir)
        if not frames:
            idx += 1
            continue

        seq = torch.stack(
            [F.to_tensor(pil_from_path(fp)) for fp in frames],
            dim=0,
        )
        aug_seq, aug_params = augmentor.apply(seq)

        out_aug_dir = real_root / f"{base_dir.name}_augM{created + 1:03d}"
        if not out_aug_dir.exists():
            save_tensor_as_images(aug_seq, out_aug_dir)
            if SAVE_NORMALIZED:
                torch.save(
                    normalize_sequence(aug_seq),
                    out_aug_dir / "seq_norm.pt",
                )

            meta = {
                "source_seq": base_dir.name,
                "label": "real",
                "split": "train",
                "mode": "multiplier",
                "params": {**aug_params, "source_path": str(base_dir)},
            }
            with open(out_aug_dir / "aug_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            created += 1
        idx += 1

    # --- augment fake ---
    created = 0
    idx = 0
    while created < add_fake and base_fake_dirs:
        base_dir = base_fake_dirs[idx % len(base_fake_dirs)]
        frames = list_frame_paths(base_dir)
        if not frames:
            idx += 1
            continue

        seq = torch.stack(
            [F.to_tensor(pil_from_path(fp)) for fp in frames],
            dim=0,
        )
        aug_seq, aug_params = augmentor.apply(seq)

        out_aug_dir = fake_root / f"{base_dir.name}_augM{created + 1:03d}"
        if not out_aug_dir.exists():
            save_tensor_as_images(aug_seq, out_aug_dir)
            if SAVE_NORMALIZED:
                torch.save(
                    normalize_sequence(aug_seq),
                    out_aug_dir / "seq_norm.pt",
                )

            meta = {
                "source_seq": base_dir.name,
                "label": "fake",
                "split": "train",
                "mode": "multiplier",
                "params": {**aug_params, "source_path": str(base_dir)},
            }
            with open(out_aug_dir / "aug_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            created += 1
        idx += 1

    n_real_final = len([d for d in real_root.iterdir() if d.is_dir()])
    n_fake_final = len([d for d in fake_root.iterdir() if d.is_dir()])
    print(f"[DONE] Final train counts → real={n_real_final}, fake={n_fake_final}")


# ============================================================
# [8] Feature extraction for analysis (PCA / t-SNE)
# ============================================================

def sequence_to_feature(seq_dir: Path, target_hw: int = 64) -> Optional[np.ndarray]:
    """
    Convert a sequence directory into a fixed-length feature vector.

    The function performs the following steps:
        1. Load ``seq_norm.pt`` if it exists; otherwise, load JPG frames and
           apply ImageNet normalization.
        2. Compute the temporal mean over frames → (3, H, W) "mean face".
        3. Normalize pixel values to [0, 1] for visualization convenience.
        4. Resize the mean face to ``(target_hw, target_hw)``.
        5. Flatten the tensor into a 1D feature vector.

    Parameters
    ----------
    seq_dir : Path
        Path to a sequence directory.
    target_hw : int, optional
        Target height and width for the resized mean face, by default 64.

    Returns
    -------
    Optional[np.ndarray]
        1D float32 NumPy array representing the sequence feature.
        Returns ``None`` if loading fails or sequence tensor has invalid shape.
    """
    seq_norm_path = seq_dir / "seq_norm.pt"
    if seq_norm_path.exists():
        try:
            seq = torch.load(seq_norm_path, map_location="cpu")  # (T, 3, H, W)
        except Exception as e:
            print(f"[WARN] Failed to load {seq_norm_path}: {e}")
            return None
    else:
        frame_paths = list_frame_paths(seq_dir)
        if not frame_paths:
            return None
        frames = [F.to_tensor(pil_from_path(fp)) for fp in frame_paths]
        seq = torch.stack(frames, dim=0)
        seq = normalize_sequence(seq)

    if seq.ndim != 4 or seq.shape[1] != 3:
        print(f"[WARN] Invalid sequence shape at {seq_dir}, shape={seq.shape}")
        return None

    mean_img = seq.mean(dim=0)  # (3, H, W)

    # Normalize to [0, 1] for visualization-friendly PCA/TSNE
    mean_img_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min() + 1e-8)
    mean_img_pil = F.to_pil_image(mean_img_norm)
    mean_img_resized = F.resize(mean_img_pil, (target_hw, target_hw))
    mean_img_tensor = F.to_tensor(mean_img_resized)  # (3, target_hw, target_hw)

    feat = mean_img_tensor.view(-1).numpy().astype(np.float32)
    return feat


def analyze_train_features(
    out_train_root: Path,
    an_out_root: Path,
    max_samples_per_class: Optional[int] = None,
) -> None:
    """
    Run PCA and t-SNE analysis on train features for visualization.

    This function extracts features from all sequences under
    ``out_train_root/real`` and ``out_train_root/fake``, and visualizes
    the distribution of:
        * class labels: real vs fake
        * augmentation status: original vs augmented

    PCA is used for an initial 2D projection, and t-SNE is applied
    on a PCA-reduced space to obtain a non-linear 2D embedding.

    Parameters
    ----------
    out_train_root : Path
        Train root directory containing ``real`` and ``fake`` subdirectories.
    an_out_root : Path
        Output directory where PCA and t-SNE plots will be saved.
    max_samples_per_class : int or None, optional
        Maximum number of original and augmented sequences to sample per class.
        If ``None``, all available sequences are used.
    """
    ensure_dir(an_out_root)

    X_list = []
    y_label = []  # "real" / "fake"
    y_aug = []    # "original" / "augmented"

    for label in ["real", "fake"]:
        class_dir = out_train_root / label
        if not class_dir.exists():
            continue

        all_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
        orig_dirs = [d for d in all_dirs if "_aug" not in d.name]
        aug_dirs = [d for d in all_dirs if "_aug" in d.name]

        if max_samples_per_class is not None:
            random.shuffle(orig_dirs)
            random.shuffle(aug_dirs)
            orig_dirs = orig_dirs[:max_samples_per_class]
            aug_dirs = aug_dirs[:max_samples_per_class]

        print(f"[ANALYSIS] {label}: original={len(orig_dirs)}, augmented={len(aug_dirs)}")

        # original
        for d in orig_dirs:
            feat = sequence_to_feature(d)
            if feat is None:
                continue
            X_list.append(feat)
            y_label.append(label)
            y_aug.append("original")

        # augmented
        for d in aug_dirs:
            feat = sequence_to_feature(d)
            if feat is None:
                continue
            X_list.append(feat)
            y_label.append(label)
            y_aug.append("augmented")

    if not X_list:
        print("[ANALYSIS] No features to analyze.")
        return

    X = np.stack(X_list, axis=0)  # (N, D)
    y_label = np.array(y_label)
    y_aug = np.array(y_aug)

    print(f"[ANALYSIS] Total samples: {X.shape[0]}, feature dim: {X.shape[1]}")

    # ---------- PCA ----------
    print("[ANALYSIS] Running PCA...")
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for lbl, color in [("real", "tab:blue"), ("fake", "tab:red")]:
        for aug_flag, marker in [("original", "o"), ("augmented", "x")]:
            mask = (y_label == lbl) & (y_aug == aug_flag)
            if not np.any(mask):
                continue
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                label=f"{lbl}-{aug_flag}",
                alpha=0.6,
                s=30,
                marker=marker,
                color=color,
            )
    plt.title("PCA (train, real/fake & original/augmented)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    pca_path = an_out_root / "pca_train_orig_vs_aug.png"
    plt.savefig(pca_path, dpi=200)
    plt.close()
    print(f"[ANALYSIS] Saved PCA plot: {pca_path}")

    # ---------- t-SNE ----------
    print("[ANALYSIS] Running t-SNE (using all samples)...")

    # (정석) 고차원 → PCA(<=50차원) → t-SNE
    tsne_dim = min(50, X.shape[1])
    pca_tsne = PCA(n_components=tsne_dim, random_state=SEED)
    X_tsne_in = pca_tsne.fit_transform(X)

    tsne = TSNE(
        n_components=2,
        random_state=SEED,
        init="pca",
        learning_rate="auto",
    )
    X_tsne = tsne.fit_transform(X_tsne_in)

    plt.figure(figsize=(8, 6))
    for lbl, color in [("real", "tab:blue"), ("fake", "tab:red")]:
        for aug_flag, marker in [("original", "o"), ("augmented", "x")]:
            mask = (y_label == lbl) & (y_aug == aug_flag)
            if not np.any(mask):
                continue
            plt.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                label=f"{lbl}-{aug_flag}",
                alpha=0.6,
                s=30,
                marker=marker,
                color=color,
            )
    plt.title("t-SNE (train, real/fake & original/augmented)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.tight_layout()
    tsne_path = an_out_root / "tsne_train_orig_vs_aug.png"
    plt.savefig(tsne_path, dpi=200)
    plt.close()
    print(f"[ANALYSIS] Saved t-SNE plot: {tsne_path}")


# ============================================================
# [9] Main entry
# ============================================================

def main() -> None:
    """
    End-to-end preprocessing, augmentation, and analysis pipeline.

    This is the main entry point of the script. It performs:

    1. Fix random seeds for Python, NumPy, and PyTorch.
    2. Prepare the output directory structure:
       ``OUT_ROOT / {train,test} / {real,fake}`` and analysis output.
    3. Initialize an MTCNN instance with the configured hyperparameters.
    4. Train split:
       * Run ``process_split_common`` for face detection / alignment / cropping.
       * If ``AUGMENT_TRAIN`` and ``BALANCE_TRAIN`` are True, run
         ``balance_train_with_augmentation`` to apply class balancing and
         additional augmentation.
    5. Test split:
       * Run ``process_split_common`` only (no augmentation / balancing).
    6. If ``RUN_ANALYSIS`` is True:
       * Run ``analyze_train_features`` to compute PCA and t-SNE visualizations.
    """
    # ---- fix seeds (including PyTorch) ----
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ---- prepare output structure ----
    ensure_dir(OUT_ROOT / "train" / "real")
    ensure_dir(OUT_ROOT / "train" / "fake")
    ensure_dir(OUT_ROOT / "test" / "real")
    ensure_dir(OUT_ROOT / "test" / "fake")
    ensure_dir(AN_OUT)

    # ---- init MTCNN ----
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=MARGIN,
        min_face_size=MIN_FACE_SIZE,
        thresholds=THRESHOLDS,
        post_process=False,      # we manually scale to [0, 1]
        keep_all=KEEP_ALL_FACES,
        device=DEVICE,
        select_largest=True,     # select the largest face if multiple
    )

    # ---------- Train: common preprocessing ----------
    in_train = IN_ROOT / "train"
    if in_train.exists():
        out_train = OUT_ROOT / "train"
        process_split_common(
            in_split_root=in_train,
            out_split_root=out_train,
            mtcnn=mtcnn,
            image_size=IMAGE_SIZE,
            save_norm=SAVE_NORMALIZED,
        )

        if AUGMENT_TRAIN and BALANCE_TRAIN:
            balance_train_with_augmentation(
                out_train_root=out_train,
                seed=SEED,
                multiplier=2.0,
            )
    else:
        print(f"[WARN] Train input folder does not exist: {in_train}")

    # ---------- Test: common preprocessing only ----------
    in_test = IN_ROOT / "test"
    if in_test.exists():
        out_test = OUT_ROOT / "test"
        process_split_common(
            in_split_root=in_test,
            out_split_root=out_test,
            mtcnn=mtcnn,
            image_size=IMAGE_SIZE,
            save_norm=SAVE_NORMALIZED,
        )
    else:
        print(f"[WARN] Test input folder does not exist: {in_test}")

    # ---------- Train feature analysis (PCA / t-SNE) ----------
    if RUN_ANALYSIS:
        print("[ANALYSIS] Start train feature analysis...")
        analyze_train_features(
            out_train_root=OUT_ROOT / "train",
            an_out_root=AN_OUT,
        )

    # print("[ALL DONE] Train(preprocess+normalize+augment/balance+analysis), "
    #       "Test(preprocess+normalize) complete.")


if __name__ == "__main__":
    main()
