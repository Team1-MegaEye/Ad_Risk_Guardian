# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 23:59:05 2025

@author: deoha
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 02:51:59 2025

@author: deoha
"""

# -*- coding: utf-8 -*-
"""

이 스크립트는 collect.py 에서 생성한
FaceForensics++(FFPP) + KoDF split 데이터셋을 대상으로 하는
기초 EDA(Exploratory Data Analysis)를 수행합니다.

주요 기능
---------
1) FPS 분포 분석
   - 각 영상의 FPS를 수집하여 히스토그램으로 시각화
   - 각 bin 위에 개수(count)를 숫자로 표기
   - 가장 많이 등장한 FPS를 기준(base_fps)으로 선택

2) 기준(base) FPS 기반 프레임 샘플링
   - base_fps 기준으로 각 영상의 "초반 구간"에서
     동일 시간 간격으로 일정 개수(max_frames_per_video) 프레임 샘플링
   - 샘플링된 프레임들을 그레이스케일로 변환 후,
     전체 픽셀 밝기 히스토그램을 누적하여 분석
   - (frame_means 리스트에 프레임별 평균 밝기도 저장하여
      추가적인 boxplot / 통계 분석에 활용 가능)

3) 해상도 분포 분석
   - 각 영상의 (Width x Height)를 수집하여
     가장 많이 등장하는 Top-20 조합을 bar plot으로 시각화
   - 각 bar 위에 빈도수를 숫자로 표기

전제 조건
---------
- collect.py 에서 CONFIG_FFPP, CONFIG_KODF, iter_videos 를 import 가능해야 합니다.
- OpenCV(cv2), matplotlib, numpy 등이 설치되어 있어야 합니다.
"""

from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt

# collect.py에서 CONFIG_FFPP, CONFIG_KODF, iter_videos 만 재사용
from collect import CONFIG_FFPP, CONFIG_KODF, iter_videos



# =========================================================
# 1. 비디오 경로 수집 유틸
# =========================================================
def collect_all_videos_for_analysis(roots: list[Path]):
    """
    Collect all video file paths under the given dataset root directories.

    Parameters
    ----------
    roots : list[Path]
        Dataset root directories to search for video files.
        예) FFPP split 결과 폴더, KoDF split 결과 폴더 등.

    Returns
    -------
    all_videos : list[Path]
        주어진 모든 루트 폴더 하위의 비디오 파일 경로 리스트.
    """
    all_videos: list[Path] = []

    for r in roots:
        if not r.exists():
            # 설정이 잘못되었거나, 특정 데이터셋이 준비되지 않은 경우를 대비
            continue

        # collect.py 에서 정의된 iter_videos 사용
        # - VIDEO_EXTS(확장자 필터)에 맞는 영상만 순회
        for p in iter_videos(r):
            all_videos.append(p)

    print(f"[ANALYSIS] 총 영상 개수: {len(all_videos)}")
    return all_videos


# =========================================================
# 2. FPS / 픽셀 밝기 / 해상도 분포 분석 함수
# =========================================================
def analyze_fps_and_pixel_distribution(
    video_paths,
    max_frames_per_video: int = 10,
    freeze_to_max_fps: bool = True,
):
    """
    Analyze FPS distribution, pixel intensity distribution, and resolution.

    이 함수는 다음과 같은 분석을 수행합니다.

    1) FPS 분석
       - 각 영상의 FPS를 수집하여 히스토그램/boxplot으로 시각화
       - 가장 자주 등장하는 FPS를 base_fps로 선정

    2) 픽셀 밝기 분포 분석
       - base_fps 기준으로, 각 영상에서 초기 구간의 프레임들을
         동일 시간 간격으로 max_frames_per_video장씩 샘플링
       - 샘플링된 모든 프레임을 그레이스케일로 변환하여
         픽셀 값(0~255) 히스토그램을 누적(hist_pixels 배열)

    3) 해상도 분포 분석
       - 각 영상의 (width x height) 문자열을 수집하여
         가장 자주 등장하는 상위 20개를 bar plot으로 시각화

    Parameters
    ----------
    video_paths : list[Path] or list[str]
        분석 대상 비디오 파일 경로 리스트.

    max_frames_per_video : int, optional
        각 영상에서 샘플링할 최대 프레임 수.
        기본값은 10.

    freeze_to_max_fps : bool, optional
        True인 경우, 전체 데이터셋에서 가장 빈도가 높은 FPS(base_fps)를 기준으로
        각 영상의 FPS를 보정하는 형태로 프레임 인덱스를 설정합니다.
        False인 경우, 단순히 전체 프레임 수 기준으로 균등 간격 샘플링을 수행합니다.
    """
    # -----------------------------
    # 2.1. FPS / 해상도 / 픽셀 통계용 변수 준비
    # -----------------------------
    fps_list = []          # 원본 FPS(real-valued) 수집
    fps_map = {}           # 각 영상 경로 -> FPS 매핑
    widths = []            # 프레임 width 수집
    heights = []           # 프레임 height 수집
    resolutions = []       # "WxH" 문자열 리스트
    hist_pixels = np.zeros(256, dtype=np.int64)  # 0~255 픽셀 히스토그램 누적

    # 프레임별 평균 픽셀 값(밝기) 저장 리스트 (추가 분석/boxplot용)
    frame_means = []

    # FPS boxplot 계산을 위한 정수형(rounded) FPS 리스트
    fps_rounded = []

    # -----------------------------
    # 2.2. 1차 패스: FPS / 해상도 정보 수집
    # -----------------------------
    for vp in video_paths:
        vp_str = str(vp)

        # OpenCV VideoCapture를 통해 영상 메타데이터 읽기
        cap = cv2.VideoCapture(vp_str)
        if not cap.isOpened():
            # 깨진 파일 or 코덱 문제 등으로 열리지 않을 경우 스킵
            continue

        # FPS 읽기
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            fps_list.append(fps)
            fps_map[vp_str] = fps
        else:
            # FPS를 읽지 못한 경우 None으로 표기
            fps_map[vp_str] = None

        # 해상도 읽기 (width, height)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            widths.append(w)
            heights.append(h)
            resolutions.append(f"{w}x{h}")

        cap.release()

    # -----------------------------
    # 2.3. FPS 분포 및 base_fps 계산
    # -----------------------------
    if fps_list:
        # 소수 FPS도 있기 때문에, 분석 및 카운팅 편의를 위해 반올림 정수 FPS로 변환
        fps_rounded = [int(round(f)) for f in fps_list if f > 0]
        fps_counter = Counter(fps_rounded)

        # 가장 많이 등장한 FPS를 기준(base) FPS로 사용
        base_fps = fps_counter.most_common(1)[0][0]
        print(f"[ANALYSIS] 기준(base) FPS = {base_fps}")
    else:
        fps_counter = Counter()
        base_fps = None

    # -----------------------------
    # 2.4. FPS 히스토그램 시각화
    # -----------------------------
    if fps_list:
        plt.figure()

        # counts: 각 bin에 들어간 개수
        # bins: bin 경계값 배열
        counts, bins, patches = plt.hist(fps_list, bins=20)

        plt.xlabel("FPS")
        plt.ylabel("Count")
        plt.title("FPS Distribution (FFPP + KoDF)")
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)

        # 각 bar 위에 해당 bin count를 숫자로 표기
        for i, count in enumerate(counts):
            if count > 0:
                plt.text(
                    (bins[i] + bins[i + 1]) / 2.0,  # bar 중앙 위치
                    count,
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()
        plt.show()

    # -----------------------------
    # 2.5. FPS Boxplot 시각화
    # -----------------------------
    if len(fps_rounded) > 0:
        plt.figure()
        # patch_artist=True: box 내부 색 채우기
        plt.boxplot(
            fps_rounded,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="lightgreen"),
        )
        plt.ylabel("FPS")
        plt.title("FPS Boxplot Distribution (FFPP + KoDF)")
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # 2.6. 2차 패스: 프레임 샘플링 + 픽셀 밝기 히스토그램 누적
    # -----------------------------
    for vp in video_paths:
        vp_str = str(vp)
        cap = cv2.VideoCapture(vp_str)
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_vid = fps_map.get(vp_str)

        if (
            freeze_to_max_fps
            and base_fps is not None
            and fps_vid
            and fps_vid > 0
        ):
            # --- Case 1: 기준 FPS(base_fps)를 사용하여 "시간 간격" 정규화 ---
            # i번째 샘플링 프레임 인덱스 ≈ i * (fps_vid / base_fps)
            # → 실제 FPS가 달라도 같은 시간 구간에서 비슷한 상대 위치 프레임을 가져오도록 함.
            for i in range(max_frames_per_video):
                frame_idx = int(round(i * fps_vid / base_fps))
                if frame_idx >= total_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                # BGR → Grayscale 변환 후 픽셀 히스토그램 누적
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist_pixels += np.bincount(gray.flatten(), minlength=256)

                # 프레임 평균 밝기 저장 (추후 boxplot 등에서 활용 가능)
                frame_means.append(gray.mean())

        else:
            # --- Case 2: FPS 고려 없이 단순 균등 간격 샘플링 ---
            step = max(1, total_frames // max_frames_per_video)
            sampled = 0
            frame_idx = 0

            while sampled < max_frames_per_video and frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist_pixels += np.bincount(gray.flatten(), minlength=256)

                frame_means.append(gray.mean())

                sampled += 1
                frame_idx += step

        cap.release()

    # -----------------------------
    # 2.7. 해상도 분포 bar plot (Top 20)
    # -----------------------------
    if resolutions:
        res_count = Counter(resolutions)
        mc = res_count.most_common(20)  # 상위 20개 해상도만 표시
        labels = [r for (r, _) in mc]
        counts = [c for (_, c) in mc]

        plt.figure()
        bars = plt.bar(range(len(labels)), counts)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.xlabel("Resolution")
        plt.ylabel("Count")
        plt.title("Resolution Distribution (Top 20)")

        # 각 bar 위에 빈도수를 숫자로 표기
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                str(counts[idx]),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.show()


# =========================================================
# 3. 스크립트 진입점 (standalone 실행 시)
# =========================================================
if __name__ == "__main__":
    # collect.py 의 설정(CONFIG_FFPP, CONFIG_KODF)을 그대로 재사용
    ffpp_out = CONFIG_FFPP["output_root"]
    kodf_out = CONFIG_KODF["output_root"]

    # 1) FFPP + KoDF split 결과에서 모든 영상 경로 수집
    all_videos = collect_all_videos_for_analysis([ffpp_out, kodf_out])

    # 2) FPS / 픽셀 밝기 / 해상도 분포 분석 실행
    analyze_fps_and_pixel_distribution(
        all_videos,
        max_frames_per_video=10,
        freeze_to_max_fps=True,
    )