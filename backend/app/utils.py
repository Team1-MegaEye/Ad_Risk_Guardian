import cv2
import re
from pathlib import Path
from typing import List
from openai import OpenAI

import yt_dlp
import uuid

from moviepy import VideoFileClip

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# ============================================
# 유튜브 영상 다운로드
# ============================================
def download_youtube_video(url: str) -> Path:
    """
    유튜브 URL에서 영상을 다운로드하여 uploads 폴더에 저장하고,
    저장된 파일 경로(Path)를 반환한다.
    """
    video_id = uuid.uuid4().hex  # 파일명 충돌 방지용 임의 ID 생성
    output_path = UPLOAD_DIR / f"{video_id}.mp4"

    # yt-dlp 다운로드 옵션
    ydl_opts = {
        "format": "mp4",              # mp4 포맷 강제
        "outtmpl": str(output_path),  # 저장 경로 템플릿 지정
    }

    # yt-dlp 실행
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path


# ============================================
# 영상 → 프레임 추출 함수
# ============================================
def extract_frames(video_path: str, output_dir: str, num_frames: int = 10) -> List[str]:
    """
    영상의 앞에서 num_frames만큼 프레임을 추출하는 함수 (OpenCV 사용)

    Args:
        video_path (str): 입력 영상 경로
        output_dir (str): 추출한 프레임 저장 경로
        num_frames (int): 뽑을 프레임 수 (기본 10)

    Returns:
        List[str]: 저장된 프레임 이미지 경로 리스트
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    extracted_paths = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # total_frames보다 num_frames이 큰 경우 조절
    actual_frames = min(num_frames, total_frames)

    # 프레임 순차 추출
    for i in range(actual_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = output_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        extracted_paths.append(str(frame_path))

    cap.release()
    return extracted_paths


# ============================================
# 오디오 추출 + Whisper STT
# ============================================

# OpenAI Whisper API 클라이언트 생성
client = OpenAI(api_key="YOUR_OPEN_API_KEY")

def extract_audio(video_path, output_audio_path=None, audio_format="mp3", whisper_model="whisper-1"):
    """
    1) 비디오에서 FFmpeg로 오디오 추출
    2) Whisper STT 적용
    3) 텍스트 반환

    :param video_path: 입력 영상 경로
    :param output_audio_path: 출력 오디오 파일명 (없으면 자동 생성)
    :param audio_format: mp3 또는 wav
    :param whisper_model: Whisper 모델 이름
    """
    video_path = Path(video_path)

    # 출력 오디오 파일명 지정
    if output_audio_path is None:
        output_audio_path = video_path.with_suffix(f".{audio_format}")
    output_audio_path = Path(output_audio_path)

    # ----------------------------
    # 1. moviepy 오디오 추출 (imageio-ffmpeg 백엔드 사용)
    # ----------------------------
    try:
        # 1. VideoFileClip 객체 생성
        video_clip = VideoFileClip(str(video_path))

        # 2. 오디오 클립 추출
        audio_clip = video_clip.audio
        
        # 3. 오디오 파일로 저장
        # audio_clip.write_audiofile()이 내부적으로 imageio-ffmpeg을 통해 FFmpeg 명령을 실행합니다.
        audio_clip.write_audiofile(
            str(output_audio_path), 
            codec=audio_format, # moviepy는 파일 확장자에 따라 코덱을 자동 선택
            logger=None           # 터미널에 FFmpeg 로그 출력을 숨겨서 깔끔하게 만듭니다.
        )

        # 4. 클립 자원 해제
        audio_clip.close()
        video_clip.close()
        
    except Exception as e:
        print(f"❌ 오디오 추출 실패 (moviepy): {e}")
        return # 추출 실패 시 함수 종료
    # ----------------------------
    # 2. Whisper API로 STT 수행
    # ----------------------------
    with open(output_audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=whisper_model,  # Whisper 모델 예: "whisper-1"
            file=audio_file,
            language="ko",
        )

    text = result.text
    return text


# ============================================
# 텍스트 → 문장 단위 분리
# ============================================
def split_sentences(paragraph: str):
    """
    문단을 '.!? ' 기준으로 문장 단위로 분리하여 리스트로 반환
    """
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    return [s.strip() for s in sentences if s.strip()]