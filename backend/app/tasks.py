# tasks.py
import os
import uuid
from celery_app import celery
from .models import models
from .utils import extract_frames, extract_audio
from celery import group, chord

# ============================================
# 영상 처리 태스크
# ============================================
@celery.task
def task_video(video_path, unique_id, num_frames=10):
    """
    영상 처리:
    - temp/{unique_id}/frames 에 프레임 저장
    - 저장된 프레임들을 기반으로 딥페이크 모델 평가
    """
    base_dir = f"./temp/{unique_id}"
    frame_dir = f"{base_dir}/frames"

    os.makedirs(frame_dir, exist_ok=True)

    # 프레임 추출
    extract_frames(video_path, frame_dir, num_frames=num_frames)

    # 딥페이크 확률 계산
    score = models.predict_video(frame_dir)

    return score


# ============================================
# 오디오 처리 태스크
# ============================================
@celery.task
def task_audio(video_path, unique_id):
    """
    오디오 처리:
    - temp/{unique_id}/audio.wav 추출
    - STT 수행 → 텍스트 모델 predict_text 호출
    - 텍스트 기반 과장 확률(exaggeration_prob) 반환
    """
    base_dir = f"./temp/{unique_id}"
    os.makedirs(base_dir, exist_ok=True)

    audio_path = f"{base_dir}/audio.wav"

    # STT 수행 → transcript(문자열) 반환
    transcript = extract_audio(video_path, audio_path, audio_format="mp3")

    # 텍스트 모델 결과
    result = models.predict_text(transcript)

    # 가장 높은 과장 문장의 확률 사용
    score_value = result.get("exaggeration_prob")
    score = float(score_value)

    return score


# ============================================
# 메인 인퍼런스 태스크
# (영상 태스크 + 오디오 태스크 병렬 실행)
# ============================================
@celery.task
def run_inference(video_path):
    """
    - 하나의 inference 수행을 위한 고유 폴더 생성
    - task_video + task_audio 병렬 실행
    - 완료 후 combine_results 에서 최종 결과 집계
    - combine_results 작업의 task_id 반환
    """
    # 개별 요청마다 uuid 폴더 생성
    unique_id = str(uuid.uuid4())

    # 두 태스크를 병렬로 실행하도록 chord 구성
    header = [
        task_video.s(video_path, unique_id),
        task_audio.s(video_path, unique_id)
    ]

    # header 완료 후 combine_results 실행
    result = chord(header)(combine_results.s(unique_id))

    # combine_results 태스크의 ID 반환
    return result.id


# ============================================
# 영상/오디오 결과 결합 태스크
# ============================================
@celery.task
def combine_results(results, unique_id):
    """
    - 병렬 태스크 결과(video_score, text_score)를 받아서 late-fusion 방식으로 최종 점수 산출
    - 0.5 기준으로 신뢰여부(label) 결정
    """
    video_score, text_score = results

    # 위험단계 분류
    if text_score >= 0.5 and video_score >= 0.5:
        danger_class = "매우위험"     # 둘 다 해당
    elif text_score >= 0.5:
        danger_class = "위험"        # 과장만 해당
    elif video_score >= 0.5:
        danger_class = "주의"        # 딥페이크만 해당
    else:
        danger_class = "안전"        # 둘 다 아님

    # 단순 late-fusion (가중치 0.5)
    joint_score = 0.7 * video_score + 0.3 * text_score
    final_score = 1 - joint_score

    return {
        "video_score": video_score,
        "text_score": text_score,
        "label": danger_class,
        "final_score": final_score
    }
