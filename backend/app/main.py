# main.py
from fastapi import FastAPI, UploadFile, File 
from pathlib import Path
from pydantic import BaseModel

from celery_app import celery
from app.utils import download_youtube_video

# ---------------------------------------------------------
# FastAPI 초기화
# ---------------------------------------------------------
app = FastAPI()

# ---------------------------------------------------------
# 업로드된 파일 저장 경로 생성
# ---------------------------------------------------------
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------
# 유튜브 URL 요청 바디 모델
# ---------------------------------------------------------
class YoutubeRequest(BaseModel):
    url: str


# ---------------------------------------------------------
# 1) /predict : 유튜브 영상 URL을 받고 분석을 시작하는 엔드포인트
# ---------------------------------------------------------
@app.post("/predict")
async def submit_predict(req: YoutubeRequest):
    youtube_url = req.url

    # Step 1: 유튜브 영상 다운로드 → 로컬 파일 경로 반환
    file_path = download_youtube_video(youtube_url)

    # Step 2: Celery 비동기 태스크 실행
    # run_inference 는 내부적으로
    #   - task_video
    #   - task_audio
    # 를 병렬 실행하고 combine_results에서 최종 결과를 합침
    task = celery.send_task("app.tasks.run_inference", args=[str(file_path)])

    # 프론트엔드는 task_id 를 사용해 /result/{task_id} 로 상태를 조회함
    return {"task_id": task.id}


# ---------------------------------------------------------
# 2) /result : 분석 결과를 조회하는 엔드포인트
# ---------------------------------------------------------
@app.get("/result/{task_id}")
def get_result(task_id: str):
    # run_inference 태스크 상태 확인
    run_inference_res = celery.AsyncResult(task_id)

    # 아직 처리 중이라면 프론트엔드는 polling 을 계속하면 됨
    if not run_inference_res.ready():
        return {"status": "processing"}

    # run_inference가 완료되면 그 결과 값은 combine_results 의 task_id 임
    combine_results_id = run_inference_res.result
    
    # combine_results 의 상태 확인
    combine_res = celery.AsyncResult(combine_results_id)

    if not combine_res.ready():
        return {"status": "processing"}

    # 모든 태스크 완료 → 최종 예측 결과 반환
    return {
        "status": "completed",
        "result": combine_res.result
    }
