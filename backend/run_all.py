# run_all.py
import subprocess
import time
import os

def run_in_background(command):
    return subprocess.Popen(command, shell=True)

if __name__ == "__main__":
    print("[1/3] Redis 서버 실행...")
    run_in_background("docker run --name some-redis -d -p 6379:6379 redis")
    time.sleep(2)   # Redis가 켜질 시간을 잠깐 줌

    print("[2/3] Celery 실행...")
    os.environ["PYTHONPATH"] = "."
    run_in_background("celery -A celery_app worker -l info -Q inference --concurrency=4 -P threads")
    time.sleep(2)

    print("[3/3] FastAPI 실행...")
    run_in_background("uvicorn app.main:app --reload")

    print("\n✨ 모든 서비스가 실행되었습니다!")
