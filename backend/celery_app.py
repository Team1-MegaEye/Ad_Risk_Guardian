from celery import Celery

celery = Celery(
    "app",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)


celery.conf.task_routes = {"app.tasks.*": {"queue": "inference"}}

# app.tasks 모듈을 명시적으로 임포트하도록 지정
celery.conf.imports = ('app.tasks',)