from src.celery.celery_app import celery_app


@celery_app.task
def long_task(message: str):
    import time

    time.sleep(20)  # 模拟耗时任务
    return f"任务完成，信息: {message}"
