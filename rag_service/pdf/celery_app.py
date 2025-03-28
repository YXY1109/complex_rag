import os

from pydantic import BaseModel

from celery import Celery

# https://github.com/celery/celery/issues/8921
# ValueError: not enough values to unpack (expected 3, got 0)
os.environ["FORKED_BY_MULTIPROCESSING"] = "1"


class CeleryConfig(BaseModel):
    timezone: str = "Asia/Shanghai"
    enable_utc: bool = False
    worker_concurrency: int = 2
    worker_max_tasks_per_child: int = 2
    worker_max_memory_per_child: int = (1 << 10) * 500  # 50
    task_default_queue: str = " default "
    task_default_exchange: str = " default "
    task_default_routing_key: str = " default "
    broker_connection_retry_on_startup: bool = True


celery_url = ""
# 创建 Celery 应用实例
celery_app = Celery("tasks", broker=celery_url, backend=celery_url)
celery_app.config_from_object(CeleryConfig())


# 自动发现任务
# celery_app.autodiscover_tasks(['app.celery.tasks'])


@celery_app.task
def long_task(name: str, age: int):
    print(f"开始执行任务1: {name}")
    print(f"开始执行任务2: {age}")
    import time

    time.sleep(10)  # 模拟耗时任务
    print("任务完成")
    return f"任务完成，信息: {name}"
