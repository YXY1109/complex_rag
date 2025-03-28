from pydantic import BaseModel

from celery import Celery
from src.config.config import settings
from src.utils.logger import logger


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


redis_host = settings.REDIS_HOST
redis_port = settings.REDIS_PORT
redis_password = settings.REDIS_PASSWORD
redis_db_broker = settings.REDIS_DB_BROKER
redis_db_backend = settings.REDIS_DB_BACKEND

celery_broker_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db_broker}"
celery_backend_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db_backend}"
logger.info(f"celery_broker_url:{celery_broker_url}")
logger.info(f"celery_backend_url:{celery_backend_url}")

# 创建 Celery 应用实例
celery_app = Celery("tasks", broker=celery_broker_url, backend=celery_backend_url)
celery_app.config_from_object(CeleryConfig())

# 自动发现任务
celery_app.autodiscover_tasks(["app.celery.tasks"])
