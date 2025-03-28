from src.celery.celery_app import celery_app

# 命令行启动：celery -A app.celery.celery_app worker --loglevel=info

# 启动 Celery worker 和 beat
if __name__ == "__main__":
    # celery_app.worker_main(["worker", "-B", "-l", "info"])
    celery_app.worker_main(["worker", "-B", "--loglevel=info", "--concurrency=4", "--queues=yxy"])
