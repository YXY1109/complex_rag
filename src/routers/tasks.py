from fastapi import APIRouter

from celery.result import AsyncResult

router = APIRouter(
    prefix="/tasks",
    tags=["后台任务"],
)


# 启动任务
@router.post("/start-task/")
async def start_task(message: str):
    # delay简单 apply_async复杂
    # task = long_task.delay(message)  # 使用Celery异步执行任务
    # task = long_task.apply_async(args=[message], queue="yxy")
    # return {"task_id": task.id, "message": "任务已提交"}
    return {"task_id": "task.id", "message": "任务已提交"}


# 查询任务状态
@router.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id)
    if task_result.state == "PENDING":
        return {"task_id": task_id, "status": "任务处理中"}
    elif task_result.state == "SUCCESS":
        return {"task_id": task_id, "status": "任务完成", "result": task_result.result}
    else:
        return {"task_id": task_id, "status": task_result.state}
