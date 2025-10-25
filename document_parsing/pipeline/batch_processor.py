"""
异步批处理器

支持大规模文档的批量异步处理，包括任务调度、负载均衡、进度监控等功能。
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, AsyncIterator
from enum import Enum
from dataclasses import dataclass, field
import uuid
import logging
from collections import defaultdict, deque

from .pipeline_config import PipelineConfig, PipelineContext
from .pipeline_orchestrator import PipelineOrchestrator, OrchestratorState
from ..interfaces.parser_interface import ProcessingStrategy
from ..strategy_config import ProcessingStrategyConfig


class BatchStatus(Enum):
    """批处理状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskStatus(Enum):
    """任务状态。"""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class BatchTask:
    """批处理任务。"""

    task_id: str
    file_path: str
    tenant_id: str
    user_id: str
    source_type: Optional[str] = None
    processing_strategy: ProcessingStrategy = ProcessingStrategy.AUTO
    strategy_config: Optional[ProcessingStrategyConfig] = None
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Any] = None

    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())

    @property
    def status(self) -> TaskStatus:
        """获取任务状态。"""
        if self.error:
            return TaskStatus.FAILED
        elif self.result is not None:
            return TaskStatus.COMPLETED
        elif self.started_at is not None:
            return TaskStatus.RUNNING
        else:
            return TaskStatus.QUEUED

    @property
    def execution_time(self) -> float:
        """获取执行时间。"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return 0.0


@dataclass
class BatchJob:
    """批处理作业。"""

    job_id: str
    name: str
    description: str = ""
    tasks: List[BatchTask] = field(default_factory=list)
    pipeline_config: Optional[PipelineConfig] = None
    max_concurrent_tasks: int = 5
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.job_id:
            self.job_id = str(uuid.uuid4())

    @property
    def status(self) -> BatchStatus:
        """获取作业状态。"""
        if not self.tasks:
            return BatchStatus.PENDING

        task_statuses = [task.status for task in self.tasks]

        if any(status == TaskStatus.FAILED for status in task_statuses):
            return BatchStatus.FAILED
        elif all(status == TaskStatus.COMPLETED for status in task_statuses):
            return BatchStatus.COMPLETED
        elif any(status == TaskStatus.RUNNING for status in task_statuses):
            return BatchStatus.RUNNING
        else:
            return BatchStatus.PENDING

    @property
    def completed_tasks(self) -> int:
        """已完成任务数。"""
        return len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])

    @property
    def failed_tasks(self) -> int:
        """失败任务数。"""
        return len([t for t in self.tasks if t.status == TaskStatus.FAILED])

    @property
    def running_tasks(self) -> int:
        """运行中任务数。"""
        return len([t for t in self.tasks if t.status == TaskStatus.RUNNING])

    @property
    def total_tasks(self) -> int:
        """总任务数。"""
        return len(self.tasks)

    @property
    def progress(self) -> float:
        """进度百分比。"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks + self.failed_tasks) / self.total_tasks

    @property
    def execution_time(self) -> float:
        """总执行时间。"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return 0.0

    def add_task(self, task: BatchTask) -> None:
        """添加任务。"""
        self.tasks.append(task)

    def get_next_tasks(self, limit: int) -> List[BatchTask]:
        """获取下一批待执行任务。"""
        queued_tasks = [t for t in self.tasks if t.status == TaskStatus.QUEUED]
        # 按优先级排序
        queued_tasks.sort(key=lambda t: t.priority, reverse=True)
        return queued_tasks[:limit]

    def get_task(self, task_id: str) -> Optional[BatchTask]:
        """获取任务。"""
        return next((t for t in self.tasks if t.task_id == task_id), None)


@dataclass
class BatchStatistics:
    """批处理统计信息。"""

    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    running_jobs: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    running_tasks: int = 0
    average_task_time: float = 0.0
    average_job_time: float = 0.0
    throughput: float = 0.0  # 任务/秒
    last_updated: datetime = field(default_factory=datetime.now)


class BatchProcessor:
    """异步批处理器。"""

    def __init__(self, orchestrator: PipelineOrchestrator):
        """
        初始化批处理器。

        Args:
            orchestrator: 流水线编排器
        """
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)

        # 作业和任务管理
        self.active_jobs: Dict[str, BatchJob] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.completed_tasks: deque = deque(maxlen=10000)  # 保留最近完成的任务
        self.failed_tasks: deque = deque(maxlen=10000)     # 保留最近失败的任务

        # 执行控制
        self.max_concurrent_jobs = 10
        self.max_concurrent_tasks_per_job = 5
        self.global_max_concurrent_tasks = 20
        self.current_running_tasks = 0
        self.current_running_jobs = 0

        # 调度和监控
        self.scheduler_task: Optional[asyncio.Task] = None
        self.monitor_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.statistics = BatchStatistics()

        # 回调函数
        self.job_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.task_callbacks: Dict[str, List[Callable]] = defaultdict(list)

    async def start(self) -> None:
        """启动批处理器。"""
        if self.is_running:
            return

        self.is_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("批处理器已启动")

    async def stop(self) -> None:
        """停止批处理器。"""
        if not self.is_running:
            return

        self.is_running = False

        # 取消调度器任务
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        # 取消监控任务
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        # 等待所有运行中的任务完成（设置超时）
        timeout = 30  # 30秒超时
        start_time = time.time()

        while self.current_running_tasks > 0 and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)

        if self.current_running_tasks > 0:
            self.logger.warning(f"停止时仍有 {self.current_running_tasks} 个任务在运行")

        self.logger.info("批处理器已停止")

    async def submit_batch_job(
        self,
        job_name: str,
        file_paths: List[str],
        tenant_id: str,
        user_id: str,
        processing_strategy: ProcessingStrategy = ProcessingStrategy.AUTO,
        strategy_config: Optional[ProcessingStrategyConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        max_concurrent_tasks: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        提交批处理作业。

        Args:
            job_name: 作业名称
            file_paths: 文件路径列表
            tenant_id: 租户ID
            user_id: 用户ID
            processing_strategy: 处理策略
            strategy_config: 策略配置
            pipeline_config: 流水线配置
            max_concurrent_tasks: 最大并发任务数
            metadata: 元数据

        Returns:
            str: 作业ID
        """
        job_id = str(uuid.uuid4())

        # 创建批处理作业
        job = BatchJob(
            job_id=job_id,
            name=job_name,
            description=f"批处理作业，包含 {len(file_paths)} 个文件",
            pipeline_config=pipeline_config,
            max_concurrent_tasks=max_concurrent_tasks,
            metadata=metadata or {}
        )

        # 创建任务
        for i, file_path in enumerate(file_paths):
            task = BatchTask(
                task_id=str(uuid.uuid4()),
                file_path=file_path,
                tenant_id=tenant_id,
                user_id=user_id,
                processing_strategy=processing_strategy,
                strategy_config=strategy_config,
                priority=len(file_paths) - i  # 先处理的文件优先级更高
            )
            job.add_task(task)

        # 添加到活跃作业
        self.active_jobs[job_id] = job

        # 添加任务到队列
        for task in job.tasks:
            await self.task_queue.put((job_id, task.task_id))

        self.logger.info(f"提交批处理作业 {job_id}，包含 {len(file_paths)} 个任务")
        return job_id

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取作业状态。"""
        job = self.active_jobs.get(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "name": job.name,
            "status": job.status.value,
            "progress": job.progress,
            "total_tasks": job.total_tasks,
            "completed_tasks": job.completed_tasks,
            "failed_tasks": job.failed_tasks,
            "running_tasks": job.running_tasks,
            "execution_time": job.execution_time,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        }

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态。"""
        # 在活跃作业中查找
        for job in self.active_jobs.values():
            task = job.get_task(task_id)
            if task:
                return {
                    "task_id": task.task_id,
                    "file_path": task.file_path,
                    "status": task.status.value,
                    "execution_time": task.execution_time,
                    "retry_count": task.retry_count,
                    "error": task.error,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None
                }

        # 在已完成任务中查找
        for completed_task in self.completed_tasks:
            if completed_task.task_id == task_id:
                return {
                    "task_id": completed_task.task_id,
                    "file_path": completed_task.file_path,
                    "status": completed_task.status.value,
                    "execution_time": completed_task.execution_time,
                    "retry_count": completed_task.retry_count,
                    "error": completed_task.error,
                    "created_at": completed_task.created_at.isoformat(),
                    "started_at": completed_task.started_at.isoformat() if completed_task.started_at else None,
                    "completed_at": completed_task.completed_at.isoformat() if completed_task.completed_at else None
                }

        return None

    async def cancel_job(self, job_id: str) -> bool:
        """取消作业。"""
        job = self.active_jobs.get(job_id)
        if not job or job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
            return False

        # 取消所有待执行的任务
        for task in job.tasks:
            if task.status == TaskStatus.QUEUED:
                task.error = "任务被取消"

        self.logger.info(f"取消批处理作业 {job_id}")
        return True

    async def pause_job(self, job_id: str) -> bool:
        """暂停作业。"""
        job = self.active_jobs.get(job_id)
        if not job or job.status != BatchStatus.RUNNING:
            return False

        # 这里可以实现暂停逻辑
        self.logger.info(f"暂停批处理作业 {job_id}")
        return True

    async def resume_job(self, job_id: str) -> bool:
        """恢复作业。"""
        job = self.active_jobs.get(job_id)
        if not job or job.status != BatchStatus.PAUSED:
            return False

        # 这里可以实现恢复逻辑
        self.logger.info(f"恢复批处理作业 {job_id}")
        return True

    async def get_statistics(self) -> BatchStatistics:
        """获取统计信息。"""
        return self.statistics

    async def get_jobs_list(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """获取作业列表。"""
        jobs = list(self.active_jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        result = []
        for job in jobs[offset:offset + limit]:
            result.append({
                "job_id": job.job_id,
                "name": job.name,
                "status": job.status.value,
                "progress": job.progress,
                "total_tasks": job.total_tasks,
                "completed_tasks": job.completed_tasks,
                "failed_tasks": job.failed_tasks,
                "created_at": job.created_at.isoformat()
            })

        return result

    async def _scheduler_loop(self) -> None:
        """调度器循环。"""
        self.logger.info("批处理器调度器已启动")

        while self.is_running:
            try:
                # 检查是否可以接受新任务
                if self.current_running_tasks < self.global_max_concurrent_tasks:
                    # 获取下一个任务
                    try:
                        job_id, task_id = await asyncio.wait_for(
                            self.task_queue.get(),
                            timeout=1.0
                        )

                        # 执行任务
                        asyncio.create_task(self._execute_task(job_id, task_id))

                    except asyncio.TimeoutError:
                        continue

                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"调度器循环异常: {e}")
                await asyncio.sleep(1.0)

        self.logger.info("批处理器调度器已停止")

    async def _execute_task(self, job_id: str, task_id: str) -> None:
        """执行任务。"""
        job = self.active_jobs.get(job_id)
        if not job:
            self.logger.warning(f"作业 {job_id} 不存在")
            return

        task = job.get_task(task_id)
        if not task:
            self.logger.warning(f"任务 {task_id} 不存在")
            return

        # 检查作业并发限制
        if job.running_tasks >= job.max_concurrent_tasks:
            # 重新排队
            await self.task_queue.put((job_id, task_id))
            await asyncio.sleep(0.1)
            return

        # 开始执行任务
        self.current_running_tasks += 1
        job.running_tasks += 1
        task.started_at = datetime.now()

        # 启动作业（如果是第一个任务）
        if job.started_at is None:
            job.started_at = datetime.now()

        self.logger.info(f"开始执行任务 {task_id} (作业 {job_id})")

        try:
            # 创建流水线上下文
            context = PipelineContext(
                pipeline_id=job.pipeline_config.pipeline_id if job.pipeline_config else "default",
                execution_id=str(uuid.uuid4()),
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                source_type=task.source_type or "unknown",
                processing_strategy=task.processing_strategy,
                strategy_config=task.strategy_config or ProcessingStrategyConfig(),
                input_data={"file_path": task.file_path}
            )

            # 使用作业的流水线配置或默认配置
            pipeline_config = job.pipeline_config or self.orchestrator.config

            # 临时替换编排器配置
            original_config = self.orchestrator.config
            self.orchestrator.config = pipeline_config

            try:
                # 执行流水线
                results = await self.orchestrator.execute_pipeline(context)
                task.result = results
                task.completed_at = datetime.now()

                # 添加到已完成任务
                self.completed_tasks.append(task)

                self.logger.info(f"任务 {task_id} 执行完成")

            finally:
                # 恢复原始配置
                self.orchestrator.config = original_config

        except Exception as e:
            task.error = str(e)
            task.completed_at = datetime.now()
            self.failed_tasks.append(task)
            self.logger.error(f"任务 {task_id} 执行失败: {e}")

            # 重试逻辑
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.error = None
                task.started_at = None
                task.completed_at = None

                # 重新排队
                await self.task_queue.put((job_id, task_id))
                self.logger.info(f"任务 {task_id} 将进行第 {task.retry_count} 次重试")

        finally:
            # 更新计数器
            self.current_running_tasks -= 1
            job.running_tasks -= 1

            # 检查作业是否完成
            if job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
                job.completed_at = datetime.now()
                self.current_running_jobs -= 1

            # 调用回调
            await self._call_task_callbacks(task_id, task)
            await self._call_job_callbacks(job_id, job)

    async def _monitor_loop(self) -> None:
        """监控循环。"""
        self.logger.info("批处理器监控器已启动")

        while self.is_running:
            try:
                # 更新统计信息
                await self._update_statistics()

                # 清理过期的作业
                await self._cleanup_expired_jobs()

                await asyncio.sleep(10.0)  # 每10秒更新一次

            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(10.0)

        self.logger.info("批处理器监控器已停止")

    async def _update_statistics(self) -> None:
        """更新统计信息。"""
        self.statistics.total_jobs = len(self.active_jobs)
        self.statistics.completed_jobs = len([j for j in self.active_jobs.values() if j.status == BatchStatus.COMPLETED])
        self.statistics.failed_jobs = len([j for j in self.active_jobs.values() if j.status == BatchStatus.FAILED])
        self.statistics.running_jobs = len([j for j in self.active_jobs.values() if j.status == BatchStatus.RUNNING])

        total_tasks = sum(len(job.tasks) for job in self.active_jobs.values())
        completed_tasks = sum(job.completed_tasks for job in self.active_jobs.values())
        failed_tasks = sum(job.failed_tasks for job in self.active_jobs.values())
        running_tasks = sum(job.running_tasks for job in self.active_jobs.values())

        self.statistics.total_tasks = total_tasks
        self.statistics.completed_tasks = completed_tasks
        self.statistics.failed_tasks = failed_tasks
        self.statistics.running_tasks = running_tasks

        # 计算平均时间
        if self.completed_tasks:
            self.statistics.average_task_time = sum(t.execution_time for t in self.completed_tasks) / len(self.completed_tasks)

        completed_jobs = [j for j in self.active_jobs.values() if j.status == BatchStatus.COMPLETED]
        if completed_jobs:
            self.statistics.average_job_time = sum(j.execution_time for j in completed_jobs) / len(completed_jobs)

        # 计算吞吐量（最近1分钟）
        recent_time = datetime.now() - timedelta(minutes=1)
        recent_tasks = [t for t in self.completed_tasks if t.completed_at and t.completed_at > recent_time]
        self.statistics.throughput = len(recent_tasks) / 60.0  # 任务/秒

        self.statistics.last_updated = datetime.now()

    async def _cleanup_expired_jobs(self) -> None:
        """清理过期的作业。"""
        expiry_time = datetime.now() - timedelta(hours=24)  # 24小时过期

        expired_jobs = [
            job_id for job_id, job in self.active_jobs.items()
            if job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED] and
               job.completed_at and job.completed_at < expiry_time
        ]

        for job_id in expired_jobs:
            self.active_jobs.pop(job_id, None)
            self.logger.info(f"清理过期作业 {job_id}")

    async def _call_task_callbacks(self, task_id: str, task: BatchTask) -> None:
        """调用任务回调。"""
        callbacks = self.task_callbacks.get(task_id, [])
        for callback in callbacks:
            try:
                await callback(task)
            except Exception as e:
                self.logger.error(f"任务回调执行失败: {e}")

    async def _call_job_callbacks(self, job_id: str, job: BatchJob) -> None:
        """调用作业回调。"""
        callbacks = self.job_callbacks.get(job_id, [])
        for callback in callbacks:
            try:
                await callback(job)
            except Exception as e:
                self.logger.error(f"作业回调执行失败: {e}")

    def register_job_callback(self, job_id: str, callback: Callable) -> None:
        """注册作业回调。"""
        self.job_callbacks[job_id].append(callback)

    def register_task_callback(self, task_id: str, callback: Callable) -> None:
        """注册任务回调。"""
        self.task_callbacks[task_id].append(callback)

    def remove_job_callback(self, job_id: str, callback: Callable) -> None:
        """移除作业回调。"""
        if callback in self.job_callbacks[job_id]:
            self.job_callbacks[job_id].remove(callback)

    def remove_task_callback(self, task_id: str, callback: Callable) -> None:
        """移除任务回调。"""
        if callback in self.task_callbacks[task_id]:
            self.task_callbacks[task_id].remove(callback)