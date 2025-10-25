"""
流水线调度器

负责流水线任务的调度、优先级管理、资源分配等功能，
支持定时任务、依赖管理、负载均衡等高级调度特性。
"""

import asyncio
import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
import json
from collections import defaultdict

from .pipeline_config import PipelineConfig, PipelineContext
from .pipeline_orchestrator import PipelineOrchestrator


class TaskPriority(Enum):
    """任务优先级。"""

    CRITICAL = 0    # 关键任务
    HIGH = 1        # 高优先级
    NORMAL = 2      # 普通优先级
    LOW = 3         # 低优先级
    BACKGROUND = 4  # 后台任务


class TaskStatus(Enum):
    """任务状态。"""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ScheduleType(Enum):
    """调度类型。"""

    IMMEDIATE = "immediate"    # 立即执行
    DELAYED = "delayed"        # 延迟执行
    PERIODIC = "periodic"      # 周期执行
    CRON = "cron"             # Cron表达式
    DEPENDENCY = "dependency"  # 依赖执行


@dataclass
class ScheduledTask:
    """调度任务。"""

    task_id: str
    pipeline_config: PipelineConfig
    context: PipelineContext
    schedule_type: ScheduleType
    priority: TaskPriority = TaskPriority.NORMAL
    scheduled_time: Optional[datetime] = None
    cron_expression: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Any] = None

    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())

    @property
    def execution_time(self) -> float:
        """执行时间。"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return 0.0

    @property
    def is_ready(self) -> bool:
        """是否准备就绪。"""
        if self.status != TaskStatus.PENDING:
            return False

        now = datetime.now()

        # 检查调度时间
        if self.schedule_type == ScheduleType.IMMEDIATE:
            return True
        elif self.schedule_type == ScheduleType.DELAYED:
            return self.scheduled_time and now >= self.scheduled_time
        elif self.schedule_type == ScheduleType.PERIODIC:
            return self.scheduled_time and now >= self.scheduled_time
        elif self.schedule_type == ScheduleType.CRON:
            # 简化的Cron检查
            return self._check_cron_time(now)
        elif self.schedule_type == ScheduleType.DEPENDENCY:
            return self._check_dependencies()

        return False

    def _check_cron_time(self, now: datetime) -> bool:
        """检查Cron时间。"""
        if not self.cron_expression:
            return False

        # 简化实现，只支持基本的分钟/小时/天
        # 实际应用中可以使用croniter库
        try:
            parts = self.cron_expression.split()
            if len(parts) >= 2:
                minute = parts[0]
                hour = parts[1]

                if minute == "*" and hour == "*":
                    return True
                elif minute != "*" and now.minute == int(minute):
                    return True
                elif hour != "*" and now.hour == int(hour):
                    return True

        except (ValueError, IndexError):
            pass

        return False

    def _check_dependencies(self) -> bool:
        """检查依赖是否满足。"""
        # 这里需要检查依赖任务的状态
        # 简化实现，假设所有依赖都满足
        return True

    def __lt__(self, other) -> bool:
        """用于优先队列排序。"""
        # 优先级数字越小，优先级越高
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value

        # 相同优先级按调度时间排序
        if self.scheduled_time and other.scheduled_time:
            return self.scheduled_time < other.scheduled_time

        return self.created_at < other.created_at


@dataclass
class ResourcePool:
    """资源池。"""

    pool_id: str
    name: str
    max_concurrent_tasks: int = 10
    current_tasks: int = 0
    cpu_limit: float = 0.8
    memory_limit: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.pool_id:
            self.pool_id = str(uuid.uuid4())

    @property
    def is_available(self) -> bool:
        """是否有可用资源。"""
        return self.current_tasks < self.max_concurrent_tasks

    @property
    def utilization(self) -> float:
        """资源利用率。"""
        return self.current_tasks / self.max_concurrent_tasks

    def acquire_resource(self) -> bool:
        """获取资源。"""
        if self.is_available:
            self.current_tasks += 1
            return True
        return False

    def release_resource(self) -> None:
        """释放资源。"""
        if self.current_tasks > 0:
            self.current_tasks -= 1


class PipelineScheduler:
    """流水线调度器。"""

    def __init__(self, orchestrator: PipelineOrchestrator, config: Optional[Dict[str, Any]] = None):
        """
        初始化调度器。

        Args:
            orchestrator: 流水线编排器
            config: 配置参数
        """
        self.orchestrator = orchestrator
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 任务队列（优先队列）
        self.task_queue: List[ScheduledTask] = []
        self.task_index: Dict[str, ScheduledTask] = {}

        # 资源池
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.default_pool = ResourcePool("default", "默认资源池", max_concurrent_tasks=5)
        self.resource_pools["default"] = self.default_pool

        # 调度状态
        self.is_running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.worker_tasks: Set[asyncio.Task] = set()

        # 统计信息
        self.statistics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "running_tasks": 0,
            "pending_tasks": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0
        }

        # 回调函数
        self.task_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # 任务依赖图
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependency_graph: Dict[str, Set[str]] = defaultdict(set)

    async def start(self) -> None:
        """启动调度器。"""
        if self.is_running:
            return

        self.is_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.logger.info("流水线调度器已启动")

    async def stop(self) -> None:
        """停止调度器。"""
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

        # 等待所有工作线程完成
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            self.worker_tasks.clear()

        self.logger.info("流水线调度器已停止")

    async def submit_task(
        self,
        pipeline_config: PipelineConfig,
        context: PipelineContext,
        schedule_type: ScheduleType = ScheduleType.IMMEDIATE,
        priority: TaskPriority = TaskPriority.NORMAL,
        scheduled_time: Optional[datetime] = None,
        cron_expression: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        resource_pool: str = "default"
    ) -> str:
        """
        提交调度任务。

        Args:
            pipeline_config: 流水线配置
            context: 执行上下文
            schedule_type: 调度类型
            priority: 任务优先级
            scheduled_time: 调度时间
            cron_expression: Cron表达式
            dependencies: 依赖任务ID列表
            timeout: 超时时间（秒）
            metadata: 元数据
            resource_pool: 资源池名称

        Returns:
            str: 任务ID
        """
        task = ScheduledTask(
            task_id=str(uuid.uuid4()),
            pipeline_config=pipeline_config,
            context=context,
            schedule_type=schedule_type,
            priority=priority,
            scheduled_time=scheduled_time,
            cron_expression=cron_expression,
            dependencies=dependencies or [],
            timeout=timeout,
            metadata=metadata or {}
        )

        # 添加到任务队列
        heapq.heappush(self.task_queue, task)
        self.task_index[task.task_id] = task

        # 建立依赖关系
        if dependencies:
            for dep_id in dependencies:
                self.dependency_graph[task.task_id].add(dep_id)
                self.reverse_dependency_graph[dep_id].add(task.task_id)

        self.statistics["total_tasks"] += 1
        self.statistics["pending_tasks"] += 1

        self.logger.info(f"提交调度任务 {task.task_id}，类型: {schedule_type.value}")

        # 触发任务提交回调
        await self._trigger_task_callback("task_submitted", task)

        return task.task_id

    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务。

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否取消成功
        """
        task = self.task_index.get(task_id)
        if not task:
            return False

        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False

        # 更新任务状态
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()

        # 更新统计
        if task.status == TaskStatus.PENDING:
            self.statistics["pending_tasks"] -= 1

        self.logger.info(f"任务 {task_id} 已取消")

        # 触发任务取消回调
        await self._trigger_task_callback("task_cancelled", task)

        return True

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态。"""
        task = self.task_index.get(task_id)
        if not task:
            return None

        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "priority": task.priority.value,
            "schedule_type": task.schedule_type.value,
            "execution_time": task.execution_time,
            "retry_count": task.retry_count,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "error_message": task.error_message,
            "dependencies": task.dependencies
        }

    async def get_task_list(
        self,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取任务列表。"""
        tasks = list(self.task_index.values())

        # 过滤条件
        if status:
            tasks = [t for t in tasks if t.status == status]

        if priority:
            tasks = [t for t in tasks if t.priority == priority]

        # 按创建时间倒序排列
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        # 转换为字典格式
        result = []
        for task in tasks[:limit]:
            result.append({
                "task_id": task.task_id,
                "status": task.status.value,
                "priority": task.priority.value,
                "schedule_type": task.schedule_type.value,
                "execution_time": task.execution_time,
                "created_at": task.created_at.isoformat()
            })

        return result

    async def _scheduler_loop(self) -> None:
        """调度器主循环。"""
        self.logger.info("调度器主循环已启动")

        while self.is_running:
            try:
                # 检查准备就绪的任务
                ready_tasks = []
                temp_queue = []

                while self.task_queue:
                    task = heapq.heappop(self.task_queue)
                    if task.is_ready:
                        ready_tasks.append(task)
                    else:
                        temp_queue.append(task)

                # 将未准备好的任务放回队列
                for task in temp_queue:
                    heapq.heappush(self.task_queue, task)

                # 调度准备就绪的任务
                for task in ready_tasks:
                    await self._schedule_task(task)

                # 清理已完成和失败的任务
                await self._cleanup_completed_tasks()

                # 等待下一次调度
                await asyncio.sleep(self.config.get("schedule_interval", 1.0))

            except Exception as e:
                self.logger.error(f"调度器循环异常: {e}")
                await asyncio.sleep(5.0)

        self.logger.info("调度器主循环已停止")

    async def _schedule_task(self, task: ScheduledTask) -> None:
        """调度任务。"""
        # 检查资源可用性
        pool_name = task.metadata.get("resource_pool", "default")
        resource_pool = self.resource_pools.get(pool_name, self.default_pool)

        if not resource_pool.is_available:
            # 资源不足，重新排队
            heapq.heappush(self.task_queue, task)
            return

        # 获取资源
        resource_pool.acquire_resource()

        # 更新任务状态
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        self.statistics["pending_tasks"] -= 1
        self.statistics["running_tasks"] += 1

        self.logger.info(f"开始执行任务 {task.task_id}")

        # 创建工作线程
        worker_task = asyncio.create_task(self._execute_task(task, resource_pool))
        self.worker_tasks.add(worker_task)
        worker_task.add_done_callback(lambda t: self.worker_tasks.discard(t))

        # 触发任务开始回调
        await self._trigger_task_callback("task_started", task)

    async def _execute_task(self, task: ScheduledTask, resource_pool: ResourcePool) -> None:
        """执行任务。"""
        try:
            # 执行流水线
            results = await asyncio.wait_for(
                self.orchestrator.execute_pipeline(task.context),
                timeout=task.timeout
            )

            task.result = results
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            # 更新统计
            self.statistics["running_tasks"] -= 1
            self.statistics["completed_tasks"] += 1
            execution_time = task.execution_time
            self.statistics["total_execution_time"] += execution_time

            if self.statistics["completed_tasks"] > 0:
                self.statistics["average_execution_time"] = (
                    self.statistics["total_execution_time"] / self.statistics["completed_tasks"]
                )

            self.logger.info(f"任务 {task.task_id} 执行完成，耗时: {execution_time:.2f}秒")

            # 触发任务完成回调
            await self._trigger_task_callback("task_completed", task)

            # 检查并触发依赖任务
            await self._check_dependent_tasks(task.task_id)

            # 处理周期任务
            if task.schedule_type == ScheduleType.PERIODIC and task.cron_expression:
                await self._reschedule_periodic_task(task)

        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error_message = "任务执行超时"
            task.completed_at = datetime.now()
            await self._handle_task_failure(task, resource_pool)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            await self._handle_task_failure(task, resource_pool)

        finally:
            # 释放资源
            resource_pool.release_resource()

    async def _handle_task_failure(self, task: ScheduledTask, resource_pool: ResourcePool) -> None:
        """处理任务失败。"""
        self.statistics["running_tasks"] -= 1
        self.statistics["failed_tasks"] += 1

        self.logger.error(f"任务 {task.task_id} 执行失败: {task.error_message}")

        # 检查是否需要重试
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.RETRYING
            task.started_at = None
            task.completed_at = None
            task.error_message = None

            # 重新排队（延迟重试）
            retry_delay = self.config.get("retry_delay", 60) * (2 ** (task.retry_count - 1))
            task.scheduled_time = datetime.now() + timedelta(seconds=retry_delay)

            heapq.heappush(self.task_queue, task)
            self.statistics["pending_tasks"] += 1

            self.logger.info(f"任务 {task.task_id} 将在第 {task.retry_count} 次重试，延迟 {retry_delay} 秒")

            # 触发任务重试回调
            await self._trigger_task_callback("task_retrying", task)
        else:
            # 触发任务失败回调
            await self._trigger_task_callback("task_failed", task)

    async def _check_dependent_tasks(self, completed_task_id: str) -> None:
        """检查并触发依赖任务。"""
        dependent_tasks = self.reverse_dependency_graph.get(completed_task_id, set())

        for task_id in dependent_tasks:
            task = self.task_index.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                # 检查所有依赖是否都已完成
                dependencies_met = all(
                    self.task_index.get(dep_id, {}).get("status") == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )

                if dependencies_met:
                    self.logger.info(f"任务 {task_id} 的依赖已满足，可以执行")

    async def _reschedule_periodic_task(self, original_task: ScheduledTask) -> None:
        """重新调度周期任务。"""
        # 创建新的任务实例
        new_task = ScheduledTask(
            task_id=str(uuid.uuid4()),
            pipeline_config=original_task.pipeline_config,
            context=original_task.context,
            schedule_type=ScheduleType.PERIODIC,
            priority=original_task.priority,
            scheduled_time=self._calculate_next_schedule_time(original_task),
            cron_expression=original_task.cron_expression,
            dependencies=original_task.dependencies.copy(),
            max_retries=original_task.max_retries,
            timeout=original_task.timeout,
            metadata=original_task.metadata.copy()
        )

        # 添加到队列
        heapq.heappush(self.task_queue, new_task)
        self.task_index[new_task.task_id] = new_task
        self.statistics["total_tasks"] += 1
        self.statistics["pending_tasks"] += 1

        self.logger.info(f"周期任务已重新调度: {new_task.task_id}")

    def _calculate_next_schedule_time(self, task: ScheduledTask) -> datetime:
        """计算下次调度时间。"""
        if task.cron_expression:
            # 简化的Cron计算
            try:
                parts = task.cron_expression.split()
                if len(parts) >= 2:
                    minute = parts[0]
                    hour = parts[1]

                    now = datetime.now()
                    if minute != "*" and hour != "*":
                        # 每天特定时间
                        next_time = now.replace(
                            hour=int(hour),
                            minute=int(minute),
                            second=0,
                            microsecond=0
                        )
                        if next_time <= now:
                            next_time += timedelta(days=1)
                        return next_time

            except (ValueError, IndexError):
                pass

        # 默认1小时后
        return datetime.now() + timedelta(hours=1)

    async def _cleanup_completed_tasks(self) -> None:
        """清理已完成的任务。"""
        retention_hours = self.config.get("task_retention_hours", 24)
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)

        # 清理任务索引
        completed_task_ids = [
            task_id for task_id, task in self.task_index.items()
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
               task.completed_at and task.completed_at < cutoff_time
        ]

        for task_id in completed_task_ids:
            del self.task_index[task_id]

            # 清理依赖关系
            self.dependency_graph.pop(task_id, None)
            self.reverse_dependency_graph.pop(task_id, None)

        if completed_task_ids:
            self.logger.debug(f"清理了 {len(completed_task_ids)} 个过期任务")

    async def _trigger_task_callback(self, event_name: str, task: ScheduledTask) -> None:
        """触发任务回调。"""
        callbacks = self.task_callbacks.get(event_name, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task)
                else:
                    callback(task)
            except Exception as e:
                self.logger.error(f"任务回调执行失败: {e}")

    def add_task_callback(self, event_name: str, callback: Callable) -> None:
        """添加任务回调。"""
        self.task_callbacks[event_name].append(callback)

    def remove_task_callback(self, event_name: str, callback: Callable) -> None:
        """移除任务回调。"""
        if callback in self.task_callbacks[event_name]:
            self.task_callbacks[event_name].remove(callback)

    def create_resource_pool(
        self,
        name: str,
        max_concurrent_tasks: int,
        cpu_limit: float = 0.8,
        memory_limit: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建资源池。"""
        pool = ResourcePool(
            pool_id=str(uuid.uuid4()),
            name=name,
            max_concurrent_tasks=max_concurrent_tasks,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
            metadata=metadata or {}
        )
        self.resource_pools[pool.pool_id] = pool
        return pool.pool_id

    def get_resource_pool(self, pool_id: str) -> Optional[ResourcePool]:
        """获取资源池。"""
        return self.resource_pools.get(pool_id)

    def get_resource_pools(self) -> List[Dict[str, Any]]:
        """获取所有资源池信息。"""
        return [
            {
                "pool_id": pool.pool_id,
                "name": pool.name,
                "max_concurrent_tasks": pool.max_concurrent_tasks,
                "current_tasks": pool.current_tasks,
                "utilization": pool.utilization,
                "is_available": pool.is_available
            }
            for pool in self.resource_pools.values()
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return self.statistics.copy()

    async def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态。"""
        pending_tasks = [t for t in self.task_index.values() if t.status == TaskStatus.PENDING]
        running_tasks = [t for t in self.task_index.values() if t.status == TaskStatus.RUNNING]

        return {
            "total_tasks": len(self.task_index),
            "pending_tasks": len(pending_tasks),
            "running_tasks": len(running_tasks),
            "queue_length": len(self.task_queue),
            "worker_count": len(self.worker_tasks),
            "resource_pools": len(self.resource_pools),
            "average_wait_time": self._calculate_average_wait_time(pending_tasks)
        }

    def _calculate_average_wait_time(self, pending_tasks: List[ScheduledTask]) -> float:
        """计算平均等待时间。"""
        if not pending_tasks:
            return 0.0

        now = datetime.now()
        total_wait_time = sum(
            (now - task.created_at).total_seconds()
            for task in pending_tasks
        )
        return total_wait_time / len(pending_tasks)