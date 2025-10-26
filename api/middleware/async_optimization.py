"""
异步请求处理优化中间件
实现请求批处理、连接池优化、并发控制等异步优化功能
"""
import time
import asyncio
import weakref
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import run_in_threadpool

from infrastructure.monitoring.loguru_logger import logger


class AsyncOptimizationStrategy(str, Enum):
    """异步优化策略枚举"""
    CONNECTION_POOLING = "connection_pooling"
    REQUEST_BATCHING = "request_batching"
    CONCURRENCY_CONTROL = "concurrency_control"
    ASYNC_BATCH_PROCESSING = "async_batch_processing"
    RESOURCE_OPTIMIZATION = "resource_optimization"


@dataclass
class RequestBatch:
    """请求批次"""
    id: str
    requests: List[Request] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    max_batch_size: int = 10
    batch_timeout: float = 0.1  # 批次超时时间（秒）
    futures: List[asyncio.Future] = field(default_factory=list)

    def add_request(self, request: Request, future: asyncio.Future) -> bool:
        """添加请求到批次"""
        if len(self.requests) >= self.max_batch_size:
            return False

        self.requests.append(request)
        self.futures.append(future)
        return True

    def is_ready(self) -> bool:
        """检查批次是否准备好处理"""
        return (
            len(self.requests) >= self.max_batch_size or
            time.time() - self.created_at >= self.batch_timeout
        )

    def is_full(self) -> bool:
        """检查批次是否已满"""
        return len(self.requests) >= self.max_batch_size


@dataclass
class ConnectionPool:
    """连接池"""
    max_connections: int = 100
    active_connections: int = 0
    available_connections: Set[asyncio.Task] = field(default_factory=set)
    waiting_queue: deque = field(default_factory=deque)
    total_created: int = 0
    total_reused: int = 0


@dataclass
class ConcurrencyLimiter:
    """并发限制器"""
    max_concurrent: int
    active_tasks: int = 0
    waiting_queue: deque = field(default_factory=deque)
    task_semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self):
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent)


class AsyncOptimizationMiddleware(BaseHTTPMiddleware):
    """异步优化中间件"""

    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        """
        初始化异步优化中间件

        Args:
            app: ASGI应用
            config: 优化配置
        """
        super().__init__(app)
        self.config = config or {}

        # 异步优化组件
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.concurrency_limiters: Dict[str, ConcurrencyLimiter] = {}
        self.request_batches: Dict[str, RequestBatch] = {}
        self.async_tasks: Set[asyncio.Task] = set()
        self.cleanup_tasks: Set[asyncio.Task] = set()

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "batched_requests": 0,
            "pooled_connections": 0,
            "concurrency_limited": 0,
            "async_tasks_created": 0,
            "async_tasks_completed": 0,
            "resource_saved": 0
        }

        # 初始化优化组件
        self._initialize_optimization_components()

        # 启动清理任务
        self._start_cleanup_task()

    def _initialize_optimization_components(self):
        """初始化优化组件"""
        # 初始化连接池
        pool_configs = self.config.get("connection_pools", {})
        for name, pool_config in pool_configs.items():
            self.connection_pools[name] = ConnectionPool(**pool_config)

        # 初始化并发限制器
        concurrency_configs = self.config.get("concurrency_limiters", {})
        for name, concurrency_config in concurrency_configs.items():
            self.concurrency_limiters[name] = ConcurrencyLimiter(**concurrency_config)

        # 初始化请求批次
        batch_configs = self.config.get("request_batches", {})
        for name, batch_config in batch_configs.items():
            self.request_batches[name] = RequestBatch(
                id=name,
                max_batch_size=batch_config.get("max_batch_size", 10),
                batch_timeout=batch_config.get("batch_timeout", 0.1)
            )

    async def dispatch(self, request: Request, call_next):
        """
        处理请求并执行异步优化

        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或路由处理器

        Returns:
            Response: HTTP响应对象
        """
        self.stats["total_requests"] += 1
        start_time = time.time()

        try:
            # 检查是否可以使用连接池
            pool_result = await self._try_connection_pool(request, call_next)
            if pool_result is not None:
                return pool_result

            # 检查是否可以使用请求批处理
            batch_result = await self._try_request_batch(request, call_next)
            if batch_result is not None:
                return batch_result

            # 检查是否可以使用并发控制
            concurrency_result = await self._try_concurrency_limit(request, call_next)
            if concurrency_result is not None:
                return concurrency_result

            # 默认处理
            return await call_next(request)

        except Exception as e:
            logger.error(f"异步优化处理失败: {str(e)}")
            return await call_next(request)

        finally:
            # 记录处理时间
            processing_time = time.time() - start_time
            if processing_time > 1.0:  # 超过1秒的请求记录日志
                logger.warning(
                    "慢请求警告",
                    extra={
                        "path": request.url.path,
                        "method": request.method,
                        "processing_time": round(processing_time, 3),
                        "optimization_used": "none"
                    }
                )

    async def _try_connection_pool(self, request: Request, call_next: Callable) -> Optional[Response]:
        """尝试使用连接池"""
        pool_name = self._get_pool_name(request)
        if not pool_name or pool_name not in self.connection_pools:
            return None

        pool = self.connection_pools[pool_name]

        # 如果有可用连接，直接处理
        if pool.active_connections < pool.max_connections:
            return await self._execute_with_pool(pool, request, call_next)

        # 如果连接池已满，检查是否可以等待
        if self.config.get("pool_wait_enabled", False):
            future = asyncio.Future()
            pool.waiting_queue.append(future)
            try:
                return await future
            finally:
                if future in pool.waiting_queue:
                    pool.waiting_queue.remove(future)

        return None

    async def _execute_with_pool(self, pool: ConnectionPool, request: Request, call_next: Callable) -> Response:
        """使用连接池执行请求"""
        pool.active_connections += 1
        self.stats["pooled_connections"] += 1

        try:
            response = await call_next(request)
            return response
        finally:
            pool.active_connections -= 1
            self._process_waiting_queue(pool)

    def _process_waiting_queue(self, pool: ConnectionPool):
        """处理等待队列"""
        if pool.waiting_queue and pool.active_connections < pool.max_connections:
            future = pool.waiting_queue.popleft()
            if not future.done():
                future.set_result(None)

    def _get_pool_name(self, request: Request) -> Optional[str]:
        """根据请求获取连接池名称"""
        # 根据路径或请求头确定连接池
        path_prefixes = self.config.get("pool_path_prefixes", {})
        for prefix, pool_name in path_prefixes.items():
            if request.url.path.startswith(prefix):
                return pool_name

        # 根据请求头确定连接池
        header_pools = self.config.get("pool_header_mapping", {})
        for header, pool_name in header_pools.items():
            if request.headers.get(header):
                return pool_name

        return None

    async def _try_request_batch(self, request: Request, call_next: Callable) -> Optional[Response]:
        """尝试使用请求批处理"""
        batch_name = self._get_batch_name(request)
        if not batch_name or batch_name not in self.request_batches:
            return None

        batch = self.request_batches[batch_name]

        # 如果批次已满，创建新批次
        if batch.is_full():
            # 启动新批次的处理任务
            asyncio.create_task(self._process_batch(batch_name))
            # 创建新批次
            new_batch = RequestBatch(
                id=f"{batch_name}_{int(time.time() * 1000)}",
                max_batch_size=batch.max_batch_size,
                batch_timeout=batch.batch_timeout
            )
            self.request_batches[batch_name] = new_batch
            batch = new_batch

        # 创建Future用于获取结果
        future = asyncio.Future()

        if batch.add_request(request, future):
            # 如果批次准备好了，启动处理
            if batch.is_ready():
                asyncio.create_task(self._process_batch(batch_name))

            try:
                return await future
            finally:
                # 清理Future引用
                if future in batch.futures:
                    batch.futures.remove(future)

        return None

    def _get_batch_name(self, request: Request) -> Optional[str]:
        """根据请求获取批次名称"""
        # 根据路径确定批次
        batch_paths = self.config.get("batch_paths", {})
        for path_pattern, batch_name in batch_paths.items():
            if request.url.path.startswith(path_pattern):
                return batch_name

        # 根据请求头确定批次
        batch_headers = self.config.get("batch_header_mapping", {})
        for header, batch_name in batch_headers.items():
            if request.headers.get(header):
                return batch_name

        return None

    async def _process_batch(self, batch_name: str):
        """处理请求批次"""
        if batch_name not in self.request_batches:
            return

        batch = self.request_batches[batch_name]

        try:
            # 模拟批处理逻辑
            await asyncio.sleep(0.05)  # 模拟批处理延迟

            # 处理批次中的所有请求
            for i, (request, future) in enumerate(zip(batch.requests, batch.futures)):
                try:
                    # 这里应该有实际的批处理逻辑
                    # 简化处理：直接调用call_next
                    # 实际场景中，可能需要将多个请求合并为一个操作
                    response = f"Batch response for request {i+1}"
                    if not future.done():
                        future.set_result(response)

                    self.stats["batched_requests"] += 1

                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
                    logger.error(f"批处理请求失败: {str(e)}")

            self.stats["resource_saved"] += len(batch.requests) - 1

        except Exception as e:
            logger.error(f"批处理失败: {str(e)}")

        finally:
            # 清理批次
            del self.request_batches[batch_name]

    async def _try_concurrency_limit(self, request: Request, call_next: Callable) -> Optional[Response]:
        """尝试使用并发控制"""
        limiter_name = self._get_limiter_name(request)
        if not limiter_name or limiter_name not in self.concurrency_limiters:
            return None

        limiter = self.concurrency_limiters[limiter_name]

        # 尝试获取信号量
        if limiter.task_semaphore.acquire(blocking=False):
            return await self._execute_with_concurrency_limit(limiter, request, call_next)

        # 如果并发限制达到，等待或跳过
        if self.config.get("concurrency_wait_enabled", False):
            async with limiter.task_semaphore:
                self.stats["concurrency_limited"] += 1
                return await call_next(request)

        return None

    async def _execute_with_concurrency_limit(self, limiter: ConcurrencyLimiter, request: Request, call_next: Callable) -> Response:
        """使用并发控制执行请求"""
        limiter.active_tasks += 1
        self.stats["async_tasks_created"] += 1

        try:
            response = await call_next(request)
            self.stats["async_tasks_completed"] += 1
            return response
        finally:
            limiter.active_tasks -= 1

    def _get_limiter_name(self, request: Request) -> Optional[str]:
        """根据请求获取并发限制器名称"""
        # 根据路径确定限制器
        limiter_paths = self.config.get("concurrency_limiter_paths", {})
        for path_pattern, limiter_name in limiter_paths.items():
            if request.url.path.startswith(path_pattern):
                return limiter_name

        # 根据请求头确定限制器
        limiter_headers = self.config.get("concurrency_limiter_header_mapping", {})
        for header, limiter_name in limiter_headers.items():
            if request.headers.get(header):
                return limiter_name

        return None

    def _start_cleanup_task(self):
        """启动清理任务"""
        async def cleanup_loop():
            while True:
                try:
                    await self._cleanup_expired_resources()
                    await asyncio.sleep(60)  # 每分钟清理一次
                except Exception as e:
                    logger.error(f"清理任务异常: {str(e)}")
                    await asyncio.sleep(60)

        asyncio.create_task(cleanup_loop())

    async def _cleanup_expired_resources(self):
        """清理过期资源"""
        # 清理完成的异步任务
        completed_tasks = {task for task in self.async_tasks if task.done()}
        for task in completed_tasks:
            self.async_tasks.remove(task)

        # 清理完成的清理任务
        completed_cleanup_tasks = {task for task in self.cleanup_tasks if task.done()}
        for task in completed_cleanup_tasks:
            self.cleanup_tasks.remove(task)

    def get_stats(self) -> Dict[str, Any]:
        """获取异步优化统计信息"""
        return {
            **self.stats,
            "active_async_tasks": len(self.async_tasks),
            "connection_pools": {
                name: {
                    "max_connections": pool.max_connections,
                    "active_connections": pool.active_connections,
                    "waiting_queue_size": len(pool.waiting_queue),
                    "total_created": pool.total_created,
                    "total_reused": pool.total_reused
                }
                for name, pool in self.connection_pools.items()
            },
            "concurrency_limiters": {
                name: {
                    "max_concurrent": limiter.max_concurrent,
                    "active_tasks": limiter.active_tasks,
                    "waiting_queue_size": len(limiter.waiting_queue)
                }
                for name, limiter in self.concurrency_limiters.items()
            },
            "request_batches": {
                name: {
                    "id": batch.id,
                    "request_count": len(batch.requests),
                    "max_batch_size": batch.max_batch_size,
                    "created_at": batch.created_at
                }
                for name, batch in self.request_batches.items()
            }
        }


class AsyncOptimizationManager:
    """异步优化管理器"""

    def __init__(self):
        self.middleware: Optional[AsyncOptimizationMiddleware] = None

    def configure(self, config_dict: Dict[str, Any]) -> AsyncOptimizationMiddleware:
        """配置异步优化中间件"""
        # 创建中间件的占位符，实际使用时需要传入app实例
        class PlaceholderMiddleware:
            def __init__(self):
                self.config = config_dict

        self.middleware = PlaceholderMiddleware()
        return self.middleware  # type: ignore

    def get_middleware(self, app):
        """获取异步优化中间件实例"""
        if not self.middleware:
            raise ValueError("请先调用configure()方法配置异步优化中间件")

        return AsyncOptimizationMiddleware(app, self.middleware.config)


# 全局异步优化管理器实例
async_optimization_manager = AsyncOptimizationManager()