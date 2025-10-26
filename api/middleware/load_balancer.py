"""
负载均衡中间件
实现请求分发和负载均衡功能
"""
import time
import asyncio
import random
import hashlib
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from infrastructure.monitoring.loguru_logger import logger


class LoadBalancingStrategy(str, Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    RANDOM = "random"


@dataclass
class BackendServer:
    """后端服务器信息"""
    id: str
    host: str
    port: int
    weight: int = 1
    max_connections: int = 1000
    health_check_url: str = "/health"
    is_healthy: bool = True
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: float = 0.0
    last_request_time: float = 0.0


@dataclass
class LoadBalancerConfig:
    """负载均衡配置"""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    health_check_interval: int = 30  # 秒
    health_check_timeout: int = 5  # 秒
    max_retries: int = 3
    retry_delay: float = 1.0  # 秒
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60  # 秒
    enable_sticky_sessions: bool = False
    session_cookie_name: str = "lb_session"


class LoadBalancerMiddleware(BaseHTTPMiddleware):
    """负载均衡中间件"""

    def __init__(self, app, config: LoadBalancerConfig):
        """
        初始化负载均衡中间件

        Args:
            app: ASGI应用
            config: 负载均衡配置
        """
        super().__init__(app)
        self.config = config

        # 后端服务器列表
        self.backends: List[BackendServer] = []
        self.backend_map: Dict[str, BackendServer] = {}

        # 负载均衡状态
        self.round_robin_index = 0
        self.weighted_round_robin_weights: List[str] = []
        self.circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "failures": 0,
            "last_failure": 0,
            "state": "closed"  # closed, open, half_open
        })

        # 会话粘性
        self.session_affinity: Dict[str, str] = {}

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retries": 0,
            "health_checks": 0,
            "backend_requests": defaultdict(int)
        }

        # 启动健康检查任务
        self._start_health_check_task()

    def add_backend(self, backend: BackendServer):
        """添加后端服务器"""
        self.backends.append(backend)
        self.backend_map[backend.id] = backend
        self._update_weighted_round_robin_weights()
        logger.info(f"添加后端服务器: {backend.id} ({backend.host}:{backend.port})")

    def remove_backend(self, backend_id: str):
        """移除后端服务器"""
        if backend_id in self.backend_map:
            backend = self.backend_map[backend_id]
            self.backends.remove(backend)
            del self.backend_map[backend_id]
            self._update_weighted_round_robin_weights()
            logger.info(f"移除后端服务器: {backend_id}")

    def _update_weighted_round_robin_weights(self):
        """更新加权轮询权重"""
        self.weighted_round_robin_weights = []
        for backend in self.backends:
            if backend.is_healthy:
                self.weighted_round_robin_weights.extend([backend.id] * backend.weight)

    async def dispatch(self, request: Request, call_next):
        """
        处理请求并执行负载均衡

        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或路由处理器

        Returns:
            Response: HTTP响应对象
        """
        self.stats["total_requests"] += 1
        start_time = time.time()

        # 检查会话粘性
        backend = self._get_backend_for_session(request)

        # 如果没有会话粘性或后端不可用，使用负载均衡策略选择后端
        if not backend or not backend.is_healthy:
            backend = self._select_backend(request)

        if not backend:
            logger.error("没有可用的后端服务器")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="服务暂时不可用，没有可用的后端服务器"
            )

        # 记录请求开始
        backend.current_connections += 1
        backend.total_requests += 1
        backend.last_request_time = start_time
        self.stats["backend_requests"][backend.id] += 1

        try:
            # 检查熔断器
            if not self._is_circuit_breaker_available(backend.id):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"后端服务器 {backend.id} 熔断器开启"
                )

            # 执行请求
            response = await call_next(request)

            # 更新统计信息
            response_time = time.time() - start_time
            self._update_backend_stats(backend, response_time, True)

            # 设置会话粘性cookie
            if self.config.enable_sticky_sessions:
                response.set_cookie(
                    self.config.session_cookie_name,
                    backend.id,
                    max_age=3600,
                    httponly=True,
                    samesite="lax"
                )

            self.stats["successful_requests"] += 1
            return response

        except Exception as e:
            # 更新错误统计
            response_time = time.time() - start_time
            self._update_backend_stats(backend, response_time, False)
            self._update_circuit_breaker(backend.id)

            self.stats["failed_requests"] += 1
            logger.error(f"请求失败: {str(e)}, 后端: {backend.id}")

            # 尝试重试
            return await self._retry_request(request, call_next, [backend.id])

        finally:
            backend.current_connections -= 1

    def _get_backend_for_session(self, request: Request) -> Optional[BackendServer]:
        """根据会话获取后端服务器"""
        if not self.config.enable_sticky_sessions:
            return None

        session_id = request.cookies.get(self.config.session_cookie_name)
        if session_id and session_id in self.backend_map:
            backend = self.backend_map[session_id]
            if backend.is_healthy:
                return backend

        return None

    def _select_backend(self, request: Request) -> Optional[BackendServer]:
        """根据策略选择后端服务器"""
        healthy_backends = [b for b in self.backends if b.is_healthy]

        if not healthy_backends:
            return None

        if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(healthy_backends)
        elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._select_weighted_round_robin(healthy_backends)
        elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._select_least_connections(healthy_backends)
        elif self.config.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._select_least_response_time(healthy_backends)
        elif self.config.strategy == LoadBalancingStrategy.IP_HASH:
            return self._select_ip_hash(healthy_backends, request)
        elif self.config.strategy == LoadBalancingStrategy.RANDOM:
            return self._select_random(healthy_backends)
        else:
            return healthy_backends[0]

    def _select_round_robin(self, backends: List[BackendServer]) -> BackendServer:
        """轮询选择"""
        backend = backends[self.round_robin_index % len(backends)]
        self.round_robin_index += 1
        return backend

    def _select_weighted_round_robin(self, backends: List[BackendServer]) -> BackendServer:
        """加权轮询选择"""
        if not self.weighted_round_robin_weights:
            return backends[0]

        backend_id = self.weighted_round_robin_weights[
            self.round_robin_index % len(self.weighted_round_robin_weights)
        ]
        self.round_robin_index += 1

        return self.backend_map[backend_id]

    def _select_least_connections(self, backends: List[BackendServer]) -> BackendServer:
        """最少连接选择"""
        return min(backends, key=lambda b: b.current_connections)

    def _select_least_response_time(self, backends: List[BackendServer]) -> BackendServer:
        """最短响应时间选择"""
        return min(backends, key=lambda b: b.avg_response_time)

    def _select_ip_hash(self, backends: List[BackendServer], request: Request) -> BackendServer:
        """IP哈希选择"""
        client_ip = self._get_client_ip(request)
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        index = hash_value % len(backends)
        return backends[index]

    def _select_random(self, backends: List[BackendServer]) -> BackendServer:
        """随机选择"""
        return random.choice(backends)

    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        if request.client:
            return request.client.host

        return "unknown"

    async def _retry_request(self, request: Request, call_next: Callable, tried_backends: List[str]) -> Response:
        """重试请求"""
        for attempt in range(self.config.max_retries):
            if attempt > 0:
                await asyncio.sleep(self.config.retry_delay * (2 ** (attempt - 1)))  # 指数退避

            backend = self._select_backend(request)
            if not backend or backend.id in tried_backends:
                continue

            tried_backends.append(backend.id)
            backend.current_connections += 1
            backend.total_requests += 1

            try:
                if not self._is_circuit_breaker_available(backend.id):
                    continue

                start_time = time.time()
                response = await call_next(request)
                response_time = time.time() - start_time

                self._update_backend_stats(backend, response_time, True)
                self.stats["successful_requests"] += 1
                self.stats["retries"] += 1

                return response

            except Exception as e:
                response_time = time.time() - start_time
                self._update_backend_stats(backend, response_time, False)
                self._update_circuit_breaker(backend.id)
                logger.warning(f"重试请求失败: {str(e)}, 后端: {backend.id}, 尝试: {attempt + 1}")

            finally:
                backend.current_connections -= 1

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="所有后端服务器都不可用"
        )

    def _update_backend_stats(self, backend: BackendServer, response_time: float, success: bool):
        """更新后端统计信息"""
        if success:
            # 更新平均响应时间（使用指数移动平均）
            if backend.avg_response_time == 0:
                backend.avg_response_time = response_time
            else:
                alpha = 0.3  # 平滑因子
                backend.avg_response_time = (
                    alpha * response_time + (1 - alpha) * backend.avg_response_time
                )
        else:
            backend.failed_requests += 1

    def _update_circuit_breaker(self, backend_id: str):
        """更新熔断器状态"""
        circuit_breaker = self.circuit_breakers[backend_id]
        circuit_breaker["failures"] += 1
        circuit_breaker["last_failure"] = time.time()

        # 检查是否需要开启熔断器
        if circuit_breaker["failures"] >= self.config.circuit_breaker_threshold:
            circuit_breaker["state"] = "open"
            logger.warning(f"熔断器开启: {backend_id}")

    def _is_circuit_breaker_available(self, backend_id: str) -> bool:
        """检查熔断器是否可用"""
        circuit_breaker = self.circuit_breakers[backend_id]

        if circuit_breaker["state"] == "closed":
            return True
        elif circuit_breaker["state"] == "open":
            # 检查是否可以进入半开状态
            if time.time() - circuit_breaker["last_failure"] > self.config.circuit_breaker_timeout:
                circuit_breaker["state"] = "half_open"
                logger.info(f"熔断器半开: {backend_id}")
                return True
            return False
        elif circuit_breaker["state"] == "half_open":
            # 半开状态允许一些请求通过
            return True

        return False

    def _start_health_check_task(self):
        """启动健康检查任务"""
        async def health_check_loop():
            while True:
                try:
                    await self._perform_health_checks()
                    await asyncio.sleep(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"健康检查异常: {str(e)}")
                    await asyncio.sleep(5)  # 出错时等待5秒后重试

        # 创建后台任务
        asyncio.create_task(health_check_loop())

    async def _perform_health_checks(self):
        """执行健康检查"""
        for backend in self.backends:
            try:
                is_healthy = await self._check_backend_health(backend)
                old_health = backend.is_healthy
                backend.is_healthy = is_healthy
                backend.last_health_check = time.time()

                # 健康状态变化时记录日志
                if old_health != is_healthy:
                    if is_healthy:
                        logger.info(f"后端服务器恢复健康: {backend.id}")
                        # 重置熔断器
                        self.circuit_breakers[backend.id] = {
                            "failures": 0,
                            "last_failure": 0,
                            "state": "closed"
                        }
                    else:
                        logger.warning(f"后端服务器不健康: {backend.id}")

                self.stats["health_checks"] += 1

            except Exception as e:
                logger.error(f"健康检查失败: {backend.id}, 错误: {str(e)}")
                backend.is_healthy = False

    async def _check_backend_health(self, backend: BackendServer) -> bool:
        """检查单个后端服务器的健康状态"""
        # 这里简化实现，实际应该发送HTTP请求到健康检查端点
        # 模拟健康检查
        await asyncio.sleep(0.1)

        # 模拟健康状态（90%概率健康）
        import random
        return random.random() > 0.1

    def get_stats(self) -> Dict[str, Any]:
        """获取负载均衡统计信息"""
        backend_stats = []
        for backend in self.backends:
            backend_stats.append({
                "id": backend.id,
                "host": f"{backend.host}:{backend.port}",
                "is_healthy": backend.is_healthy,
                "current_connections": backend.current_connections,
                "total_requests": backend.total_requests,
                "failed_requests": backend.failed_requests,
                "avg_response_time": round(backend.avg_response_time, 3),
                "success_rate": round(
                    (backend.total_requests - backend.failed_requests) / max(1, backend.total_requests) * 100, 2
                )
            })

        return {
            **self.stats,
            "strategy": self.config.strategy,
            "total_backends": len(self.backends),
            "healthy_backends": len([b for b in self.backends if b.is_healthy]),
            "backend_stats": backend_stats,
            "circuit_breakers": dict(self.circuit_breakers)
        }


class LoadBalancerManager:
    """负载均衡管理器"""

    def __init__(self):
        self.middleware: Optional[LoadBalancerMiddleware] = None
        self.backends: List[BackendServer] = []

    def configure(self, config_dict: Dict[str, Any]) -> LoadBalancerMiddleware:
        """配置负载均衡"""
        config = LoadBalancerConfig(**config_dict)

        # 创建中间件的占位符，实际使用时需要传入app实例
        class PlaceholderMiddleware:
            def __init__(self):
                self.config = config

        self.middleware = PlaceholderMiddleware()
        return self.middleware  # type: ignore

    def add_backend(self, host: str, port: int, **kwargs):
        """添加后端服务器"""
        backend = BackendServer(
            id=f"{host}:{port}",
            host=host,
            port=port,
            **kwargs
        )
        self.backends.append(backend)
        return backend

    def get_middleware(self, app):
        """获取负载均衡中间件实例"""
        if not self.middleware:
            raise ValueError("请先调用configure()方法配置负载均衡")

        middleware = LoadBalancerMiddleware(app, self.middleware.config)

        # 添加配置的后端服务器
        for backend in self.backends:
            middleware.add_backend(backend)

        return middleware


# 全局负载均衡管理器实例
load_balancer_manager = LoadBalancerManager()