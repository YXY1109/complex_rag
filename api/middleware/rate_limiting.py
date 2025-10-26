"""
请求限流中间件
实现基于不同策略的请求限流功能
"""
import time
import asyncio
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from infrastructure.monitoring.loguru_logger import logger


class RateLimitStrategy(str, Enum):
    """限流策略枚举"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """限流配置"""
    requests_per_window: int
    window_seconds: int
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_size: Optional[int] = None  # 突发请求大小
    refill_rate: Optional[float] = None  # 令牌桶补充速率


@dataclass
class ClientInfo:
    """客户端信息"""
    request_count: int = 0
    window_start: float = 0
    last_request_time: float = 0
    tokens: float = 0  # 令牌桶令牌数量
    last_refill: float = 0  # 最后补充时间
    queue: deque = None  # 漏桶队列

    def __post_init__(self):
        if self.queue is None:
            self.queue = deque()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """请求限流中间件"""

    def __init__(
        self,
        app,
        default_config: RateLimitConfig,
        path_configs: Optional[Dict[str, RateLimitConfig]] = None,
        ip_whitelist: Optional[List[str]] = None,
        redis_enabled: bool = False
    ):
        """
        初始化限流中间件

        Args:
            app: ASGI应用
            default_config: 默认限流配置
            path_configs: 路径特定的限流配置
            ip_whitelist: IP白名单
            redis_enabled: 是否启用Redis分布式限流
        """
        super().__init__(app)
        self.default_config = default_config
        self.path_configs = path_configs or {}
        self.ip_whitelist = set(ip_whitelist or [])
        self.redis_enabled = redis_enabled

        # 内存存储（非分布式）
        self.clients: Dict[str, Dict[str, ClientInfo]] = defaultdict(lambda: defaultdict(ClientInfo))

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "limited_clients": set(),
            "window_starts": defaultdict(float)
        }

        # Redis客户端（如果启用）
        self.redis_client = None
        if redis_enabled:
            self._init_redis_client()

    def _init_redis_client(self):
        """初始化Redis客户端"""
        try:
            from infrastructure.cache.implementations.redis_cache_adapter import RedisCache
            self.redis_client = RedisCache()
            logger.info("Redis分布式限流已启用")
        except Exception as e:
            logger.warning(f"Redis初始化失败，使用内存限流: {str(e)}")
            self.redis_enabled = False

    async def dispatch(self, request: Request, call_next):
        """
        处理请求并执行限流检查

        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或路由处理器

        Returns:
            Response: HTTP响应对象
        """
        self.stats["total_requests"] += 1

        # 获取客户端标识
        client_id = self._get_client_id(request)

        # 检查IP白名单
        if self._is_whitelisted(client_id):
            return await call_next(request)

        # 获取适用的限流配置
        config = self._get_config_for_path(request.url.path)

        # 执行限流检查
        if not await self._check_rate_limit(client_id, config, request.url.path):
            self.stats["blocked_requests"] += 1
            self.stats["limited_clients"].add(client_id)

            # 记录限流日志
            logger.warning(
                "请求被限流",
                extra={
                    "client_id": client_id,
                    "path": request.url.path,
                    "method": request.method,
                    "config": {
                        "requests": config.requests_per_window,
                        "window": config.window_seconds,
                        "strategy": config.strategy
                    }
                }
            )

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"请求频率超过限制，{config.window_seconds}秒内最多{config.requests_per_window}个请求",
                    "retry_after": config.window_seconds,
                    "limit": config.requests_per_window,
                    "window": config.window_seconds,
                    "strategy": config.strategy
                },
                headers={
                    "Retry-After": str(config.window_seconds),
                    "X-RateLimit-Limit": str(config.requests_per_window),
                    "X-RateLimit-Window": str(config.window_seconds),
                    "X-RateLimit-Strategy": config.strategy
                }
            )

        # 执行请求
        response = await call_next(request)

        # 添加限流信息头
        client_info = self._get_client_info(client_id, config, request.url.path)
        response.headers["X-RateLimit-Limit"] = str(config.requests_per_window)
        response.headers["X-RateLimit-Window"] = str(config.window_seconds)
        response.headers["X-RateLimit-Strategy"] = config.strategy

        if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            response.headers["X-RateLimit-Remaining"] = str(int(client_info.tokens))
        else:
            remaining = max(0, config.requests_per_window - client_info.request_count)
            response.headers["X-RateLimit-Remaining"] = str(remaining)

        response.headers["X-RateLimit-Reset"] = str(
            int(client_info.window_start + config.window_seconds - time.time())
        )

        return response

    def _get_client_id(self, request: Request) -> str:
        """获取客户端标识"""
        # 优先使用X-Forwarded-For头
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # 使用X-Real-IP头
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # 使用客户端IP
        if request.client:
            return request.client.host

        # 使用User-Agent作为后备
        user_agent = request.headers.get("user-agent", "unknown")
        return f"ua:{hash(user_agent)}"

    def _is_whitelisted(self, client_id: str) -> bool:
        """检查客户端是否在白名单中"""
        return client_id in self.ip_whitelist

    def _get_config_for_path(self, path: str) -> RateLimitConfig:
        """根据路径获取限流配置"""
        # 精确匹配
        if path in self.path_configs:
            return self.path_configs[path]

        # 前缀匹配
        for path_pattern, config in self.path_configs.items():
            if path.startswith(path_pattern):
                return config

        # 使用默认配置
        return self.default_config

    def _get_client_info(self, client_id: str, config: RateLimitConfig, path: str) -> ClientInfo:
        """获取客户端信息"""
        if self.redis_enabled:
            return self._get_client_info_redis(client_id, config, path)
        else:
            return self._get_client_info_memory(client_id, config, path)

    def _get_client_info_memory(self, client_id: str, config: RateLimitConfig, path: str) -> ClientInfo:
        """获取内存中的客户端信息"""
        return self.clients[client_id][path]

    def _get_client_info_redis(self, client_id: str, config: RateLimitConfig, path: str) -> ClientInfo:
        """获取Redis中的客户端信息"""
        # 这里简化实现，实际应该使用Redis存储
        return self._get_client_info_memory(client_id, config, path)

    async def _check_rate_limit(self, client_id: str, config: RateLimitConfig, path: str) -> bool:
        """检查限流"""
        current_time = time.time()
        client_info = self._get_client_info(client_id, config, path)

        if config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._check_fixed_window(client_info, config, current_time)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._check_sliding_window(client_info, config, current_time)
        elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(client_info, config, current_time)
        elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return self._check_leaky_bucket(client_info, config, current_time)
        else:
            return True

    def _check_fixed_window(self, client_info: ClientInfo, config: RateLimitConfig, current_time: float) -> bool:
        """检查固定窗口限流"""
        # 检查是否需要重置窗口
        if current_time - client_info.window_start >= config.window_seconds:
            client_info.request_count = 0
            client_info.window_start = current_time

        # 检查请求数量
        if client_info.request_count >= config.requests_per_window:
            return False

        client_info.request_count += 1
        return True

    def _check_sliding_window(self, client_info: ClientInfo, config: RateLimitConfig, current_time: float) -> bool:
        """检查滑动窗口限流"""
        # 初始化窗口开始时间
        if client_info.window_start == 0:
            client_info.window_start = current_time

        # 维护请求时间队列
        window_start = current_time - config.window_seconds

        # 清理过期的请求记录
        while client_info.queue and client_info.queue[0] < window_start:
            client_info.queue.popleft()
            client_info.request_count -= 1

        # 检查请求数量
        if client_info.request_count >= config.requests_per_window:
            return False

        # 添加当前请求
        client_info.queue.append(current_time)
        client_info.request_count += 1
        client_info.last_request_time = current_time

        return True

    async def _check_token_bucket(self, client_info: ClientInfo, config: RateLimitConfig, current_time: float) -> bool:
        """检查令牌桶限流"""
        # 初始化令牌桶
        if client_info.tokens == 0:
            client_info.tokens = float(config.requests_per_window)
            client_info.last_refill = current_time

        # 补充令牌
        if client_info.last_refill > 0:
            time_passed = current_time - client_info.last_refill
            refill_rate = config.refill_rate or (config.requests_per_window / config.window_seconds)
            tokens_to_add = time_passed * refill_rate

            client_info.tokens = min(
                float(config.requests_per_window),
                client_info.tokens + tokens_to_add
            )
            client_info.last_refill = current_time

        # 检查是否有足够的令牌
        burst_size = config.burst_size or 1
        if client_info.tokens < burst_size:
            return False

        # 消耗令牌
        client_info.tokens -= burst_size
        client_info.last_request_time = current_time

        return True

    def _check_leaky_bucket(self, client_info: ClientInfo, config: RateLimitConfig, current_time: float) -> bool:
        """检查漏桶限流"""
        # 初始化漏桶
        if client_info.window_start == 0:
            client_info.window_start = current_time
            client_info.last_request_time = current_time

        # 计算漏出速率
        leak_rate = config.requests_per_window / config.window_seconds
        time_passed = current_time - client_info.last_request_time

        # 漏出请求
        leaked = time_passed * leak_rate
        while client_info.queue and leaked >= 1:
            client_info.queue.popleft()
            client_info.request_count -= 1
            leaked -= 1

        # 检查桶是否已满
        if client_info.request_count >= config.requests_per_window:
            return False

        # 添加请求到桶
        client_info.queue.append(current_time)
        client_info.request_count += 1
        client_info.last_request_time = current_time

        return True

    def get_stats(self) -> Dict:
        """获取限流统计信息"""
        return {
            **self.stats,
            "limited_clients_count": len(self.stats["limited_clients"]),
            "total_clients": len(self.clients),
            "block_rate": (
                self.stats["blocked_requests"] / max(1, self.stats["total_requests"])
            )
        }

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "limited_clients": set(),
            "window_starts": defaultdict(float)
        }

    def clear_client_data(self, client_id: Optional[str] = None):
        """清理客户端数据"""
        if client_id:
            if client_id in self.clients:
                del self.clients[client_id]
        else:
            self.clients.clear()


class RateLimitManager:
    """限流管理器"""

    def __init__(self):
        self.middleware: Optional[RateLimitMiddleware] = None

    def configure(self, middleware_config: Dict[str, Any]) -> RateLimitMiddleware:
        """配置限流中间件"""
        default_config = RateLimitConfig(**middleware_config.get("default", {}))

        path_configs = {}
        for path, config in middleware_config.get("paths", {}).items():
            path_configs[path] = RateLimitConfig(**config)

        ip_whitelist = middleware_config.get("ip_whitelist", [])
        redis_enabled = middleware_config.get("redis_enabled", False)

        # 创建中间件的占位符，实际使用时需要传入app实例
        class PlaceholderMiddleware:
            def __init__(self):
                self.default_config = default_config
                self.path_configs = path_configs
                self.ip_whitelist = ip_whitelist
                self.redis_enabled = redis_enabled

        self.middleware = PlaceholderMiddleware()
        return self.middleware  # type: ignore

    def get_middleware(self, app):
        """获取限流中间件实例"""
        if not self.middleware:
            raise ValueError("请先调用configure()方法配置限流中间件")

        return RateLimitMiddleware(
            app,
            self.middleware.default_config,
            self.middleware.path_configs,
            self.middleware.ip_whitelist,
            self.middleware.redis_enabled
        )


# 全局限流管理器实例
rate_limit_manager = RateLimitManager()