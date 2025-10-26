"""
通用依赖注入函数
提供FastAPI路由中常用的依赖项
"""
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps

from fastapi import Request, HTTPException, Depends, Header
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from infrastructure.monitoring.loguru_logger import logger
from api.exceptions import ValidationError


async def get_current_timestamp() -> str:
    """
    获取当前时间戳

    Returns:
        str: ISO格式的时间戳
    """
    return datetime.utcnow().isoformat() + "Z"


async def get_client_info(
    request: Request,
    x_forwarded_for: Optional[str] = Header(None),
    x_real_ip: Optional[str] = Header(None),
    user_agent: Optional[str] = Header(None)
) -> Dict[str, str]:
    """
    获取客户端信息

    Args:
        request: FastAPI请求对象
        x_forwarded_for: X-Forwarded-For头
        x_real_ip: X-Real-IP头
        user_agent: User-Agent头

    Returns:
        Dict[str, str]: 客户端信息
    """
    # 获取真实IP地址
    if x_forwarded_for:
        # X-Forwarded-For可能包含多个IP，取第一个
        client_ip = x_forwarded_for.split(",")[0].strip()
    elif x_real_ip:
        client_ip = x_real_ip
    elif request.client:
        client_ip = request.client.host
    else:
        client_ip = "unknown"

    return {
        "ip": client_ip,
        "user_agent": user_agent or "unknown",
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path
    }


class RequestSizeValidator:
    """请求大小验证器"""

    def __init__(self, max_size_mb: int = 10):
        """
        初始化请求大小验证器

        Args:
            max_size_mb: 最大请求大小（MB）
        """
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024

    async def __call__(self, request: Request) -> Dict[str, Any]:
        """
        验证请求大小

        Args:
            request: FastAPI请求对象

        Returns:
            Dict[str, Any]: 请求信息

        Raises:
            ValidationError: 请求大小超限
        """
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size_bytes:
                    raise ValidationError(
                        f"请求大小超过限制：{size / (1024 * 1024):.2f}MB > {self.max_size_mb}MB"
                    )
            except ValueError:
                pass

        return {
            "content_length": content_length,
            "max_size_mb": self.max_size_mb
        }


# 创建默认的请求大小验证器
validate_request_size = RequestSizeValidator(max_size_mb=10)


class SimpleRateLimiter:
    """简单的内存速率限制器"""

    def __init__(self, max_requests: int = 100, time_window: int = 60):
        """
        初始化速率限制器

        Args:
            max_requests: 时间窗口内最大请求数
            time_window: 时间窗口（秒）
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}  # {ip: [(timestamp, count), ...]}

    async def __call__(self, request: Request) -> Dict[str, Any]:
        """
        执行速率限制检查

        Args:
            request: FastAPI请求对象

        Returns:
            Dict[str, Any]: 速率限制信息

        Raises:
            HTTPException: 请求频率超限
        """
        client_ip = request.client.host if request.client else "unknown"
        current_time = datetime.utcnow().timestamp()

        # 清理过期记录
        if client_ip in self.requests:
            self.requests[client_ip] = [
                (timestamp, count) for timestamp, count in self.requests[client_ip]
                if current_time - timestamp < self.time_window
            ]
        else:
            self.requests[client_ip] = []

        # 检查请求数量
        total_requests = sum(count for _, count in self.requests[client_ip])
        if total_requests >= self.max_requests:
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail=f"请求频率超限，{self.time_window}秒内最多{self.max_requests}个请求"
            )

        # 记录当前请求
        self.requests[client_ip].append((current_time, 1))

        return {
            "client_ip": client_ip,
            "max_requests": self.max_requests,
            "time_window": self.time_window,
            "current_requests": total_requests + 1
        }


# 创建默认的速率限制器
rate_limiter = SimpleRateLimiter(max_requests=100, time_window=60)


def require_content_type(content_types: list):
    """
    要求特定内容类型的装饰器工厂

    Args:
        content_types: 允许的内容类型列表

    Returns:
        function: 装饰器函数
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                for key, value in kwargs.items():
                    if isinstance(value, Request):
                        request = value
                        break

            if request:
                content_type = request.headers.get("content-type", "").split(";")[0]
                if content_type not in content_types:
                    raise ValidationError(
                        f"不支持的内容类型: {content_type}，支持的类型: {', '.join(content_types)}"
                    )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


async def get_pagination_params(
    page: int = 1,
    page_size: int = 20
) -> Dict[str, int]:
    """
    获取分页参数

    Args:
        page: 页码
        page_size: 每页数量

    Returns:
        Dict[str, int]: 分页参数

    Raises:
        ValidationError: 分页参数无效
    """
    if page < 1:
        raise ValidationError("页码必须大于0")

    if page_size < 1 or page_size > 100:
        raise ValidationError("每页数量必须在1-100之间")

    offset = (page - 1) * page_size

    return {
        "page": page,
        "page_size": page_size,
        "offset": offset
    }


async def validate_id_param(id_param: str) -> str:
    """
    验证ID参数

    Args:
        id_param: ID参数

    Returns:
        str: 验证后的ID

    Raises:
        ValidationError: ID格式无效
    """
    if not id_param or not id_param.strip():
        raise ValidationError("ID不能为空")

    # 简单的UUID格式验证
    import re
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
        re.IGNORECASE
    )

    if not uuid_pattern.match(id_param):
        raise ValidationError("ID格式无效")

    return id_param


class SecurityHeaders:
    """安全头添加器"""

    @staticmethod
    def add_security_headers(response):
        """
        添加安全头到响应

        Args:
            response: HTTP响应对象
        """
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"


def async_lru_cache(maxsize: int = 128):
    """
    简单的异步LRU缓存装饰器

    Args:
        maxsize: 最大缓存大小

    Returns:
        function: 装饰器函数
    """
    def decorator(func):
        cache = {}
        cache_keys = []

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 创建缓存键
            cache_key = str(args) + str(sorted(kwargs.items()))

            # 检查缓存
            if cache_key in cache:
                return cache[cache_key]

            # 执行函数
            result = await func(*args, **kwargs)

            # 添加到缓存
            cache[cache_key] = result
            cache_keys.append(cache_key)

            # 维护缓存大小
            if len(cache_keys) > maxsize:
                oldest_key = cache_keys.pop(0)
                cache.pop(oldest_key, None)

            return result

        wrapper.cache_clear = lambda: cache.clear() or cache_keys.clear()
        wrapper.cache_info = lambda: {
            "size": len(cache),
            "maxsize": maxsize
        }

        return wrapper
    return decorator