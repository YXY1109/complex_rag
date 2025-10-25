"""
Sanic中间件配置

提供请求处理、性能监控、日志记录等中间件功能。
"""

import time
import uuid
from typing import Callable, Optional

from sanic import Sanic, Request, Response
from sanic.response import json
from sanic.log import logger
from sanic.exceptions import SanicException

from ..infrastructure.monitoring.loguru_logger import get_logger


# 获取结构化日志记录器
structured_logger = get_logger("rag_service.middleware")


def setup_middleware(app: Sanic) -> None:
    """
    设置所有中间件

    Args:
        app: Sanic应用实例
    """
    # 请求ID中间件
    @app.middleware("request")
    async def add_request_id(request: Request):
        """添加请求ID"""
        request.ctx.request_id = str(uuid.uuid4())
        request.ctx.start_time = time.time()

    # 请求日志中间件
    @app.middleware("request")
    async def log_request(request: Request):
        """记录请求日志"""
        structured_logger.info(
            "请求开始",
            extra={
                "request_id": getattr(request.ctx, "request_id", "unknown"),
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "client_ip": request.ip,
                "user_agent": request.headers.get("user-agent", ""),
            }
        )

    # 限流中间件
    @app.middleware("request")
    async def rate_limiting(request: Request):
        """简单限流中间件"""
        # 这里可以实现更复杂的限流逻辑
        # 例如使用Redis进行分布式限流
        pass

    # 响应中间件
    @app.middleware("response")
    async def log_response(request: Request, response: Response):
        """记录响应日志"""
        end_time = time.time()
        duration = end_time - getattr(request.ctx, "start_time", end_time)

        structured_logger.info(
            "请求完成",
            extra={
                "request_id": getattr(request.ctx, "request_id", "unknown"),
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status,
                "duration_ms": round(duration * 1000, 2),
                "response_size": len(response.body) if hasattr(response, 'body') else 0,
            }
        )

    # CORS头中间件（补充sanic-cors）
    @app.middleware("response")
    async def add_cors_headers(request: Request, response: Response):
        """添加CORS响应头"""
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "86400"

    # 安全头中间件
    @app.middleware("response")
    async def add_security_headers(request: Request, response: Response):
        """添加安全响应头"""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # 生产环境添加HSTS
        if not app.debug:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # API版本头中间件
    @app.middleware("response")
    async def add_api_version_headers(request: Request, response: Response):
        """添加API版本信息"""
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Service-Name"] = "rag-service"

    # 请求验证中间件
    @app.middleware("request")
    async def validate_content_type(request: Request):
        """验证请求内容类型"""
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")

            # 对于API路由，验证JSON内容类型
            if request.path.startswith("/v1/") and "application/json" not in content_type:
                # 允许multipart/form-data（用于文件上传）
                if "multipart/form-data" not in content_type:
                    logger.warning(
                        f"可能不支持的内容类型: {content_type}",
                        extra={"request_id": getattr(request.ctx, "request_id", "unknown")}
                    )

    # 健康检查中间件
    @app.middleware("request")
    async def health_check_bypass(request: Request):
        """健康检查路径绕过其他中间件"""
        if request.path in ["/health", "/healthz", "/ping"]:
            request.ctx.bypass_auth = True
            request.ctx.bypass_rate_limit = True

    # 性能监控中间件
    @app.middleware("response")
    async def performance_monitoring(request: Request, response: Response):
        """性能监控中间件"""
        duration = time.time() - getattr(request.ctx, "start_time", time.time())

        # 记录慢请求
        if duration > 5.0:  # 超过5秒的请求
            structured_logger.warning(
                "慢请求检测",
                extra={
                    "request_id": getattr(request.ctx, "request_id", "unknown"),
                    "method": request.method,
                    "url": str(request.url),
                    "duration_seconds": round(duration, 2),
                    "status_code": response.status,
                }
            )

    # 错误处理中间件
    @app.middleware("request")
    async def error_handling_preprocess(request: Request):
        """错误预处理中间件"""
        request.ctx.errors = []

    @app.middleware("response")
    async def error_handling_postprocess(request: Request, response: Response):
        """错误后处理中间件"""
        if hasattr(request.ctx, 'errors') and request.ctx.errors:
            structured_logger.warning(
                "请求处理过程中发生错误",
                extra={
                    "request_id": getattr(request.ctx, "request_id", "unknown"),
                    "errors": request.ctx.errors,
                }
            )

    # 缓存控制中间件
    @app.middleware("response")
    async def cache_control(request: Request, response: Response):
        """设置缓存控制头"""
        if request.method == "GET":
            # 对于GET请求，设置合理的缓存策略
            if request.path.startswith("/v1/models"):
                # 模型信息可以缓存较长时间
                response.headers["Cache-Control"] = "public, max-age=3600"
            elif request.path.startswith("/health"):
                # 健康检查可以短时间缓存
                response.headers["Cache-Control"] = "public, max-age=30"
            else:
                # 其他GET请求不缓存
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        else:
            # 非GET请求不缓存
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

    # 压缩中间件（对于大响应）
    @app.middleware("response")
    async def compression_check(request: Request, response: Response):
        """检查是否需要压缩响应"""
        if hasattr(response, 'body') and len(response.body) > 1024:  # 大于1KB
            accept_encoding = request.headers.get("accept-encoding", "")
            if "gzip" in accept_encoding:
                response.headers["Content-Encoding"] = "gzip"
                # 注意：实际的压缩逻辑需要根据响应类型实现

    structured_logger.info("中间件设置完成", extra={"service": "rag_service"})


# 高级中间件函数
def create_rate_limiter(max_requests: int = 100, window_seconds: int = 60) -> Callable:
    """
    创建限流中间件

    Args:
        max_requests: 时间窗口内最大请求数
        window_seconds: 时间窗口（秒）

    Returns:
        Callable: 限流中间件函数
    """
    # 这里可以实现基于Redis的分布式限流
    # 暂时返回一个空的中间件

    async def rate_limit_middleware(request: Request):
        """限流中间件"""
        # TODO: 实现具体的限流逻辑
        pass

    return rate_limit_middleware


def create_auth_middleware(required_token: Optional[str] = None) -> Callable:
    """
    创建认证中间件

    Args:
        required_token: 必需的认证令牌

    Returns:
        Callable: 认证中间件函数
    """
    async def auth_middleware(request: Request):
        """认证中间件"""
        # 跳过健康检查和公开端点
        if (hasattr(request.ctx, 'bypass_auth') and request.ctx.bypass_auth or
            request.path in ["/", "/health", "/healthz", "/ping", "/v1/openapi.json"]):
            return

        # 检查Authorization头
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            if required_token:
                return json(
                    {"error": "Missing or invalid authorization header"},
                    status=401
                )

        # 这里可以实现更复杂的认证逻辑
        # 例如JWT验证、API密钥验证等

    return auth_middleware


def create_metrics_middleware() -> Callable:
    """
    创建指标收集中间件

    Returns:
        Callable: 指标收集中间件函数
    """
    async def metrics_middleware(request: Request):
        """指标收集中间件"""
        # 记录请求指标
        request.ctx.metrics = {
            "start_time": time.time(),
            "method": request.method,
            "path": request.path,
        }

    return metrics_middleware


# 中间件工具函数
def get_request_info(request: Request) -> dict:
    """
    获取请求信息

    Args:
        request: Sanic请求对象

    Returns:
        dict: 请求信息
    """
    return {
        "request_id": getattr(request.ctx, "request_id", "unknown"),
        "method": request.method,
        "url": str(request.url),
        "path": request.path,
        "query_string": str(request.query_string),
        "client_ip": request.ip,
        "user_agent": request.headers.get("user-agent", ""),
        "content_type": request.headers.get("content-type", ""),
        "content_length": request.headers.get("content-length", "0"),
    }


def log_slow_request(request: Request, response: Response, threshold_seconds: float = 5.0) -> None:
    """
    记录慢请求

    Args:
        request: 请求对象
        response: 响应对象
        threshold_seconds: 慢请求阈值（秒）
    """
    duration = time.time() - getattr(request.ctx, "start_time", time.time())

    if duration > threshold_seconds:
        structured_logger.warning(
            "检测到慢请求",
            extra={
                "request_id": getattr(request.ctx, "request_id", "unknown"),
                "method": request.method,
                "url": str(request.url),
                "duration_seconds": round(duration, 2),
                "threshold_seconds": threshold_seconds,
                "status_code": response.status,
                "response_size": len(response.body) if hasattr(response, 'body') else 0,
            }
        )