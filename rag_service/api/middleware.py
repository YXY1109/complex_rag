"""
中间件配置

设置各种中间件，包括CORS、压缩、认证、日志等。
"""

import time
import uuid
import logging
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件。"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """记录请求和响应日志。"""
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # 记录请求开始
        start_time = time.time()
        method = request.method
        url = str(request.url)
        client_ip = request.client.host if request.client else "unknown"

        logger.info(
            f"请求开始 - ID: {request_id}, "
            f"方法: {method}, "
            f"URL: {url}, "
            f"客户端IP: {client_ip}"
        )

        # 执行请求
        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # 记录响应
            logger.info(
                f"请求完成 - ID: {request_id}, "
                f"状态码: {response.status_code}, "
                f"处理时间: {process_time:.3f}s"
            )

            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"请求异常 - ID: {request_id}, "
                f"错误: {str(e)}, "
                f"处理时间: {process_time:.3f}s"
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """速率限制中间件。"""

    def __init__(self, app, calls: int = 100, period: int = 60):
        """
        初始化速率限制中间件。

        Args:
            app: FastAPI应用
            calls: 允许的请求数
            period: 时间周期（秒）
        """
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """执行速率限制检查。"""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # 清理过期记录
        self._cleanup_expired_records(current_time)

        # 检查客户端请求记录
        if client_ip in self.clients:
            requests = self.clients[client_ip]
            # 统计最近时间窗口内的请求
            recent_requests = [req_time for req_time in requests if current_time - req_time < self.period]

            if len(recent_requests) >= self.calls:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"每{self.period}秒最多允许{self.calls}个请求",
                        "retry_after": self.period
                    }
                )

            self.clients[client_ip] = recent_requests + [current_time]
        else:
            self.clients[client_ip] = [current_time]

        return await call_next(request)

    def _cleanup_expired_records(self, current_time: float) -> None:
        """清理过期的请求记录。"""
        expired_clients = []
        for client_ip, requests in self.clients.items():
            # 保留最近的请求记录
            recent_requests = [req_time for req_time in requests if current_time - req_time < self.period * 2]
            if recent_requests:
                self.clients[client_ip] = recent_requests
            else:
                expired_clients.append(client_ip)

        # 删除过期的客户端记录
        for client_ip in expired_clients:
            del self.clients[client_ip]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """安全头中间件。"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """添加安全响应头。"""
        response = await call_next(request)

        # 添加安全头
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response


def setup_middleware(app: FastAPI) -> None:
    """
    设置所有中间件。

    Args:
        app: FastAPI应用实例
    """
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境中应该限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Gzip压缩中间件
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # 自定义中间件（顺序很重要）
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, calls=100, period=60)
    app.add_middleware(RequestLoggingMiddleware)