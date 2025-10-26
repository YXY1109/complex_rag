"""
请求日志中间件
记录所有API请求的详细信息，包括请求路径、方法、状态码、处理时间等
"""
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from infrastructure.monitoring.loguru_logger import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件
    记录所有HTTP请求的详细信息，用于调试和监控
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理HTTP请求并记录日志

        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或路由处理器

        Returns:
            Response: HTTP响应对象
        """
        start_time = time.time()

        # 记录请求开始
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        logger.info(
            "请求开始",
            extra={
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": client_ip,
                "user_agent": user_agent,
            }
        )

        try:
            # 执行下一个中间件或路由处理器
            response = await call_next(request)

            # 计算处理时间
            process_time = time.time() - start_time

            # 记录请求完成
            logger.info(
                "请求完成",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": round(process_time, 4),
                    "client_ip": client_ip,
                }
            )

            # 在响应头中添加处理时间
            response.headers["X-Process-Time"] = str(round(process_time, 4))

            return response

        except Exception as e:
            # 计算处理时间
            process_time = time.time() - start_time

            # 记录请求异常
            logger.error(
                "请求异常",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "process_time": round(process_time, 4),
                    "client_ip": client_ip,
                },
                exc_info=True
            )

            # 重新抛出异常，让错误处理中间件处理
            raise