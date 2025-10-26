"""
简化的API中间件模块
为统一FastAPI服务提供基础中间件功能
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from config.loguru_config import get_logger

logger = get_logger("api.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # 记录请求开始时间
        start_time = time.time()

        # 记录请求信息
        logger.info(
            f"API请求开始: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "user_agent": request.headers.get("user-agent", "unknown"),
                "client_ip": request.client.host if request.client else "unknown",
            }
        )

        # 处理请求
        try:
            response = await call_next(request)

            # 计算处理时间
            processing_time = time.time() - start_time

            # 记录响应信息
            logger.info(
                f"API请求完成: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "processing_time_seconds": round(processing_time, 3),
                }
            )

            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}"

            return response

        except Exception as e:
            # 计算处理时间
            processing_time = time.time() - start_time

            # 记录错误
            logger.error(
                f"API请求错误: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "processing_time_seconds": round(processing_time, 3),
                },
                exc_info=True
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """错误处理中间件"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")

            logger.error(
                f"未处理的API错误: {str(e)}",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "error": str(e),
                },
                exc_info=True
            )

            # 返回标准化错误响应
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Internal server error",
                        "type": "internal_server_error",
                        "code": "internal_error",
                        "request_id": request_id,
                    }
                }
            )


class PerformanceMiddleware(BaseHTTPMiddleware):
    """性能监控中间件"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")

        # 处理请求
        response = await call_next(request)

        # 计算处理时间
        processing_time = time.time() - start_time

        # 记录性能指标
        if processing_time > 1.0:  # 记录慢请求
            logger.warning(
                f"慢API请求检测: {request.method} {request.url.path} - {processing_time:.3f}s",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "processing_time_seconds": round(processing_time, 3),
                    "status_code": response.status_code,
                }
            )

        return response


# 简化的中间件管理器
class MiddlewareManager:
    """中间件管理器"""

    def __init__(self):
        self.logger = get_logger("api.middleware_manager")

    def setup_middleware(self, app):
        """设置所有中间件"""
        try:
            # 按顺序添加中间件
            app.add_middleware(ErrorHandlingMiddleware)
            app.add_middleware(PerformanceMiddleware)
            app.add_middleware(RequestLoggingMiddleware)

            self.logger.info("所有中间件设置完成")
        except Exception as e:
            self.logger.error(f"中间件设置失败: {e}", exc_info=True)
            raise


# 创建全局中间件管理器实例
middleware_manager = MiddlewareManager()

# 创建兼容性别名以支持现有代码
rate_limit_manager = middleware_manager
load_balancer_manager = middleware_manager
cache_manager = middleware_manager
async_optimization_manager = middleware_manager
monitoring_manager = middleware_manager


def setup_middleware(app):
    """设置中间件的便捷函数"""
    middleware_manager.setup_middleware(app)