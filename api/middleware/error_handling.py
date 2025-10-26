"""
错误处理中间件
提供统一的异常处理机制和标准的错误响应格式
"""
from typing import Callable, Dict, Any

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from infrastructure.monitoring.loguru_logger import logger


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    错误处理中间件
    捕获所有异常并返回标准化的错误响应
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理HTTP请求并捕获异常

        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或路由处理器

        Returns:
            Response: HTTP响应对象
        """
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            return await self._handle_exception(request, e)

    async def _handle_exception(self, request: Request, exception: Exception) -> JSONResponse:
        """
        处理异常并返回标准化错误响应

        Args:
            request: HTTP请求对象
            exception: 异常对象

        Returns:
            JSONResponse: 标准化错误响应
        """
        # 记录错误日志
        logger.error(
            "API请求异常",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "error_type": type(exception).__name__,
                "error_message": str(exception),
                "client_ip": request.client.host if request.client else "unknown"
            },
            exc_info=True
        )

        # 处理HTTP异常
        if isinstance(exception, (HTTPException, StarletteHTTPException)):
            return self._create_error_response(
                status_code=exception.status_code,
                error_code=type(exception).__name__,
                message=str(exception.detail) if hasattr(exception, 'detail') else str(exception),
                path=request.url.path
            )

        # 处理业务异常
        elif isinstance(exception, BusinessException):
            return self._create_error_response(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code=exception.error_code,
                message=str(exception),
                path=request.url.path,
                details=exception.details
            )

        # 处理验证异常
        elif isinstance(exception, ValueError):
            return self._create_error_response(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code="VALIDATION_ERROR",
                message="请求参数验证失败",
                path=request.url.path,
                details={"validation_error": str(exception)}
            )

        # 处理其他未知异常
        else:
            return self._create_error_response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_code="INTERNAL_SERVER_ERROR",
                message="服务器内部错误",
                path=request.url.path
            )

    def _create_error_response(
        self,
        status_code: int,
        error_code: str,
        message: str,
        path: str,
        details: Dict[str, Any] = None
    ) -> JSONResponse:
        """
        创建标准化错误响应

        Args:
            status_code: HTTP状态码
            error_code: 错误代码
            message: 错误消息
            path: 请求路径
            details: 错误详情

        Returns:
            JSONResponse: 标准化错误响应
        """
        error_response = {
            "success": False,
            "error": {
                "code": error_code,
                "message": message,
                "path": path,
                "timestamp": self._get_current_timestamp()
            }
        }

        if details:
            error_response["error"]["details"] = details

        return JSONResponse(
            status_code=status_code,
            content=error_response
        )

    def _get_current_timestamp(self) -> str:
        """
        获取当前时间戳

        Returns:
            str: ISO格式的时间戳
        """
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


class BusinessException(Exception):
    """
    业务异常基类
    用于处理业务逻辑中的异常情况
    """

    def __init__(self, message: str, error_code: str = "BUSINESS_ERROR", details: Dict[str, Any] = None):
        """
        初始化业务异常

        Args:
            message: 异常消息
            error_code: 错误代码
            details: 异常详情
        """
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ValidationError(BusinessException):
    """验证异常"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "VALIDATION_ERROR", details)


class NotFoundError(BusinessException):
    """资源未找到异常"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "NOT_FOUND", details)


class ConflictError(BusinessException):
    """资源冲突异常"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "CONFLICT", details)


class ServiceUnavailableError(BusinessException):
    """服务不可用异常"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "SERVICE_UNAVAILABLE", details)