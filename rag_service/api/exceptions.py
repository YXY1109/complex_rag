"""
异常处理器

定义和注册全局异常处理器，统一错误响应格式。
"""

import logging
from typing import Union
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class APIException(Exception):
    """API异常基类。"""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        details: dict = None
    ):
        """
        初始化API异常。

        Args:
            message: 错误消息
            status_code: HTTP状态码
            error_code: 错误代码
            details: 错误详情
        """
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class ValidationError(APIException):
    """验证错误。"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="VALIDATION_ERROR",
            details=details
        )


class AuthenticationError(APIException):
    """认证错误。"""

    def __init__(self, message: str = "认证失败"):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(APIException):
    """授权错误。"""

    def __init__(self, message: str = "权限不足"):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="AUTHORIZATION_ERROR"
        )


class ResourceNotFoundError(APIException):
    """资源未找到错误。"""

    def __init__(self, message: str = "资源未找到"):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="RESOURCE_NOT_FOUND"
        )


class ResourceConflictError(APIException):
    """资源冲突错误。"""

    def __init__(self, message: str = "资源冲突"):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="RESOURCE_CONFLICT"
        )


class RateLimitError(APIException):
    """速率限制错误。"""

    def __init__(self, message: str = "请求过于频繁"):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_ERROR"
        )


class ServiceUnavailableError(APIException):
    """服务不可用错误。"""

    def __init__(self, message: str = "服务不可用"):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="SERVICE_UNAVAILABLE"
        )


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """API异常处理器。"""
    logger.error(
        f"API异常 - 路径: {request.url.path}, "
        f"错误代码: {exc.error_code}, "
        f"消息: {exc.message}, "
        f"详情: {exc.details}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


async def http_exception_handler(request: Request, exc: Union[HTTPException, StarletteHTTPException]) -> JSONResponse:
    """HTTP异常处理器。"""
    logger.error(
        f"HTTP异常 - 路径: {request.url.path}, "
        f"状态码: {exc.status_code}, "
        f"详情: {exc.detail}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_code": "HTTP_ERROR",
            "message": str(exc.detail),
            "status_code": exc.status_code,
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """请求验证异常处理器。"""
    logger.error(
        f"请求验证异常 - 路径: {request.url.path}, "
        f"错误: {exc.errors()}"
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "error_code": "VALIDATION_ERROR",
            "message": "请求参数验证失败",
            "details": {
                "validation_errors": exc.errors()
            },
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


async def pydantic_validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Pydantic验证异常处理器。"""
    logger.error(
        f"Pydantic验证异常 - 路径: {request.url.path}, "
        f"错误: {exc.errors()}"
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "error_code": "VALIDATION_ERROR",
            "message": "数据验证失败",
            "details": {
                "validation_errors": exc.errors()
            },
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """通用异常处理器。"""
    logger.error(
        f"未处理的异常 - 路径: {request.url.path}, "
        f"类型: {type(exc).__name__}, "
        f"消息: {str(exc)}",
        exc_info=True
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "error_code": "INTERNAL_ERROR",
            "message": "服务器内部错误",
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """
    设置异常处理器。

    Args:
        app: FastAPI应用实例
    """
    # API异常
    app.add_exception_handler(APIException, api_exception_handler)

    # HTTP异常
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # 验证异常
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, pydantic_validation_exception_handler)

    # 通用异常（最后注册）
    app.add_exception_handler(Exception, general_exception_handler)