"""
简化的API异常处理模块
为统一FastAPI服务提供基础异常处理功能
"""

from typing import Callable
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.responses import Response

from config.loguru_config import get_logger

logger = get_logger("api.exceptions")


class APIException(Exception):
    """API异常基类"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "api_error",
        error_code: str = "unknown_error",
        details: dict = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(APIException):
    """验证错误"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(
            message=message,
            status_code=400,
            error_type="validation_error",
            error_code="invalid_request",
            details=details or {}
        )


class NotFoundError(APIException):
    """资源未找到错误"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(
            message=message,
            status_code=404,
            error_type="not_found_error",
            error_code="not_found",
            details=details or {}
        )


class ModelError(APIException):
    """模型错误"""

    def __init__(self, message: str, error_code: str = "model_error", details: dict = None):
        super().__init__(
            message=message,
            status_code=500,
            error_type="model_error",
            error_code=error_code,
            details=details or {}
        )


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """API异常处理器"""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"API异常: {exc.message}",
        extra={
            "request_id": request_id,
            "error_type": exc.error_type,
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        }
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "code": exc.error_code,
                "request_id": request_id,
                **exc.details
            }
        }
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """HTTP异常处理器"""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        f"HTTP异常: {exc.detail}",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
        }
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "http_error",
                "code": f"http_{exc.status_code}",
                "request_id": request_id,
            }
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """通用异常处理器"""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"未处理的异常: {str(exc)}",
        extra={
            "request_id": request_id,
            "error_type": type(exc).__name__,
            "path": request.url.path,
            "method": request.method,
        },
        exc_info=True
    )

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


def setup_exception_handlers(app: FastAPI):
    """设置异常处理器"""

    # API异常处理器
    app.add_exception_handler(APIException, api_exception_handler)

    # HTTP异常处理器
    app.add_exception_handler(HTTPException, http_exception_handler)

    # 通用异常处理器
    app.add_exception_handler(Exception, general_exception_handler)

    logger.info("异常处理器设置完成")


# 导出常用异常类
__all__ = [
    "APIException",
    "ValidationError",
    "NotFoundError",
    "ModelError",
    "setup_exception_handler"
]