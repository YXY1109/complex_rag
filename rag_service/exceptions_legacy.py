"""
Sanic异常处理器配置

提供统一的异常处理和错误响应格式。
"""

import traceback
from typing import Any, Dict, Optional

from sanic import Sanic, Request, Response
from sanic.response import json
from sanic.exceptions import SanicException, NotFound, MethodNotAllowed, ServerError
from sanic.log import logger

from ..infrastructure.monitoring.loguru_logger import get_logger


# 获取结构化日志记录器
structured_logger = get_logger("rag_service.exceptions")


class RAGServiceException(Exception):
    """RAG服务基础异常类"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class ValidationError(RAGServiceException):
    """验证错误异常"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details
        )


class AuthenticationError(RAGServiceException):
    """认证错误异常"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401
        )


class AuthorizationError(RAGServiceException):
    """授权错误异常"""

    def __init__(self, message: str = "Access denied"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403
        )


class NotFoundError(RAGServiceException):
    """资源未找到异常"""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            status_code=404
        )


class RateLimitError(RAGServiceException):
    """限流异常"""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429
        )


class ModelError(RAGServiceException):
    """模型相关错误"""

    def __init__(self, message: str, model: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if model:
            details["model"] = model
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            status_code=500,
            details=details
        )


class ServiceUnavailableError(RAGServiceException):
    """服务不可用异常"""

    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            status_code=503
        )


def setup_exception_handlers(app: Sanic) -> None:
    """
    设置异常处理器

    Args:
        app: Sanic应用实例
    """

    @app.exception(RAGServiceException)
    async def handle_rag_service_exception(request: Request, exception: RAGServiceException):
        """处理RAG服务异常"""
        request_id = getattr(request.ctx, "request_id", "unknown")

        # 记录异常日志
        structured_logger.error(
            f"RAG服务异常: {exception.message}",
            extra={
                "request_id": request_id,
                "error_code": exception.error_code,
                "status_code": exception.status_code,
                "details": exception.details,
                "url": str(request.url),
                "method": request.method,
            }
        )

        return json({
            "error": {
                "message": exception.message,
                "error_code": exception.error_code,
                "status_code": exception.status_code,
                "details": exception.details,
                "request_id": request_id,
                "timestamp": _get_timestamp(),
            }
        }, status=exception.status_code)

    @app.exception(ValidationError)
    async def handle_validation_error(request: Request, exception: ValidationError):
        """处理验证错误"""
        request_id = getattr(request.ctx, "request_id", "unknown")

        structured_logger.warning(
            f"验证错误: {exception.message}",
            extra={
                "request_id": request_id,
                "error_details": exception.details,
                "url": str(request.url),
                "method": request.method,
            }
        )

        return json({
            "error": {
                "message": exception.message,
                "error_code": exception.error_code,
                "status_code": exception.status_code,
                "details": exception.details,
                "request_id": request_id,
                "timestamp": _get_timestamp(),
            }
        }, status=exception.status_code)

    @app.exception(AuthenticationError)
    async def handle_authentication_error(request: Request, exception: AuthenticationError):
        """处理认证错误"""
        request_id = getattr(request.ctx, "request_id", "unknown")

        structured_logger.warning(
            f"认证错误: {exception.message}",
            extra={
                "request_id": request_id,
                "url": str(request.url),
                "method": request.method,
                "client_ip": request.ip,
            }
        )

        return json({
            "error": {
                "message": exception.message,
                "error_code": exception.error_code,
                "status_code": exception.status_code,
                "request_id": request_id,
                "timestamp": _get_timestamp(),
            }
        }, status=exception.status_code)

    @app.exception(AuthorizationError)
    async def handle_authorization_error(request: Request, exception: AuthorizationError):
        """处理授权错误"""
        request_id = getattr(request.ctx, "request_id", "unknown")

        structured_logger.warning(
            f"授权错误: {exception.message}",
            extra={
                "request_id": request_id,
                "url": str(request.url),
                "method": request.method,
                "client_ip": request.ip,
            }
        )

        return json({
            "error": {
                "message": exception.message,
                "error_code": exception.error_code,
                "status_code": exception.status_code,
                "request_id": request_id,
                "timestamp": _get_timestamp(),
            }
        }, status=exception.status_code)

    @app.exception(NotFoundError)
    async def handle_not_found_error(request: Request, exception: NotFoundError):
        """处理未找到错误"""
        request_id = getattr(request.ctx, "request_id", "unknown")

        structured_logger.info(
            f"资源未找到: {exception.message}",
            extra={
                "request_id": request_id,
                "url": str(request.url),
                "method": request.method,
            }
        )

        return json({
            "error": {
                "message": exception.message,
                "error_code": exception.error_code,
                "status_code": exception.status_code,
                "request_id": request_id,
                "timestamp": _get_timestamp(),
            }
        }, status=exception.status_code)

    @app.exception(RateLimitError)
    async def handle_rate_limit_error(request: Request, exception: RateLimitError):
        """处理限流错误"""
        request_id = getattr(request.ctx, "request_id", "unknown")

        structured_logger.warning(
            f"限流触发: {exception.message}",
            extra={
                "request_id": request_id,
                "url": str(request.url),
                "method": request.method,
                "client_ip": request.ip,
            }
        )

        return json({
            "error": {
                "message": exception.message,
                "error_code": exception.error_code,
                "status_code": exception.status_code,
                "request_id": request_id,
                "timestamp": _get_timestamp(),
            }
        }, status=exception.status_code, headers={
            "Retry-After": "60"  # 建议60秒后重试
        })

    @app.exception(ModelError)
    async def handle_model_error(request: Request, exception: ModelError):
        """处理模型错误"""
        request_id = getattr(request.ctx, "request_id", "unknown")

        structured_logger.error(
            f"模型错误: {exception.message}",
            extra={
                "request_id": request_id,
                "error_details": exception.details,
                "url": str(request.url),
                "method": request.method,
            }
        )

        return json({
            "error": {
                "message": exception.message,
                "error_code": exception.error_code,
                "status_code": exception.status_code,
                "details": exception.details,
                "request_id": request_id,
                "timestamp": _get_timestamp(),
            }
        }, status=exception.status_code)

    @app.exception(ServiceUnavailableError)
    async def handle_service_unavailable_error(request: Request, exception: ServiceUnavailableError):
        """处理服务不可用错误"""
        request_id = getattr(request.ctx, "request_id", "unknown")

        structured_logger.error(
            f"服务不可用: {exception.message}",
            extra={
                "request_id": request_id,
                "url": str(request.url),
                "method": request.method,
            }
        )

        return json({
            "error": {
                "message": exception.message,
                "error_code": exception.error_code,
                "status_code": exception.status_code,
                "request_id": request_id,
                "timestamp": _get_timestamp(),
            }
        }, status=exception.status_code)

    @app.exception(NotFound)
    async def handle_sanic_not_found(request: Request, exception: NotFound):
        """处理Sanic 404异常"""
        request_id = getattr(request.ctx, "request_id", "unknown")

        structured_logger.info(
            f"路径未找到: {request.url}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.path,
            }
        )

        return json({
            "error": {
                "message": f"The requested resource '{request.path}' was not found",
                "error_code": "ENDPOINT_NOT_FOUND",
                "status_code": 404,
                "request_id": request_id,
                "timestamp": _get_timestamp(),
            }
        }, status=404)

    @app.exception(MethodNotAllowed)
    async def handle_sanic_method_not_allowed(request: Request, exception: MethodNotAllowed):
        """处理Sanic方法不允许异常"""
        request_id = getattr(request.ctx, "request_id", "unknown")

        structured_logger.warning(
            f"方法不允许: {request.method} {request.url}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.path,
            }
        )

        return json({
            "error": {
                "message": f"The method '{request.method}' is not allowed for the requested resource",
                "error_code": "METHOD_NOT_ALLOWED",
                "status_code": 405,
                "request_id": request_id,
                "timestamp": _get_timestamp(),
            }
        }, status=405)

    @app.exception(ServerError)
    async def handle_sanic_server_error(request: Request, exception: ServerError):
        """处理Sanic服务器错误异常"""
        request_id = getattr(request.ctx, "request_id", "unknown")

        structured_logger.error(
            f"服务器内部错误: {exception}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "error_details": str(exception),
            }
        )

        return json({
            "error": {
                "message": "Internal server error occurred",
                "error_code": "INTERNAL_SERVER_ERROR",
                "status_code": 500,
                "request_id": request_id,
                "timestamp": _get_timestamp(),
            }
        }, status=500)

    @app.exception(Exception)
    async def handle_generic_exception(request: Request, exception: Exception):
        """处理通用异常"""
        request_id = getattr(request.ctx, "request_id", "unknown")

        # 记录完整的异常信息
        structured_logger.error(
            f"未处理的异常: {type(exception).__name__}: {exception}",
            extra={
                "request_id": request_id,
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "method": request.method,
                "url": str(request.url),
                "traceback": traceback.format_exc(),
            }
        )

        # 在开发模式下返回详细的错误信息
        if request.app.debug:
            return json({
                "error": {
                    "message": str(exception),
                    "error_code": "UNEXPECTED_ERROR",
                    "status_code": 500,
                    "exception_type": type(exception).__name__,
                    "request_id": request_id,
                    "timestamp": _get_timestamp(),
                    "traceback": traceback.format_exc(),
                }
            }, status=500)
        else:
            # 生产模式下返回通用错误信息
            return json({
                "error": {
                    "message": "An unexpected error occurred",
                    "error_code": "UNEXPECTED_ERROR",
                    "status_code": 500,
                    "request_id": request_id,
                    "timestamp": _get_timestamp(),
                }
            }, status=500)

    structured_logger.info("异常处理器设置完成", extra={"service": "rag_service"})


def _get_timestamp() -> str:
    """
    获取当前时间戳

    Returns:
        str: ISO格式的时间戳
    """
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"


# 错误响应工具函数
def create_error_response(
    message: str,
    error_code: str = "UNKNOWN_ERROR",
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建标准错误响应

    Args:
        message: 错误消息
        error_code: 错误代码
        status_code: HTTP状态码
        details: 错误详情
        request_id: 请求ID

    Returns:
        Dict[str, Any]: 错误响应字典
    """
    return {
        "error": {
            "message": message,
            "error_code": error_code,
            "status_code": status_code,
            "details": details or {},
            "request_id": request_id,
            "timestamp": _get_timestamp(),
        }
    }


def create_validation_error_response(
    message: str,
    field_errors: Optional[Dict[str, str]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建验证错误响应

    Args:
        message: 错误消息
        field_errors: 字段级别的错误信息
        request_id: 请求ID

    Returns:
        Dict[str, Any]: 验证错误响应字典
    """
    details = {}
    if field_errors:
        details["field_errors"] = field_errors

    return create_error_response(
        message=message,
        error_code="VALIDATION_ERROR",
        status_code=400,
        details=details,
        request_id=request_id
    )


def create_model_error_response(
    message: str,
    model: Optional[str] = None,
    error_type: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    创建模型错误响应

    Args:
        message: 错误消息
        model: 模型名称
        error_type: 错误类型
        request_id: 请求ID

    Returns:
        Dict[str, Any]: 模型错误响应字典
    """
    details = {}
    if model:
        details["model"] = model
    if error_type:
        details["error_type"] = error_type

    return create_error_response(
        message=message,
        error_code="MODEL_ERROR",
        status_code=500,
        details=details,
        request_id=request_id
    )