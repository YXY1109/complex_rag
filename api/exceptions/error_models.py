"""
错误响应数据模型
定义标准化的错误响应格式和业务异常类
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """错误详情模型"""
    code: str
    message: str
    path: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """标准错误响应模型"""
    success: bool = False
    error: ErrorDetail


class ValidationErrorDetail(ErrorDetail):
    """验证错误详情模型"""
    validation_errors: Dict[str, Any]


class ValidationErrorResponse(ErrorResponse):
    """验证错误响应模型"""
    error: ValidationErrorDetail


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


class DatabaseError(BusinessException):
    """数据库异常"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "DATABASE_ERROR", details)


class ExternalServiceError(BusinessException):
    """外部服务异常"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "EXTERNAL_SERVICE_ERROR", details)


class AuthenticationError(BusinessException):
    """认证异常（预留，当前系统无需认证）"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "AUTHENTICATION_ERROR", details)


class AuthorizationError(BusinessException):
    """授权异常（预留，当前系统无需授权）"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "AUTHORIZATION_ERROR", details)