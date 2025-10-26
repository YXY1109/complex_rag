"""
基础设施层的Loguru日志配置
为FastAPI和整个系统提供统一的日志记录器
"""

from config.loguru_config import get_logger

# 创建API层的日志记录器
logger = get_logger(__name__)

# 可以在这里添加额外的日志配置或工具函数
def log_api_request(method: str, path: str, **kwargs):
    """记录API请求日志的便捷函数"""
    logger.info(
        f"API请求: {method} {path}",
        extra={
            "type": "request",
            "method": method,
            "path": path,
            **kwargs
        }
    )

def log_api_response(method: str, path: str, status_code: int, **kwargs):
    """记录API响应日志的便捷函数"""
    logger.info(
        f"API响应: {method} {path} {status_code}",
        extra={
            "type": "request",
            "method": method,
            "path": path,
            "status_code": status_code,
            **kwargs
        }
    )

def log_performance(operation: str, duration_ms: float, **kwargs):
    """记录性能日志的便捷函数"""
    logger.info(
        f"性能监控: {operation} 耗时 {duration_ms:.2f}ms",
        extra={
            "type": "performance",
            "operation": operation,
            "duration_ms": duration_ms,
            **kwargs
        }
    )

def log_error(error: Exception, context: str = "", **kwargs):
    """记录错误日志的便捷函数"""
    logger.error(
        f"错误发生: {context} - {str(error)}",
        extra={
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **kwargs
        },
        exc_info=True
    )