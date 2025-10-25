"""
依赖注入

定义FastAPI依赖项，提供认证、RAG服务实例等功能。
"""

from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)

# HTTP Bearer认证
security = HTTPBearer(auto_error=False)


async def get_rag_service(request: Request) -> "UnifiedRAGService":
    """
    获取RAG服务实例。

    Args:
        request: FastAPI请求对象

    Returns:
        UnifiedRAGService: RAG服务实例

    Raises:
        HTTPException: 当服务未初始化时
    """
    if not hasattr(request.app.state, 'rag_service'):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG服务未初始化"
        )

    return request.app.state.rag_service


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    获取当前用户信息。

    Args:
        credentials: HTTP认证凭据

    Returns:
        Dict[str, Any]: 用户信息

    Raises:
        HTTPException: 当认证失败时
    """
    # 简化实现，实际项目中应该验证JWT token
    if not credentials:
        # 如果没有提供token，返回默认用户
        return {
            "user_id": "anonymous",
            "tenant_id": "default",
            "roles": ["anonymous"]
        }

    try:
        # 这里应该验证JWT token
        # 暂时返回模拟用户信息
        return {
            "user_id": "demo_user",
            "tenant_id": "demo_tenant",
            "roles": ["user"],
            "token": credentials.credentials
        }
    except Exception as e:
        logger.error(f"用户认证失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证失败",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    获取管理员用户信息。

    Args:
        current_user: 当前用户

    Returns:
        Dict[str, Any]: 管理员用户信息

    Raises:
        HTTPException: 当用户不是管理员时
    """
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )

    return current_user


async def get_tenant_context(current_user: Dict[str, Any] = Depends(get_current_user)) -> str:
    """
    获取租户上下文。

    Args:
        current_user: 当前用户

    Returns:
        str: 租户ID
    """
    return current_user.get("tenant_id", "default")


async def validate_request_size(request: Request, max_size: int = 10 * 1024 * 1024) -> None:
    """
    验证请求大小。

    Args:
        request: FastAPI请求对象
        max_size: 最大允许大小（字节）

    Raises:
        HTTPException: 当请求过大时
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"请求过大，最大允许{max_size}字节"
        )


async def get_request_context(request: Request) -> Dict[str, Any]:
    """
    获取请求上下文信息。

    Args:
        request: FastAPI请求对象

    Returns:
        Dict[str, Any]: 请求上下文
    """
    return {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
        "method": request.method,
        "url": str(request.url)
    }