"""
健康检查路由

提供系统健康检查、状态监控等功能的API接口。
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends
import logging
from datetime import datetime

from ..dependencies import get_rag_service, get_request_context
from ...services.unified_rag_service import UnifiedRAGService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    基础健康检查。

    Args:
        rag_service: RAG服务实例
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 健康状态
    """
    try:
        # 执行健康检查
        health_status = await rag_service.health_check()

        # 基础健康信息
        health_info = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "service": "complex-rag-api"
        }

        # 合并RAG服务健康状态
        if health_status:
            health_info.update({
                "rag_service": health_status
            })

        return health_info

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "service": "complex-rag-api",
            "error": str(e)
        }


@router.get("/health/ready", response_model=Dict[str, Any])
async def readiness_check(
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    就绪检查（Kubernetes就绪探针）。

    Args:
        rag_service: RAG服务实例
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 就绪状态
    """
    try:
        # 检查RAG服务是否已初始化
        service_status = await rag_service.get_service_status()

        if not service_status.get("initialized", False):
            return {
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "RAG服务未初始化"
            }

        # 检查关键组件状态
        components = service_status.get("components", {})
        critical_components = ["vector_store", "embedding_service", "retrieval_engine"]

        failed_components = []
        for component in critical_components:
            component_status = components.get(component, "unknown")
            if component_status in ["error", "not_initialized", "unavailable"]:
                failed_components.append(component)

        if failed_components:
            return {
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": f"关键组件未就绪: {', '.join(failed_components)}",
                "failed_components": failed_components
            }

        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "components": components
        }

    except Exception as e:
        logger.error(f"就绪检查失败: {e}")
        return {
            "status": "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "reason": f"就绪检查异常: {str(e)}"
        }


@router.get("/health/live", response_model=Dict[str, Any])
async def liveness_check(
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    存活检查（Kubernetes存活探针）。

    Args:
        rag_service: RAG服务实例
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 存活状态
    """
    try:
        # 简单的存活检查，主要确认服务进程是否正常运行
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": "running"  # 这里应该计算实际运行时间
        }

    except Exception as e:
        logger.error(f"存活检查失败: {e}")
        return {
            "status": "dead",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/health/detailed", response_model=Dict[str, Any])
async def detailed_health_check(
    rag_service: UnifiedRAGService = Depends(get_rag_service),
    request_context: Dict[str, Any] = Depends(get_request_context)
):
    """
    详细健康检查。

    Args:
        rag_service: RAG服务实例
        request_context: 请求上下文

    Returns:
        Dict[str, Any]: 详细健康状态
    """
    try:
        # 获取服务状态
        service_status = await rag_service.get_service_status()
        health_status = await rag_service.health_check()

        # 获取统计信息
        statistics = await rag_service.get_statistics()

        # 系统资源信息（简化版本）
        system_info = {
            "memory_usage": "unknown",  # 这里需要实现实际的内存监控
            "cpu_usage": "unknown",     # 这里需要实现实际的CPU监控
            "disk_usage": "unknown",    # 这里需要实现实际的磁盘监控
            "network_status": "connected"
        }

        detailed_status = {
            "status": health_status.get("status", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "service": "complex-rag-api",
            "request_id": request_context.get("request_id"),
            "components": {
                "service_status": service_status,
                "health_status": health_status,
                "statistics": statistics,
                "system_info": system_info
            }
        }

        return detailed_status

    except Exception as e:
        logger.error(f"详细健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "service": "complex-rag-api",
            "error": str(e),
            "request_id": request_context.get("request_id")
        }


@router.get("/version", response_model=Dict[str, Any])
async def get_version():
    """
    获取版本信息。

    Returns:
        Dict[str, Any]: 版本信息
    """
    return {
        "version": "1.0.0",
        "build_time": "2024-01-01T00:00:00Z",
        "git_commit": "unknown",
        "python_version": "3.9+",
        "fastapi_version": "0.104+",
        "service": "complex-rag-api"
    }


@router.get("/info", response_model=Dict[str, Any])
async def get_service_info():
    """
    获取服务信息。

    Returns:
        Dict[str, Any]: 服务信息
    """
    return {
        "service_name": "复杂RAG服务API",
        "description": "基于多模态文档解析和智能检索的RAG服务系统",
        "version": "1.0.0",
        "documentation": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "health": "/health",
        "admin": {
            "status": "/admin/system/status",
            "statistics": "/admin/system/statistics"
        },
        "features": [
            "文档解析与处理",
            "向量存储与检索",
            "智能问答",
            "多轮对话",
            "知识库管理",
            "批量处理",
            "流式响应"
        ]
    }