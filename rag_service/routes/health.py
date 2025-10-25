"""
健康检查路由

提供服务健康状态检查和基本信息。
"""

import time
import psutil
from typing import Dict, Any

from sanic import Blueprint, Request, Response
from sanic.response import json

from ..infrastructure.monitoring.loguru_logger import get_logger


# 创建健康检查蓝图
health_router = Blueprint("health", url_prefix="/health")
structured_logger = get_logger("rag_service.health")


@health_router.get("/")
async def health_check(request: Request) -> Response:
    """
    基本健康检查

    Returns:
        Response: 健康状态响应
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        # 获取服务实例
        service = request.app.ctx.service

        # 检查各个服务的健康状态
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "request_id": request_id,
            "service": "rag-service",
            "version": "1.0.0",
            "services": {
                "llm": await _check_service_health(service.get_llm_service()),
                "embedding": await _check_service_health(service.get_embedding_service()),
                "rerank": await _check_service_health(service.get_rerank_service()),
                "memory": await _check_service_health(service.get_memory_service()),
            }
        }

        # 判断整体健康状态
        service_statuses = [status["status"] for status in health_status["services"].values() if status is not None]
        if any(status == "unhealthy" for status in service_statuses):
            health_status["status"] = "degraded"
        elif any(status == "unknown" for status in service_statuses):
            health_status["status"] = "partial"

        structured_logger.info(
            "健康检查完成",
            extra={
                "request_id": request_id,
                "overall_status": health_status["status"],
                "services": health_status["services"],
            }
        )

        status_code = 200 if health_status["status"] == "healthy" else 503
        return json(health_status, status=status_code)

    except Exception as e:
        structured_logger.error(
            f"健康检查失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )

        return json({
            "status": "unhealthy",
            "timestamp": time.time(),
            "request_id": request_id,
            "service": "rag-service",
            "error": str(e),
        }, status=503)


@health_router.get("/detailed")
async def detailed_health_check(request: Request) -> Response:
    """
    详细健康检查

    Returns:
        Response: 详细健康状态响应
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        service = request.app.ctx.service

        # 获取系统资源信息
        system_info = _get_system_info()

        # 检查各个服务
        services_health = {}
        if service.get_llm_service():
            services_health["llm"] = await service.get_llm_service().health_check()
        if service.get_embedding_service():
            services_health["embedding"] = await service.get_embedding_service().health_check()
        if service.get_rerank_service():
            services_health["rerank"] = await service.get_rerank_service().health_check()
        if service.get_memory_service():
            services_health["memory"] = await service.get_memory_service().health_check()

        detailed_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "request_id": request_id,
            "service": "rag-service",
            "version": "1.0.0",
            "system": system_info,
            "services": services_health,
            "configuration": {
                "debug": request.app.debug,
                "environment": "development" if request.app.debug else "production",
            }
        }

        # 判断整体健康状态
        service_statuses = [status.get("status", "unknown") for status in services_health.values()]
        if any(status == "unhealthy" for status in service_statuses):
            detailed_status["status"] = "degraded"
        elif any(status in ["unknown", "error"] for status in service_statuses):
            detailed_status["status"] = "partial"

        structured_logger.info(
            "详细健康检查完成",
            extra={
                "request_id": request_id,
                "overall_status": detailed_status["status"],
                "system_load": system_info.get("cpu_percent"),
                "memory_usage": system_info.get("memory_percent"),
            }
        )

        status_code = 200 if detailed_status["status"] == "healthy" else 503
        return json(detailed_status, status=status_code)

    except Exception as e:
        structured_logger.error(
            f"详细健康检查失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )

        return json({
            "status": "unhealthy",
            "timestamp": time.time(),
            "request_id": request_id,
            "service": "rag-service",
            "error": str(e),
        }, status=503)


@health_router.get("/ping")
async def ping(request: Request) -> Response:
    """
    简单的ping检查

    Returns:
        Response: ping响应
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    return json({
        "message": "pong",
        "timestamp": time.time(),
        "request_id": request_id,
        "service": "rag-service",
    })


@health_router.get("/ready")
async def readiness_check(request: Request) -> Response:
    """
    就绪检查（用于Kubernetes等容器编排）

    Returns:
        Response: 就绪状态响应
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    try:
        service = request.app.ctx.service

        # 检查关键服务是否就绪
        ready_services = []
        if service.get_llm_service():
            ready_services.append("llm")
        if service.get_embedding_service():
            ready_services.append("embedding")

        is_ready = len(ready_services) > 0  # 至少有一个服务就绪

        readiness_status = {
            "ready": is_ready,
            "timestamp": time.time(),
            "request_id": request_id,
            "service": "rag-service",
            "ready_services": ready_services,
        }

        structured_logger.info(
            "就绪检查完成",
            extra={
                "request_id": request_id,
                "ready": is_ready,
                "ready_services": ready_services,
            }
        )

        status_code = 200 if is_ready else 503
        return json(readiness_status, status=status_code)

    except Exception as e:
        structured_logger.error(
            f"就绪检查失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )

        return json({
            "ready": False,
            "timestamp": time.time(),
            "request_id": request_id,
            "service": "rag-service",
            "error": str(e),
        }, status=503)


@health_router.get("/live")
async def liveness_check(request: Request) -> Response:
    """
    存活检查（用于Kubernetes等容器编排）

    Returns:
        Response: 存活状态响应
    """
    request_id = getattr(request.ctx, "request_id", "unknown")

    # 简单的存活检查 - 如果能响应这个请求就说明服务是存活的
    liveness_status = {
        "alive": True,
        "timestamp": time.time(),
        "request_id": request_id,
        "service": "rag-service",
    }

    structured_logger.debug(
        "存活检查完成",
        extra={
            "request_id": request_id,
        }
    )

    return json(liveness_status)


async def _check_service_health(service) -> Dict[str, Any]:
    """
    检查单个服务的健康状态

    Args:
        service: 服务实例

    Returns:
        Dict[str, Any]: 服务健康状态
    """
    if service is None:
        return {"status": "not_configured", "message": "Service not configured"}

    try:
        health_result = await service.health_check()
        return {
            "status": health_result.get("status", "unknown"),
            "provider": health_result.get("provider", "unknown"),
            "model": health_result.get("model", "unknown"),
            "message": health_result.get("message", ""),
            "response_time_ms": health_result.get("response_time_ms"),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def _get_system_info() -> Dict[str, Any]:
    """
    获取系统信息

    Returns:
        Dict[str, Any]: 系统信息
    """
    try:
        # CPU信息
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        # 内存信息
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # 磁盘信息
        disk = psutil.disk_usage('/')

        # 网络信息
        network = psutil.net_io_counters()

        return {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
                "frequency": {
                    "current": cpu_freq.current if cpu_freq else None,
                    "min": cpu_freq.min if cpu_freq else None,
                    "max": cpu_freq.max if cpu_freq else None,
                },
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free,
            },
            "swap": {
                "total": swap.total,
                "used": swap.used,
                "free": swap.free,
                "percent": swap.percent,
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100 if disk.total > 0 else 0,
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv,
            },
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
        }

    except Exception as e:
        structured_logger.warning(f"获取系统信息失败: {e}")
        return {
            "error": str(e),
            "message": "Failed to retrieve system information",
        }


structured_logger.info("健康检查路由加载完成")