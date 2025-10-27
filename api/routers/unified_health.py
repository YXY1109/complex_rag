"""
统一健康检查路由
从Sanic RAG服务迁移的健康检查功能到FastAPI
"""

import time
import psutil
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from config.loguru_config import get_logger

# 创建路由器
router = APIRouter()
structured_logger = get_logger("api.unified_health")


# 响应模型定义
class ServiceHealth(BaseModel):
    status: str = Field(..., description="服务状态: healthy/unhealthy")
    response_time_ms: Optional[float] = Field(None, description="响应时间（毫秒）")
    error: Optional[str] = Field(None, description="错误信息")
    last_check: str = Field(..., description="最后检查时间")


class SystemHealth(BaseModel):
    cpu_usage_percent: float = Field(..., description="CPU使用率")
    memory_usage_percent: float = Field(..., description="内存使用率")
    memory_available_gb: float = Field(..., description="可用内存（GB）")
    disk_usage_percent: float = Field(..., description="磁盘使用率")
    disk_available_gb: float = Field(..., description="可用磁盘空间（GB）")
    uptime_seconds: float = Field(..., description="系统运行时间（秒）")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="总体状态: healthy/unhealthy")
    timestamp: float = Field(..., description="检查时间戳")
    service: str = Field(..., description="服务名称")
    version: str = Field(..., description="服务版本")
    uptime_seconds: float = Field(..., description="服务运行时间")
    services: Dict[str, ServiceHealth] = Field(..., description="各服务健康状态")
    system: Optional[SystemHealth] = Field(None, description="系统信息")


class SimpleHealthResponse(BaseModel):
    status: str = Field(..., description="健康状态")
    message: str = Field(..., description="状态消息")
    timestamp: float = Field(..., description="时间戳")


async def check_service_health(service) -> ServiceHealth:
    """
    检查单个服务的健康状态

    Args:
        service: 服务实例

    Returns:
        ServiceHealth: 服务健康状态
    """
    start_time = time.time()

    try:
        if service is None:
            return ServiceHealth(
                status="unhealthy",
                error="服务实例为空",
                last_check=time.strftime("%Y-%m-%d %H:%M:%S")
            )

        # 尝试调用服务的健康检查方法
        if hasattr(service, 'health_check'):
            health_result = await service.health_check()
            response_time = (time.time() - start_time) * 1000

            if isinstance(health_result, dict):
                return ServiceHealth(
                    status=health_result.get('status', 'healthy'),
                    response_time_ms=response_time,
                    error=health_result.get('error'),
                    last_check=time.strftime("%Y-%m-%d %H:%M:%S")
                )
            else:
                return ServiceHealth(
                    status="healthy" if health_result else "unhealthy",
                    response_time_ms=response_time,
                    last_check=time.strftime("%Y-%m-%d %H:%M:%S")
                )
        else:
            # 如果没有health_check方法，检查服务是否可用
            response_time = (time.time() - start_time) * 1000
            return ServiceHealth(
                status="healthy",
                response_time_ms=response_time,
                last_check=time.strftime("%Y-%m-%d %H:%M:%S")
            )

    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return ServiceHealth(
            status="unhealthy",
            response_time_ms=response_time,
            error=str(e),
            last_check=time.strftime("%Y-%m-%d %H:%M:%S")
        )


def get_system_info() -> SystemHealth:
    """
    获取系统信息

    Returns:
        SystemHealth: 系统信息
    """
    try:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)

        # 内存信息
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)

        # 磁盘信息
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_available_gb = disk.free / (1024**3)

        # 系统运行时间
        uptime = time.time() - psutil.boot_time()

        return SystemHealth(
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory_percent,
            memory_available_gb=round(memory_available_gb, 2),
            disk_usage_percent=round(disk_percent, 2),
            disk_available_gb=round(disk_available_gb, 2),
            uptime_seconds=round(uptime, 2)
        )

    except Exception as e:
        structured_logger.error(f"获取系统信息失败: {str(e)}")
        # 返回默认值
        return SystemHealth(
            cpu_usage_percent=0.0,
            memory_usage_percent=0.0,
            memory_available_gb=0.0,
            disk_usage_percent=0.0,
            disk_available_gb=0.0,
            uptime_seconds=0.0
        )


@router.get("/", response_model=HealthCheckResponse, summary="详细健康检查")
async def detailed_health_check(request: Request, include_system: bool = False):
    """
    详细健康检查

    Args:
        request: FastAPI请求对象
        include_system: 是否包含系统信息

    Returns:
        HealthCheckResponse: 详细健康状态
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', 'unknown')

    try:
        # 获取应用启动时间
        app_start_time = getattr(request.app.state, 'start_time', start_time)
        uptime = start_time - app_start_time

        # 检查各个服务的健康状态
        health_status = HealthCheckResponse(
            status="healthy",
            timestamp=start_time,
            service="unified-rag-service",
            version="2.0.0",
            uptime_seconds=round(uptime, 2),
            services={}
        )

        # 获取服务实例
        llm_service = getattr(request.app.state, 'llm_service', None)
        embedding_service = getattr(request.app.state, 'embedding_service', None)
        rerank_service = getattr(request.app.state, 'rerank_service', None)
        memory_service = getattr(request.app.state, 'memory_service', None)

        # 检查各个服务
        health_status.services["llm"] = await check_service_health(llm_service)
        health_status.services["embedding"] = await check_service_health(embedding_service)
        health_status.services["rerank"] = await check_service_health(rerank_service)
        health_status.services["memory"] = await check_service_health(memory_service)

        # 判断总体健康状态
        unhealthy_services = [
            name for name, service in health_status.services.items()
            if service.status == "unhealthy"
        ]

        if unhealthy_services:
            health_status.status = "unhealthy"
            structured_logger.warning(
                "服务健康检查发现问题",
                extra={
                    "request_id": request_id,
                    "unhealthy_services": unhealthy_services
                }
            )

        # 添加系统信息（如果需要）
        if include_system:
            health_status.system = get_system_info()

        structured_logger.info(
            "健康检查完成",
            extra={
                "request_id": request_id,
                "overall_status": health_status.status,
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            }
        )

        return health_status

    except Exception as e:
        structured_logger.error(f"健康检查失败: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": start_time,
                "service": "unified-rag-service",
                "version": "2.0.0",
                "error": str(e)
            }
        )


@router.get("/simple", response_model=SimpleHealthResponse, summary="简单健康检查")
async def simple_health_check(request: Request):
    """
    简单健康检查

    Args:
        request: FastAPI请求对象

    Returns:
        SimpleHealthResponse: 简单健康状态
    """
    timestamp = time.time()

    try:
        # 快速检查关键服务是否可用
        llm_service = getattr(request.app.state, 'llm_service', None)
        embedding_service = getattr(request.app.state, 'embedding_service', None)

        if llm_service and embedding_service:
            status = "healthy"
            message = "所有关键服务正常运行"
        else:
            status = "unhealthy"
            message = "部分关键服务不可用"

        return SimpleHealthResponse(
            status=status,
            message=message,
            timestamp=timestamp
        )

    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": f"健康检查失败: {str(e)}",
                "timestamp": timestamp
            }
        )


@router.get("/ping", response_model=SimpleHealthResponse, summary="Ping检查")
async def ping_check(request: Request):
    """
    Ping检查 - 最简单的健康检查

    Args:
        request: FastAPI请求对象

    Returns:
        SimpleHealthResponse: Ping响应
    """
    return SimpleHealthResponse(
        status="ok",
        message="pong",
        timestamp=time.time()
    )


@router.get("/ready", summary="就绪检查")
async def readiness_check(request: Request):
    """
    Kubernetes就绪检查

    Args:
        request: FastAPI请求对象

    Returns:
        Dict: 就绪状态
    """
    try:
        # 检查关键服务是否就绪
        llm_service = getattr(request.app.state, 'llm_service', None)
        embedding_service = getattr(request.app.state, 'embedding_service', None)
        rerank_service = getattr(request.app.state, 'rerank_service', None)

        if all([llm_service, embedding_service, rerank_service]):
            return {"status": "ready"}
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "not ready", "message": "关键服务未就绪"}
            )

    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "message": str(e)}
        )


@router.get("/live", summary="存活检查")
async def liveness_check(request: Request):
    """
    Kubernetes存活检查

    Args:
        request: FastAPI请求对象

    Returns:
        Dict: 存活状态
    """
    return {"status": "alive", "timestamp": time.time()}


@router.get("/system", response_model=SystemHealth, summary="系统信息")
async def system_info(request: Request):
    """
    获取系统信息

    Args:
        request: FastAPI请求对象

    Returns:
        SystemHealth: 系统信息
    """
    try:
        return get_system_info()
    except Exception as e:
        structured_logger.error(f"获取系统信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取系统信息失败")


@router.get("/services", summary="服务状态")
async def services_status(request: Request):
    """
    获取各服务状态

    Args:
        request: FastAPI请求对象

    Returns:
        Dict: 各服务状态
    """
    try:
        services = {}

        # 检查各个服务
        service_names = ['llm', 'embedding', 'rerank', 'memory']
        for service_name in service_names:
            service = getattr(request.app.state, f'{service_name}_service', None)
            services[service_name] = await check_service_health(service)

        return {"services": services}

    except Exception as e:
        structured_logger.error(f"获取服务状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取服务状态失败")