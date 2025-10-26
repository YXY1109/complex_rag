"""
健康检查API路由
提供系统健康状态检查和监控功能
"""
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from infrastructure.monitoring.loguru_logger import logger
from infrastructure.database.implementations.mysql_client_adapter import MySQLClient
from infrastructure.database.implementations.milvus_client_adapter import MilvusClient
from infrastructure.database.implementations.elasticsearch_client_adapter import ElasticsearchClient
from infrastructure.cache.implementations.redis_cache_adapter import RedisCache

router = APIRouter()


class HealthStatus(BaseModel):
    """健康状态响应模型"""
    status: str
    timestamp: str
    version: str
    uptime: str
    services: Dict[str, Dict[str, Any]]


class ServiceHealth(BaseModel):
    """服务健康状态模型"""
    status: str  # healthy, unhealthy, degraded
    response_time: float  # 响应时间（毫秒）
    last_check: str
    details: Dict[str, Any] = {}


# 全局变量记录启动时间
_start_time = datetime.utcnow()


def get_uptime() -> str:
    """获取系统运行时间"""
    uptime = datetime.utcnow() - _start_time
    days = uptime.days
    hours, remainder = divmod(uptime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days > 0:
        return f"{days}天 {hours}小时 {minutes}分钟"
    elif hours > 0:
        return f"{hours}小时 {minutes}分钟"
    elif minutes > 0:
        return f"{minutes}分钟 {seconds}秒"
    else:
        return f"{seconds}秒"


@router.get("/", response_model=HealthStatus, summary="系统健康检查")
async def health_check():
    """
    检查系统整体健康状态

    Returns:
        HealthStatus: 包含系统状态和各服务健康信息的响应
    """
    logger.info("执行系统健康检查")

    # 检查各个服务的健康状态
    services_status = {}

    # 检查MySQL数据库
    services_status["mysql"] = await check_mysql_health()

    # 检查Milvus向量数据库
    services_status["milvus"] = await check_milvus_health()

    # 检查Elasticsearch搜索引擎
    services_status["elasticsearch"] = await check_elasticsearch_health()

    # 检查Redis缓存
    services_status["redis"] = await check_redis_health()

    # 判断整体系统状态
    all_healthy = all(
        service["status"] == "healthy"
        for service in services_status.values()
    )

    overall_status = "healthy" if all_healthy else "degraded"

    return HealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0",
        uptime=get_uptime(),
        services=services_status
    )


@router.get("/detailed", summary="详细健康检查")
async def detailed_health_check():
    """
    获取详细的系统健康检查信息，包括性能指标

    Returns:
        Dict: 详细的健康检查信息
    """
    logger.info("执行详细系统健康检查")

    basic_health = await health_check()

    # 添加更多系统信息
    detailed_info = basic_health.dict()
    detailed_info.update({
        "system_info": {
            "environment": "development",  # 从配置获取
            "api_version": "1.0.0",
            "python_version": "3.9+",  # 动态获取
        },
        "performance_metrics": {
            "memory_usage": "N/A",  # 实现内存使用监控
            "cpu_usage": "N/A",     # 实现CPU使用监控
            "disk_usage": "N/A",    # 实现磁盘使用监控
        }
    })

    return detailed_info


@router.get("/service/{service_name}", summary="单个服务健康检查")
async def service_health_check(service_name: str):
    """
    检查指定服务的健康状态

    Args:
        service_name: 服务名称 (mysql, milvus, elasticsearch, redis)

    Returns:
        ServiceHealth: 指定服务的健康状态
    """
    logger.info(f"检查服务健康状态: {service_name}")

    service_checks = {
        "mysql": check_mysql_health,
        "milvus": check_milvus_health,
        "elasticsearch": check_elasticsearch_health,
        "redis": check_redis_health,
    }

    if service_name not in service_checks:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"未知服务名称: {service_name}. 支持的服务: {list(service_checks.keys())}"
        )

    service_health = await service_checks[service_name]()
    return ServiceHealth(**service_health)


async def check_mysql_health() -> Dict[str, Any]:
    """检查MySQL数据库健康状态"""
    start_time = datetime.utcnow()

    try:
        client = MySQLClient()
        # 简单的连接测试
        await client.execute_query("SELECT 1")

        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return {
            "status": "healthy",
            "response_time": round(response_time, 2),
            "last_check": datetime.utcnow().isoformat() + "Z",
            "details": {
                "connection": "successful",
                "database": "mysql"
            }
        }

    except Exception as e:
        logger.error(f"MySQL健康检查失败: {str(e)}")
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return {
            "status": "unhealthy",
            "response_time": round(response_time, 2),
            "last_check": datetime.utcnow().isoformat() + "Z",
            "details": {
                "connection": "failed",
                "error": str(e),
                "database": "mysql"
            }
        }


async def check_milvus_health() -> Dict[str, Any]:
    """检查Milvus向量数据库健康状态"""
    start_time = datetime.utcnow()

    try:
        client = MilvusClient()
        # 检查连接状态
        status = await client.health_check()

        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return {
            "status": "healthy" if status else "unhealthy",
            "response_time": round(response_time, 2),
            "last_check": datetime.utcnow().isoformat() + "Z",
            "details": {
                "connection": "successful" if status else "failed",
                "database": "milvus"
            }
        }

    except Exception as e:
        logger.error(f"Milvus健康检查失败: {str(e)}")
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return {
            "status": "unhealthy",
            "response_time": round(response_time, 2),
            "last_check": datetime.utcnow().isoformat() + "Z",
            "details": {
                "connection": "failed",
                "error": str(e),
                "database": "milvus"
            }
        }


async def check_elasticsearch_health() -> Dict[str, Any]:
    """检查Elasticsearch搜索引擎健康状态"""
    start_time = datetime.utcnow()

    try:
        client = ElasticsearchClient()
        # 检查集群健康状态
        health = await client.cluster_health()

        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        status_map = {
            "green": "healthy",
            "yellow": "degraded",
            "red": "unhealthy"
        }

        return {
            "status": status_map.get(health.get("status", "unknown"), "unknown"),
            "response_time": round(response_time, 2),
            "last_check": datetime.utcnow().isoformat() + "Z",
            "details": {
                "cluster_status": health.get("status"),
                "nodes": health.get("number_of_nodes"),
                "database": "elasticsearch"
            }
        }

    except Exception as e:
        logger.error(f"Elasticsearch健康检查失败: {str(e)}")
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return {
            "status": "unhealthy",
            "response_time": round(response_time, 2),
            "last_check": datetime.utcnow().isoformat() + "Z",
            "details": {
                "connection": "failed",
                "error": str(e),
                "database": "elasticsearch"
            }
        }


async def check_redis_health() -> Dict[str, Any]:
    """检查Redis缓存健康状态"""
    start_time = datetime.utcnow()

    try:
        cache = RedisCache()
        # 简单的ping测试
        result = await cache.ping()

        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return {
            "status": "healthy" if result else "unhealthy",
            "response_time": round(response_time, 2),
            "last_check": datetime.utcnow().isoformat() + "Z",
            "details": {
                "connection": "successful" if result else "failed",
                "database": "redis"
            }
        }

    except Exception as e:
        logger.error(f"Redis健康检查失败: {str(e)}")
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return {
            "status": "unhealthy",
            "response_time": round(response_time, 2),
            "last_check": datetime.utcnow().isoformat() + "Z",
            "details": {
                "connection": "failed",
                "error": str(e),
                "database": "redis"
            }
        }