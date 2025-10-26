"""
向后兼容性路由
为旧的Sanic服务端点提供弃用警告和迁移指导
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Request
from config.loguru_config import get_logger

# 创建路由器
router = APIRouter()
structured_logger = get_logger("api.legacy_compatibility")


@router.get("/v1/chat/completions")
@router.post("/v1/chat/completions")
@router.get("/v1/chat/models")
@router.post("/v1/embeddings")
@router.get("/v1/embeddings/models")
@router.post("/v1/embeddings/similarity")
@router.post("/v1/embeddings/batch")
@router.post("/v1/rerank")
async def legacy_deprecation_warning(request: Request) -> Dict[str, Any]:
    """
    旧API端点的弃用警告

    Args:
        request: FastAPI请求对象

    Returns:
        Dict: 弃用警告和迁移信息
    """
    request_id = getattr(request.state, "request_id", "unknown")
    method = request.method
    path = request.url.path

    structured_logger.warning(
        "旧API端点访问",
        extra={
            "request_id": request_id,
            "method": method,
            "path": path,
            "user_agent": request.headers.get("user-agent", "unknown"),
        }
    )

    # 构建新的端点路径
    new_path = path.replace("/v1/", "/v1/")
    if "/chat/" in path:
        new_path = path.replace("/v1/chat/", "/v1/chat/")
    elif "/embeddings/" in path:
        new_path = path.replace("/v1/embeddings/", "/v1/embeddings/")
    elif "/rerank/" in path:
        new_path = path.replace("/v1/rerank/", "/v1/rerank/")

    return {
        "error": {
            "message": "This API endpoint is deprecated. Please use the new unified FastAPI endpoints.",
            "type": "deprecated_endpoint",
            "code": "deprecated_api",
            "request_id": request_id,
            "deprecated_endpoint": path,
            "new_endpoint": new_path,
            "migration_guide": {
                "chat_completions": "Use POST /v1/chat/completions",
                "embeddings": "Use POST /v1/embeddings/",
                "rerank": "Use POST /v1/rerank/",
                "models": "Use GET /v1/{service}/models",
            },
            "deprecation_date": "2024-01-01",
            "removal_date": "2024-03-01",
            "affected_services": [
                "LLM Service",
                "Embedding Service",
                "Rerank Service"
            ]
        },
        "alternatives": [
            {
                "name": "New Unified Chat API",
                "endpoint": "/v1/chat/completions",
                "description": "OpenAI-compatible chat completions with enhanced features"
            },
            {
                "name": "New Unified Embeddings API",
                "endpoint": "/v1/embeddings/",
                "description": "High-performance text embeddings with batch processing"
            },
            {
                "name": "New Unified Rerank API",
                "endpoint": "/v1/rerank/",
                "description": "Advanced document reranking with multiple models"
            }
        ],
        "documentation": {
            "api_docs": "/docs",
            "openapi_spec": "/openapi.json",
            "migration_guide": "https://docs.example.com/migration-v1-to-v2"
        }
    }


@router.get("/health")
async def legacy_health_check() -> Dict[str, Any]:
    """
    旧健康检查端点的兼容性处理

    Returns:
        Dict: 健康状态信息
    """
    structured_logger.info("旧健康检查端点访问")

    return {
        "status": "ok",
        "message": "Legacy health check endpoint - please migrate to /api/health",
        "deprecated": True,
        "new_endpoint": "/api/health",
        "services": {
            "unified_chat": "healthy",
            "unified_embeddings": "healthy",
            "unified_rerank": "healthy",
            "legacy_sanic": "deprecated"
        }
    }


structured_logger.info("向后兼容性路由加载完成")