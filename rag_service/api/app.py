"""
FastAPI应用主入口

创建和配置FastAPI应用实例，集成所有API路由和中间件。
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
from typing import Dict, Any

from ..services.unified_rag_service import UnifiedRAGService
from ..interfaces.rag_interface import RAGConfig
from .middleware import setup_middleware
from .routes import (
    rag_routes,
    chat_routes,
    knowledge_routes,
    document_routes,
    admin_routes,
    health_routes
)
from .exceptions import setup_exception_handlers


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    # 启动时初始化
    logger.info("RAG API服务启动中...")

    # 初始化RAG服务
    config = app.state.config
    rag_service = UnifiedRAGService(config)

    success = await rag_service.initialize()
    if not success:
        logger.error("RAG服务初始化失败")
        raise RuntimeError("RAG服务初始化失败")

    app.state.rag_service = rag_service
    logger.info("RAG服务初始化成功")

    yield

    # 关闭时清理
    logger.info("RAG API服务关闭中...")
    if hasattr(app.state, 'rag_service'):
        await app.state.rag_service.cleanup()
    logger.info("RAG服务资源清理完成")


def create_app(config: Dict[str, Any]) -> FastAPI:
    """
    创建FastAPI应用实例。

    Args:
        config: 应用配置

    Returns:
        FastAPI: 配置好的应用实例
    """
    # 创建FastAPI应用
    app = FastAPI(
        title="复杂RAG服务API",
        description="基于多模态文档解析和智能检索的RAG服务系统",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # 存储配置到应用状态
    app.state.config = config

    # 设置中间件
    setup_middleware(app)

    # 设置异常处理器
    setup_exception_handlers(app)

    # 注册路由
    _register_routes(app)

    # 添加根路径处理器
    @app.get("/")
    async def root():
        """根路径。"""
        return {
            "message": "复杂RAG服务API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/health"
        }

    return app


def _register_routes(app: FastAPI) -> None:
    """注册所有路由。"""
    # API v1 路由
    api_prefix = "/api/v1"

    # RAG核心功能路由
    app.include_router(
        rag_routes.router,
        prefix=f"{api_prefix}/rag",
        tags=["RAG核心功能"]
    )

    # 聊天功能路由
    app.include_router(
        chat_routes.router,
        prefix=f"{api_prefix}/chat",
        tags=["聊天功能"]
    )

    # 知识库管理路由
    app.include_router(
        knowledge_routes.router,
        prefix=f"{api_prefix}/knowledge",
        tags=["知识库管理"]
    )

    # 文档管理路由
    app.include_router(
        document_routes.router,
        prefix=f"{api_prefix}/documents",
        tags=["文档管理"]
    )

    # 管理功能路由
    app.include_router(
        admin_routes.router,
        prefix=f"{api_prefix}/admin",
        tags=["管理功能"]
    )

    # 健康检查路由
    app.include_router(
        health_routes.router,
        prefix="",
        tags=["健康检查"]
    )