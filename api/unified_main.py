"""
统一的FastAPI应用主入口
整合原有的FastAPI和Sanic功能，提供完整的RAG系统服务
"""

import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from config.settings_simple import get_settings
from config.loguru_config import setup_logging, get_logger
from config.unified_embedding_config import get_unified_embedding_config

# 设置日志
logger = get_logger("api.unified_main")

# 导入路由
from api.routers.unified_chat import router as chat_router
from api.routers.unified_embeddings import router as embeddings_router
from api.routers.unified_rerank import router as rerank_router
from api.routers.unified_memory import router as memory_router
from api.routers.unified_health import router as health_router

# 导入服务类（整合原Sanic服务）
from rag_service.services.llm_service import LLMService
from rag_service.services.unified_embedding_service import UnifiedEmbeddingService
from rag_service.services.rerank_service import RerankService
from rag_service.services.memory_service import MemoryService


class UnifiedRAGService:
    """
    统一的RAG服务

    整合原有的FastAPI和Sanic功能，提供完整的AI服务
    """

    def __init__(self):
        self.settings = get_settings()
        self.llm_service = None
        self.embedding_service = None
        self.rerank_service = None
        self.memory_service = None

    async def initialize(self):
        """初始化所有服务"""
        try:
            logger.info("正在初始化统一RAG服务...")

            # 初始化服务实例
            self.llm_service = LLMService()
            self.embedding_service = UnifiedEmbeddingService(get_unified_embedding_config())
            self.rerank_service = RerankService()
            self.memory_service = MemoryService()

            # 初始化各个服务
            await self.llm_service.initialize()
            await self.embedding_service.initialize()
            await self.rerank_service.initialize()
            await self.memory_service.initialize()

            logger.info("✅ 统一RAG服务初始化完成")

        except Exception as e:
            logger.error(f"❌ 统一RAG服务初始化失败: {str(e)}")
            raise

    async def shutdown(self):
        """关闭所有服务"""
        try:
            logger.info("正在关闭统一RAG服务...")

            if self.llm_service:
                await self.llm_service.shutdown()
            if self.embedding_service:
                await self.embedding_service.shutdown()
            if self.rerank_service:
                await self.rerank_service.shutdown()
            if self.memory_service:
                await self.memory_service.shutdown()

            logger.info("✅ 统一RAG服务关闭完成")

        except Exception as e:
            logger.error(f"❌ 统一RAG服务关闭失败: {str(e)}")


# 全局服务实例
unified_service = UnifiedRAGService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    # 启动时执行
    logger.info("🚀 统一FastAPI应用正在启动...")
    startup_time = time.time()

    try:
        # 初始化统一服务
        await unified_service.initialize()

        # 将服务实例注入到应用状态中，供路由使用
        app.state.llm_service = unified_service.llm_service
        app.state.embedding_service = unified_service.embedding_service
        app.state.rerank_service = unified_service.rerank_service
        app.state.memory_service = unified_service.memory_service
        app.state.start_time = startup_time

        startup_duration = time.time() - startup_time
        logger.info(f"✅ 统一FastAPI应用启动完成，耗时: {startup_duration:.2f}秒")

    except Exception as e:
        logger.error(f"❌ 应用启动失败: {str(e)}")
        raise

    yield

    # 关闭时执行
    logger.info("🔄 统一FastAPI应用正在关闭...")
    try:
        await unified_service.shutdown()
        logger.info("✅ 统一FastAPI应用关闭完成")
    except Exception as e:
        logger.error(f"❌ 应用关闭失败: {str(e)}")


def create_unified_app() -> FastAPI:
    """
    创建并配置统一的FastAPI应用实例

    Returns:
        FastAPI: 配置完成的FastAPI应用实例
    """
    # 获取配置
    settings = get_settings()

    # 创建FastAPI应用
    app = FastAPI(
        title="Complex RAG Unified API",
        description="统一的RAG系统API服务，整合聊天、嵌入、重排序和记忆管理功能",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        contact={
            "name": "Complex RAG Team",
            "email": "team@complexrag.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        }
    )

    # 配置CORS中间件 - 支持跨域请求
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源
        allow_credentials=True,
        allow_methods=["*"],  # 允许所有HTTP方法
        allow_headers=["*"],  # 允许所有请求头
    )

    # 配置受信任主机中间件
    if settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # 生产环境应根据实际配置调整
        )

    # 设置全局异常处理器
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"请求验证错误: {exc}")
        return JSONResponse(
            status_code=422,
            content={"error": "Validation Error", "details": exc.errors()}
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "HTTP Error", "message": exc.detail}
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": "服务内部错误"}
        )

    # 注册核心路由
    app.include_router(
        chat_router,
        prefix="/v1/chat",
        tags=["聊天服务"]
    )

    app.include_router(
        embeddings_router,
        prefix="/v1/embeddings",
        tags=["嵌入服务"]
    )

    app.include_router(
        rerank_router,
        prefix="/v1/rerank",
        tags=["重排序服务"]
    )

    app.include_router(
        memory_router,
        prefix="/v1/memory",
        tags=["记忆服务"]
    )

    app.include_router(
        health_router,
        prefix="/health",
        tags=["健康检查"]
    )

  
    # 根路径
    @app.get("/", tags=["根路径"])
    async def root():
        return {
            "service": "Complex RAG Unified API",
            "version": "2.0.0",
            "status": "running",
            "features": [
                "聊天对话 (LLM)",
                "文本嵌入 (Embedding)",
                "文档重排序 (Rerank)",
                "记忆管理 (Memory)"
            ],
            "endpoints": {
                "chat": "/v1/chat/completions",
                "embeddings": "/v1/embeddings/",
                "rerank": "/v1/rerank/",
                "memory": "/v1/memory/",
                "docs": "/docs",
                "health": "/health/detailed"
            }
        }

    # 简单健康检查
    @app.get("/ping", tags=["健康检查"])
    async def ping():
        return {"status": "ok", "message": "pong", "service": "unified-rag"}

    # K8s就绪检查
    @app.get("/health/ready", tags=["健康检查"])
    async def ready_check():
        """Kubernetes就绪检查"""
        try:
            # 检查关键服务是否就绪
            if not all([
                unified_service.llm_service,
                unified_service.embedding_service,
                unified_service.rerank_service
            ]):
                raise HTTPException(status_code=503, detail="服务未就绪")

            return {"status": "ready"}
        except Exception:
            return JSONResponse(
                status_code=503,
                content={"status": "not ready"}
            )

    # K8s存活检查
    @app.get("/health/live", tags=["健康检查"])
    async def live_check():
        """Kubernetes存活检查"""
        return {"status": "alive"}

    return app


# 创建应用实例
app = create_unified_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    # 运行统一的FastAPI应用
    uvicorn.run(
        "api.unified_main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level="info",
        access_log=True,
        workers=1,  # 单进程模式
    )