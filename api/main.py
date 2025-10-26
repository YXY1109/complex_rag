"""
FastAPI主应用入口
高性能异步API服务，专注于核心RAG功能，无需用户认证
"""
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from config.settings_simple import get_settings
from config.loguru_config import setup_logging, get_logger

# 设置日志
logger = get_logger("api.main")
from api.middleware_simple import setup_middleware
from api.exceptions_simple import setup_exception_handlers
# from api.routers import chat, documents, knowledge, models, health, users, system, analytics
# 暂时注释掉旧路由，专注于统一路由
from api.routers import unified_chat, unified_embeddings, unified_rerank, legacy_compatibility


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    # 启动时执行
    logger.info("FastAPI应用正在启动...")
    startup_time = time.time()

    # 初始化配置
    settings = get_settings()
    logger.info(f"运行环境: {settings.environment}")
    logger.info(f"服务端口: {settings.api_port}")

    startup_duration = time.time() - startup_time
    logger.info(f"FastAPI应用启动完成，耗时: {startup_duration:.2f}秒")

    yield

    # 关闭时执行
    logger.info("FastAPI应用正在关闭...")


def create_fastapi_app() -> FastAPI:
    """
    创建并配置FastAPI应用实例

    Returns:
        FastAPI: 配置完成的FastAPI应用实例
    """
    # 获取配置
    settings = get_settings()

    # 创建FastAPI应用
    app = FastAPI(
        title="Complex RAG API",
        description="高性能RAG系统API服务，提供智能问答和文档检索功能",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        # 无需用户认证，所有接口开放访问
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

    # 设置中间件
    setup_middleware(app)

    # 设置异常处理器
    setup_exception_handlers(app)

    # 注册统一路由模块 (暂时注释掉旧路由)
    # app.include_router(
    #     health.router,
    #     prefix="/api/health",
    #     tags=["健康检查"]
    # )
    # ... 其他旧路由暂时注释

    # 添加统一路由 (OpenAI兼容API)
    app.include_router(
        unified_chat.router,
        prefix="/v1/chat",
        tags=["统一对话服务"]
    )

    app.include_router(
        unified_embeddings.router,
        prefix="/v1/embeddings",
        tags=["统一嵌入服务"]
    )

    app.include_router(
        unified_rerank.router,
        prefix="/v1/rerank",
        tags=["统一重排序服务"]
    )

    # 添加向后兼容性路由
    app.include_router(
        legacy_compatibility.router,
        tags=["向后兼容性"]
    )

    # 根路径
    @app.get("/", tags=["根路径"])
    async def root():
        return {
            "message": "Complex RAG API服务",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/api/health"
        }

    # 健康检查端点（额外的简单检查）
    @app.get("/ping", tags=["健康检查"])
    async def ping():
        return {"status": "ok", "message": "pong"}

    return app


# 创建应用实例
app = create_fastapi_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    # 运行FastAPI应用
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level="info",
        access_log=True,
        # 单进程模式，符合要求
        workers=1,
    )