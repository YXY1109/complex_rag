"""
Sanic高性能AI服务主入口

创建和配置Sanic应用实例，提供OpenAI兼容的AI服务接口。
"""

import asyncio
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import sanic
from sanic import Sanic, Request, Response
from sanic.response import json, stream
from sanic_cors import CORS
from sanic.middleware.http import Middleware
from sanic.log import logger
from sanic.exceptions import SanicException

from .services.llm_service import LLMService
from .services.embedding_service import EmbeddingService
from .services.rerank_service import RerankService
from .services.memory_service import MemoryService
from .interfaces.llm_interface import LLMConfig
from .interfaces.embedding_interface import EmbeddingConfig
from .interfaces.rerank_interface import RerankConfig
from .routes.llm import llm_router
from .routes.embeddings import embeddings_router
from .routes.rerank import rerank_router
from .routes.memory import memory_router
from .routes.health import health_router
from .middleware import setup_middleware
from .exceptions import setup_exception_handlers
from ..config.services.rag_service_config import RAGServiceConfig


class SanicRAGService:
    """
    Sanic高性能RAG服务

    提供OpenAI兼容的AI服务接口，包括LLM、Embedding、Rerank等服务。
    """

    def __init__(self, config: RAGServiceConfig):
        """
        初始化Sanic RAG服务

        Args:
            config: RAG服务配置
        """
        self.config = config
        self.app: Optional[Sanic] = None
        self.llm_service: Optional[LLMService] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.rerank_service: Optional[RerankService] = None
        self.memory_service: Optional[MemoryService] = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> bool:
        """
        初始化服务

        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("初始化Sanic RAG服务...")

            # 创建Sanic应用
            self.app = self._create_sanic_app()

            # 初始化服务
            await self._initialize_services()

            # 注册路由
            self._register_routes()

            # 设置中间件
            setup_middleware(self.app)

            # 设置异常处理器
            setup_exception_handlers(self.app)

            logger.info("Sanic RAG服务初始化成功")
            return True

        except Exception as e:
            logger.error(f"Sanic RAG服务初始化失败: {e}")
            return False

    def _create_sanic_app(self) -> Sanic:
        """
        创建Sanic应用实例

        Returns:
            Sanic: 配置好的Sanic应用
        """
        app = Sanic(
            name="rag-service",
            version="1.0.0",
            description="高性能RAG服务，提供OpenAI兼容的AI接口",
        )

        # 配置CORS
        CORS(
            app,
            resources={r"/*": {"origins": "*"}},
            expose_headers=["*"],
            allow_headers=["*"],
            allow_methods=["*"],
        )

        # 设置应用配置
        app.ctx.config = self.config
        app.ctx.service = self

        # 配置信号处理
        app.signal(sanic.server.READY).partial(self._on_server_ready)
        app.signal(sanic.server.BEFORE_STOP).partial(self._on_server_stop)

        return app

    async def _initialize_services(self) -> None:
        """初始化所有服务"""
        logger.info("初始化AI服务...")

        # 初始化LLM服务
        if self.config.llm:
            llm_config = LLMConfig(**self.config.llm.dict())
            self.llm_service = LLMService(llm_config)
            await self.llm_service.initialize()
            logger.info("LLM服务初始化成功")

        # 初始化Embedding服务
        if self.config.embedding:
            embedding_config = EmbeddingConfig(**self.config.embedding.dict())
            self.embedding_service = EmbeddingService(embedding_config)
            await self.embedding_service.initialize()
            logger.info("Embedding服务初始化成功")

        # 初始化Rerank服务
        if self.config.rerank:
            rerank_config = RerankConfig(**self.config.rerank.dict())
            self.rerank_service = RerankService(rerank_config)
            await self.rerank_service.initialize()
            logger.info("Rerank服务初始化成功")

        # 初始化Memory服务
        if self.config.memory:
            self.memory_service = MemoryService(self.config.memory)
            await self.memory_service.initialize()
            logger.info("Memory服务初始化成功")

    def _register_routes(self) -> None:
        """注册所有路由"""
        # API v1 路由前缀
        api_prefix = "/v1"

        # 注册OpenAI兼容的路由
        if self.llm_service:
            self.app.blueprint(llm_router, url_prefix=f"{api_prefix}")
            logger.info("LLM路由注册成功")

        if self.embedding_service:
            self.app.blueprint(embeddings_router, url_prefix=f"{api_prefix}")
            logger.info("Embedding路由注册成功")

        if self.rerank_service:
            self.app.blueprint(rerank_router, url_prefix=f"{api_prefix}")
            logger.info("Rerank路由注册成功")

        if self.memory_service:
            self.app.blueprint(memory_router, url_prefix=f"{api_prefix}")
            logger.info("Memory路由注册成功")

        # 注册健康检查路由
        self.app.blueprint(health_router)
        logger.info("健康检查路由注册成功")

        # 根路径处理器
        @self.app.get("/")
        async def root(request: Request) -> Response:
            """根路径处理器"""
            return json({
                "message": "复杂RAG服务",
                "version": "1.0.0",
                "status": "running",
                "services": {
                    "llm": self.llm_service is not None,
                    "embedding": self.embedding_service is not None,
                    "rerank": self.rerank_service is not None,
                    "memory": self.memory_service is not None,
                },
                "docs": {
                    "openapi": "/v1/openapi.json",
                    "health": "/health",
                    "models": "/v1/models",
                }
            })

    async def _on_server_ready(self, app: Sanic, **kwargs) -> None:
        """服务器启动就绪回调"""
        logger.info("Sanic RAG服务启动完成")
        logger.info(f"服务地址: http://localhost:{app.config.PORT}")
        logger.info("API文档: /v1/openapi.json")

    async def _on_server_stop(self, app: Sanic, **kwargs) -> None:
        """服务器停止回调"""
        logger.info("正在停止Sanic RAG服务...")
        await self.cleanup()
        logger.info("Sanic RAG服务已停止")

    async def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.llm_service:
                await self.llm_service.cleanup()
                logger.info("LLM服务资源清理完成")

            if self.embedding_service:
                await self.embedding_service.cleanup()
                logger.info("Embedding服务资源清理完成")

            if self.rerank_service:
                await self.rerank_service.cleanup()
                logger.info("Rerank服务资源清理完成")

            if self.memory_service:
                await self.memory_service.cleanup()
                logger.info("Memory服务资源清理完成")

        except Exception as e:
            logger.error(f"资源清理时发生错误: {e}")

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        debug: bool = False,
        access_log: bool = True,
        **kwargs
    ) -> None:
        """
        运行Sanic服务

        Args:
            host: 监听地址
            port: 监听端口
            workers: 工作进程数（单进程模式）
            debug: 调试模式
            access_log: 访问日志
            **kwargs: 其他Sanic运行参数
        """
        if not self.app:
            raise RuntimeError("服务未初始化，请先调用initialize()")

        # 配置Sanic运行参数
        run_config = {
            "host": host,
            "port": port,
            "workers": workers,  # 单进程模式，便于调试
            "debug": debug,
            "access_log": access_log,
            "auto_reload": debug,
            "motd": False,
            "fast": True,
            "single_process": True,  # 单进程模式
            **kwargs
        }

        logger.info(f"启动Sanic RAG服务 - {host}:{port}")

        # 设置信号处理
        self._setup_signal_handlers()

        try:
            self.app.run(**run_config)
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在关闭服务...")
        except Exception as e:
            logger.error(f"服务运行时发生错误: {e}")
            raise

    def _setup_signal_handlers(self) -> None:
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，正在优雅关闭...")
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def get_app(self) -> Sanic:
        """
        获取Sanic应用实例

        Returns:
            Sanic: Sanic应用实例
        """
        if not self.app:
            raise RuntimeError("服务未初始化")
        return self.app

    # 服务访问方法
    def get_llm_service(self) -> Optional[LLMService]:
        """获取LLM服务"""
        return self.llm_service

    def get_embedding_service(self) -> Optional[EmbeddingService]:
        """获取Embedding服务"""
        return self.embedding_service

    def get_rerank_service(self) -> Optional[RerankService]:
        """获取Rerank服务"""
        return self.rerank_service

    def get_memory_service(self) -> Optional[MemoryService]:
        """获取Memory服务"""
        return self.memory_service


async def create_sanic_service(config: RAGServiceConfig) -> SanicRAGService:
    """
    创建Sanic RAG服务

    Args:
        config: RAG服务配置

    Returns:
        SanicRAGService: 配置好的服务实例
    """
    service = SanicRAGService(config)
    success = await service.initialize()

    if not success:
        raise RuntimeError("Sanic服务初始化失败")

    return service


def create_app(config: Dict[str, Any]) -> Sanic:
    """
    创建Sanic应用（用于外部调用）

    Args:
        config: 配置字典

    Returns:
        Sanic: Sanic应用实例
    """
    service_config = RAGServiceConfig(**config)
    service = SanicRAGService(service_config)

    # 初始化服务（同步方式）
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(service.initialize())
        return service.get_app()
    finally:
        loop.close()


if __name__ == "__main__":
    # 测试运行
    import yaml

    # 加载配置
    with open("config/services/rag_service_config.yaml", "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    config = RAGServiceConfig(**config_data)

    # 创建并运行服务
    async def main():
        service = await create_sanic_service(config)
        service.run(host="0.0.0.0", port=8000, debug=True)

    asyncio.run(main())