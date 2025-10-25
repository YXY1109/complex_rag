"""
统一RAG服务

提供统一的RAG服务入口，整合所有RAG组件，
简化使用和部署。
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, AsyncGenerator

from ..interfaces.rag_interface import (
    RAGInterface, RAGConfig, RAGQuery, RAGResult, RetrievalResult,
    KnowledgeBase, ChatSession, ChatInterface
)
from ..core.rag_engine import RAGEngine
from ..services.retrieval_engine import RetrievalEngine
from ..services.context_builder import ContextBuilder
from ..services.generation_service import GenerationService
from ..services.document_ranker import DocumentRanker
from ..services.knowledge_manager import KnowledgeManager
from ..services.vector_store import VectorStore
from ..services.embedding_service import EmbeddingService
from ..services.chat_service import ChatService


class UnifiedRAGService(RAGInterface, ChatInterface):
    """统一RAG服务。"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化统一RAG服务。

        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 组件实例
        self.rag_engine: Optional[RAGEngine] = None
        self.retrieval_engine: Optional[RetrievalEngine] = None
        self.context_builder: Optional[ContextBuilder] = None
        self.generation_service: Optional[GenerationService] = None
        self.document_ranker: Optional[DocumentRanker] = None
        self.knowledge_manager: Optional[KnowledgeManager] = None
        self.vector_store: Optional[VectorStore] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.chat_service: Optional[ChatService] = None

        # 配置
        self.default_rag_config = RAGConfig(**config.get("rag", {}))

        # 状态
        self.is_initialized = False
        self.initialization_time: Optional[datetime] = None

    async def initialize(self, config: Optional[RAGConfig] = None) -> bool:
        """
        初始化统一RAG服务。

        Args:
            config: RAG配置

        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info("开始初始化统一RAG服务")

            # 初始化配置
            if config:
                await self.rag_engine.initialize_config(config) if self.rag_engine else None

            # 初始化底层服务
            await self._initialize_services()

            # 初始化核心引擎
            await self._initialize_rag_engine()

            self.is_initialized = True
            self.initialization_time = datetime.now()

            self.logger.info("统一RAG服务初始化成功")
            return True

        except Exception as e:
            self.logger.error(f"统一RAG服务初始化失败: {e}")
            return False

    async def _initialize_services(self) -> None:
        """初始化底层服务。"""
        # 初始化向量存储
        self.vector_store = VectorStore(self.config.get("vector_store", {}))
        if await self.vector_store.initialize():
            self.logger.info("向量存储服务初始化成功")
        else:
            raise Exception("向量存储服务初始化失败")

        # 初始化嵌入服务
        self.embedding_service = EmbeddingService(self.config.get("embedding", {}))
        if await self.embedding_service.initialize():
            self.logger.info("嵌入服务初始化成功")
        else:
            raise Exception("嵌入服务初始化失败")

        # 初始化知识管理服务
        self.knowledge_manager = KnowledgeManager(self.config.get("knowledge_manager", {}))
        # 注意：知识管理需要依赖服务，在后面初始化

        # 初始化检索引擎
        self.retrieval_engine = RetrievalEngine(self.config.get("retrieval", {}))
        if await self.retrieval_engine.initialize(
            self.vector_store,
            self.embedding_service,
            self.knowledge_manager
        ):
            self.logger.info("检索引擎初始化成功")
        else:
            raise Exception("检索引擎初始化失败")

        # 初始化上下文构建器
        self.context_builder = ContextBuilder(self.config.get("context_builder", {}))
        self.logger.info("上下文构建器初始化成功")

        # 初始化生成服务
        self.generation_service = GenerationService(self.config.get("generation", {}))
        self.logger.info("生成服务初始化成功")

        # 初始化文档重排服务
        self.document_ranker = DocumentRanker(self.config.get("rerank", {}))
        self.logger.info("文档重排服务初始化成功")

        # 初始化知识管理服务（依赖其他服务）
        if await self.knowledge_manager.initialize(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service
        ):
            self.logger.info("知识管理服务初始化成功")
        else:
            raise Exception("知识管理服务初始化失败")

        # 初始化聊天服务
        self.chat_service = ChatService(self.config.get("chat", {}))
        if await self.chat_service.initialize(
            self.retrieval_engine,
            self.context_builder,
            self.generation_service
        ):
            self.logger.info("聊天服务初始化成功")
        else:
            raise Exception("聊天服务初始化失败")

    async def _initialize_rag_engine(self) -> None:
        """初始化RAG引擎。"""
        self.rag_engine = RAGEngine(self.config.get("rag_engine", {}))
        if await self.rag_engine.initialize(
            self.retrieval_engine,
            self.context_builder,
            self.generation_service,
            self.document_ranker,
            self.knowledge_manager
        ):
            self.logger.info("RAG引擎初始化成功")
        else:
            raise Exception("RAG引擎初始化失败")

    async def cleanup(self) -> None:
        """清理统一RAG服务资源。"""
        try:
            if self.rag_engine:
                await self.rag_engine.cleanup()
            if self.chat_service:
                await self.chat_service.cleanup()
            if self.retrieval_engine:
                # RetrievalEngine没有cleanup方法，跳过
                pass
            if self.context_builder:
                # ContextBuilder没有cleanup方法，跳过
                pass
            if self.generation_service:
                await self.generation_service.cleanup()
            if self.document_ranker:
                # DocumentRanker没有cleanup方法，跳过
                pass
            if self.knowledge_manager:
                await self.knowledge_manager.cleanup()
            if self.vector_store:
                await self.vector_store.cleanup()
            if self.embedding_service:
                await self.embedding_service.cleanup()

            self.is_initialized = False
            self.logger.info("统一RAG服务资源清理完成")

        except Exception as e:
            self.logger.error(f"统一RAG服务清理失败: {e}")

    # RAG接口实现
    async def query(self, query: RAGQuery) -> RAGResult:
        """执行RAG查询。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        if not self.rag_engine:
            raise RuntimeError("RAG引擎未初始化")

        return await self.rag_engine.query(query)

    async def query_stream(self, query: RAGQuery) -> AsyncGenerator[str, None]:
        """流式RAG查询。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        if not self.rag_engine:
            raise RuntimeError("RAG引擎未初始化")

        async for chunk in self.rag_engine.query_stream(query):
            yield chunk

    async def batch_query(self, queries: List[RAGQuery], max_concurrent: Optional[int] = None) -> List[RAGResult]:
        """批量RAG查询。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        return await self.rag_engine.batch_query(queries, max_concurrent)

    # 聊天接口实现
    async def create_session(
        self,
        user_id: str,
        tenant_id: str,
        title: Optional[str] = None,
        knowledge_bases: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        """创建聊天会话。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        return await self.chat_service.create_session(
            user_id=user_id,
            tenant_id=tenant_id,
            title=title,
            knowledge_bases=knowledge_bases,
            config=config,
            metadata=metadata
        )

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """获取聊天会话。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        return await self.chat_service.get_session(session_id)

    async def chat(
        self,
        session_id: str,
        message: str,
        config: Optional[Dict[str, Any]] = None
    ) -> RAGResult:
        """发送聊天消息。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        return await self.chat_service.chat(session_id, message, config)

    async def delete_session(self, session_id: str) -> bool:
        """删除聊天会话。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        return await self.chat_service.delete_session(session_id)

    # 知识管理方法
    async def create_knowledge_base(
        self,
        name: str,
        description: str,
        tenant_id: str,
        config: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> KnowledgeBase:
        """创建知识库。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        return await self.knowledge_manager.create_knowledge_base(
            name=name,
            description=description,
            tenant_id=tenant_id,
            config=config,
            created_by=created_by
        )

    async def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """获取知识库。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        return await self.knowledge_manager.get_knowledge_base(kb_id)

    async def list_knowledge_bases(
        self,
        tenant_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[KnowledgeBase]:
        """列出知识库。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        return await self.knowledge_manager.list_knowledge_bases(tenant_id, limit, offset)

    async def delete_knowledge_base(self, kb_id: str) -> bool:
        """删除知识库。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        return await self.knowledge_manager.delete_knowledge_base(kb_id)

    async def update_knowledge_base(self, kb_id: str, updates: Dict[str, Any]) -> bool:
        """更新知识库。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        return await self.knowledge_manager.update_knowledge_base(kb_id, updates)

    async def add_document_to_kb(
        self,
        kb_id: str,
        title: str,
        content: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> str:
        """添加文档到知识库。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        return await self.knowledge_manager.add_document(
            kb_id=kb_id,
            title=title,
            content=content,
            file_path=file_path,
            file_type=file_type,
            metadata=metadata,
            created_by=created_by
        )

    async def search_documents(
        self,
        kb_id: str,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """在知识库中搜索文档。"""
        if not self.is_initialized:
            raise RuntimeError("RAG服务未初始化")

        chunks = await self.knowledge_manager.search_documents(
            kb_id=kb_id,
            query=query,
            top_k=top_k,
            filters=filters
        )

        # 转换为字典格式
        return [
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "score": chunk.score,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]

    # 高级方法
    async def simple_qa(
        self,
        question: str,
        knowledge_bases: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        简单问答接口。

        Args:
            question: 问题
            knowledge_bases: 知识库列表
            user_id: 用户ID
            tenant_id: 租户ID

        Returns:
            Dict[str, Any]: 问答结果
        """
        # 创建RAG查询
        rag_query = RAGQuery(
            query_id=str(hash(question + str(datetime.now()))),
            query=question,
            knowledge_bases=knowledge_bases or [],
            user_id=user_id,
            tenant_id=tenant_id
        )

        # 执行查询
        result = await self.query(rag_query)

        return {
            "question": question,
            "answer": result.answer,
            "sources": [
                {
                    "title": chunk.title,
                    "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "score": chunk.score,
                    "metadata": chunk.metadata
                }
                for chunk in result.retrieval_result.chunks
            ],
            "metadata": result.metadata
        }

    async def document_summary(
        self,
        document_content: str,
        max_length: int = 500
    ) -> str:
        """
        文档摘要。

        Args:
            document_content: 文档内容
            max_length: 最大长度

        Returns:
            str: 摘要结果
        """
        # 创建摘要查询
        rag_query = RAGQuery(
            query_id=str(hash(document_content[:100] + str(datetime.now()))),
            query=f"请对以下文档进行简洁的总结，不超过{max_length}字：\n\n{document_content}",
            generation_mode=GenerationMode.DIRECT,
            max_tokens=max_length
        )

        # 执行查询
        result = await self.query(rag_query)
        return result.answer

    async def compare_documents(
        self,
        documents: List[str],
        criteria: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        文档比较。

        Args:
            documents: 文档内容列表
            criteria: 比较标准

        Returns:
            List[Dict[str, Any]]: 比较结果
        """
        comparison_results = []

        for i, doc in enumerate(documents):
            # 创建分析查询
            analysis_query = RAGQuery(
                query_id=str(hash(f"compare_{i}" + str(datetime.now()))),
                query=f"请分析以下文档的主要特点、关键信息和适用场景：\n\n{doc}",
                generation_mode=GenerationMode.DIRECT,
                max_tokens=500
            )

            # 执行分析
            result = await self.query(analysis_query)

            comparison_results.append({
                "document_index": i,
                "summary": result.answer,
                "key_points": result.answer.split('\n')[:5],  # 前5个要点
                "metadata": result.metadata
            })

        return comparison_results

    # 管理方法
    async def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态。"""
        status = {
            "initialized": self.is_initialized,
            "initialization_time": self.initialization_time.isoformat() if self.initialization_time else None,
            "components": {}
        }

        if self.is_initialized:
            # 收集各组件状态
            components = {
                "rag_engine": self.rag_engine,
                "retrieval_engine": self.retrieval_engine,
                "context_builder": self.context_builder,
                "generation_service": self.generation_service,
                "document_ranker": self.document_ranker,
                "knowledge_manager": self.knowledge_manager,
                "vector_store": self.vector_store,
                "embedding_service": self.embedding_service,
                "chat_service": self.chat_service
            }

            for component_name, component in components.items():
                if component and hasattr(component, 'health_check'):
                    try:
                        component_health = await component.health_check()
                        status["components"][component_name] = component_health.get("status", "unknown")
                    except Exception as e:
                        status["components"][component_name] = f"error: {str(e)}"
                else:
                    status["components"][component_name] = "not_initialized" if not component else "available"

        return status

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息。"""
        stats = {
            "service_info": {
                "initialized": self.is_initialized,
                "initialization_time": self.initialization_time.isoformat() if self.initialization_time else None
            }
        }

        if self.is_initialized:
            # 收集各组件统计
            if self.rag_engine:
                rag_stats = await self.rag_engine.get_statistics()
                stats["rag_engine"] = rag_stats

            if self.chat_service:
                chat_stats = await self.chat_service.get_statistics()
                stats["chat_service"] = chat_stats

            if self.knowledge_manager:
                kb_stats = await self.knowledge_manager.get_knowledge_base_statistics("default")  # 简化处理
                stats["knowledge_manager"] = {"sample_stats": kb_stats}

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """健康检查。"""
        if not self.is_initialized:
            return {
                "status": "not_initialized",
                "message": "RAG服务未初始化",
                "components": {}
            }

        health_status = {
            "status": "healthy",
            "components": {},
            "issues": []
        }

        # 检查各组件健康状态
        components = {
            "rag_engine": self.rag_engine,
            "retrieval_engine": self.retrieval_engine,
            "context_builder": self.context_builder,
            "generation_service": self.generation_service,
            "document_ranker": self.document_ranker,
            "knowledge_manager": self.knowledge_manager,
            "vector_store": self.vector_store,
            "embedding_service": self.embedding_service,
            "chat_service": self.chat_service
        }

        unhealthy_count = 0
        for component_name, component in components.items():
            if component and hasattr(component, 'health_check'):
                try:
                    component_health = await component.health_check()
                    if component_health.get("status") != "healthy":
                        unhealthy_count += 1
                        health_status["issues"].append(f"{component_name}: {component_health.get('status')}")
                    health_status["components"][component_name] = component_health.get("status")
                except Exception as e:
                    unhealthy_count += 1
                    health_status["issues"].append(f"{component_name}: {str(e)}")
                    health_status["components"][component_name] = "error"
            else:
                health_status["components"][component_name] = "not_available"

        # 总体健康状态
        if unhealthy_count > 0:
            health_status["status"] = "degraded" if unhealthy_count < len(components) else "unhealthy"

        return health_status