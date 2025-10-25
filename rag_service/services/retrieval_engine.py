"""
检索引擎

基于RAGFlow架构的高性能检索引擎，
支持多种检索模式、查询理解、结果融合等功能。
"""

import asyncio
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import numpy as np
from dataclasses import dataclass, field

from ..interfaces.rag_interface import (
    RetrievalInterface, RetrievalResult, DocumentChunk, RetrievalException,
    RAGQuery, RetrievalMode
)
from .vector_store import VectorStore
from .embedding_service import EmbeddingService
from .knowledge_manager import KnowledgeManager


@dataclass
class QueryExpansion:
    """查询扩展。"""

    original_query: str
    expanded_queries: List[str]
    expansion_method: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    """检索配置。"""

    # 基础配置
    top_k: int = 10
    similarity_threshold: float = 0.7
    max_context_length: int = 4000

    # 检索模式配置
    retrieval_modes: List[RetrievalMode] = field(default_factory=lambda: [RetrievalMode.VECTOR])
    mode_weights: Dict[str, float] = field(default_factory=lambda: {
        "vector": 0.7,
        "keyword": 0.3
    })

    # 查询处理配置
    enable_query_expansion: bool = True
    enable_query_decomposition: bool = True
    enable_hyde: bool = False  # Hypothetical Document Embeddings
    enable_self_query: bool = True

    # 过滤和重排配置
    enable_reranking: bool = True
    enable_filtering: bool = True
    enable_deduplication: bool = True

    # 性能配置
    parallel_retrieval: bool = True
    cache_results: bool = True
    timeout_seconds: int = 30


class RetrievalEngine(RetrievalInterface):
    """检索引擎。"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化检索引擎。

        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 依赖服务
        self.vector_store: Optional[VectorStore] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.knowledge_manager: Optional[KnowledgeManager] = None

        # 检索配置
        self.default_config = RetrievalConfig(**config.get("retrieval", {}))

        # 查询处理器
        self.query_processors: Dict[str, Callable] = {}
        self._init_query_processors()

        # 缓存
        self.retrieval_cache: Dict[str, RetrievalResult] = {}
        self.cache_ttl = config.get("cache_ttl", 3600)

        # 统计信息
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "average_retrieval_time": 0.0,
            "mode_usage": {},
            "query_expansions": 0
        }

    def _init_query_processors(self) -> None:
        """初始化查询处理器。"""
        self.query_processors = {
            "basic_preprocessing": self._basic_query_preprocessing,
            "keyword_extraction": self._extract_keywords,
            "query_expansion": self._expand_query,
            "hyde_generation": self._generate_hyde_documents,
            "query_decomposition": self._decompose_query
        }

    async def initialize(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        knowledge_manager: KnowledgeManager
    ) -> bool:
        """
        初始化检索引擎。

        Args:
            vector_store: 向量存储服务
            embedding_service: 嵌入服务
            knowledge_manager: 知识管理服务

        Returns:
            bool: 初始化是否成功
        """
        try:
            self.vector_store = vector_store
            self.embedding_service = embedding_service
            self.knowledge_manager = knowledge_manager

            self.logger.info("检索引擎初始化成功")
            return True

        except Exception as e:
            self.logger.error(f"检索引擎初始化失败: {e}")
            return False

    async def cleanup(self) -> None:
        """清理检索引擎资源。"""
        try:
            self.retrieval_cache.clear()
            self.query_processors.clear()
            self.logger.info("检索引擎资源清理完成")

        except Exception as e:
            self.logger.error(f"检索引擎清理失败: {e}")

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        knowledge_bases: Optional[List[str]] = None,
        config: Optional[RetrievalConfig] = None
    ) -> RetrievalResult:
        """
        检索相关文档。

        Args:
            query: 查询字符串
            top_k: 返回结果数量
            filters: 过滤条件
            knowledge_bases: 知识库列表
            config: 检索配置

        Returns:
            RetrievalResult: 检索结果
        """
        start_time = datetime.now()
        query_id = str(hash(query + str(datetime.now())))

        self.stats["total_queries"] += 1

        try:
            # 使用配置
            retrieval_config = config or self.default_config

            # 检查缓存
            cache_key = self._get_cache_key(query, top_k, filters, knowledge_bases)
            if retrieval_config.cache_results and cache_key in self.retrieval_cache:
                self.stats["cache_hits"] += 1
                cached_result = self.retrieval_cache[cache_key]
                self.logger.info(f"检索缓存命中: {query[:50]}...")
                return cached_result

            # 查询预处理
            processed_query = await self._preprocess_query(query, retrieval_config)

            # 执行多模态检索
            if retrieval_config.parallel_retrieval:
                results = await self._parallel_retrieve(
                    processed_query,
                    top_k,
                    filters,
                    knowledge_bases,
                    retrieval_config
                )
            else:
                results = await self._sequential_retrieve(
                    processed_query,
                    top_k,
                    filters,
                    knowledge_bases,
                    retrieval_config
                )

            # 后处理
            final_results = await self._postprocess_results(
                results,
                query,
                retrieval_config
            )

            # 创建检索结果
            retrieval_result = RetrievalResult(
                query_id=query_id,
                chunks=final_results[:retrieval_config.top_k],
                total_found=len(final_results),
                search_time=(datetime.now() - start_time).total_seconds(),
                retrieval_metadata={
                    "original_query": query,
                    "processed_query": processed_query.query,
                    "modes_used": retrieval_config.retrieval_modes,
                    "expansions": len(processed_query.expanded_queries) if processed_query.expanded_queries else 0
                }
            )

            # 缓存结果
            if retrieval_config.cache_results:
                self.retrieval_cache[cache_key] = retrieval_result

            # 更新统计
            retrieval_time = retrieval_result.search_time
            self.stats["average_retrieval_time"] = (
                (self.stats["average_retrieval_time"] * (self.stats["total_queries"] - 1) + retrieval_time) /
                self.stats["total_queries"]
            )

            for mode in retrieval_config.retrieval_modes:
                mode_name = mode.value
                if mode_name not in self.stats["mode_usage"]:
                    self.stats["mode_usage"][mode_name] = 0
                self.stats["mode_usage"][mode_name] += 1

            self.logger.info(f"检索完成，返回 {len(retrieval_result.chunks)} 个结果，耗时 {retrieval_time:.3f}s")

            return retrieval_result

        except Exception as e:
            self.logger.error(f"检索失败: {e}")
            raise RetrievalException(f"检索失败: {str(e)}")

    async def _preprocess_query(self, query: str, config: RetrievalConfig) -> QueryExpansion:
        """查询预处理。"""
        try:
            # 基础预处理
            processed_query = await self.query_processors["basic_preprocessing"](query)

            # 查询扩展
            expanded_queries = []
            if config.enable_query_expansion:
                expansion_result = await self.query_processors["query_expansion"](processed_query)
                expanded_queries.extend(expansion_result.expanded_queries)
                self.stats["query_expansions"] += 1

            # HyDE文档生成
            if config.enable_hyde:
                hyde_docs = await self.query_processors["hyde_generation"](processed_query)
                expanded_queries.extend(hyde_docs)

            # 查询分解
            if config.enable_query_decomposition:
                decomposed_queries = await self.query_processors["query_decomposition"](processed_query)
                expanded_queries.extend(decomposed_queries)

            return QueryExpansion(
                original_query=query,
                expanded_queries=expanded_queries,
                expansion_method="combined",
                metadata={"processed": processed_query}
            )

        except Exception as e:
            self.logger.error(f"查询预处理失败: {e}")
            return QueryExpansion(
                original_query=query,
                expanded_queries=[query],
                expansion_method="fallback"
            )

    async def _parallel_retrieve(
        self,
        query_expansion: QueryExpansion,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        knowledge_bases: Optional[List[str]],
        config: RetrievalConfig
    ) -> List[DocumentChunk]:
        """并行检索。"""
        tasks = []

        # 原始查询
        tasks.append(asyncio.create_task(
            self._retrieve_by_modes(
                query_expansion.original_query,
                top_k,
                filters,
                knowledge_bases,
                config.retrieval_modes
            )
        ))

        # 扩展查询
        for expanded_query in query_expansion.expanded_queries[:3]:  # 限制扩展查询数量
            tasks.append(asyncio.create_task(
                self._retrieve_by_modes(
                    expanded_query,
                    top_k // 2,
                    filters,
                    knowledge_bases,
                    config.retrieval_modes
                )
            ))

        # 等待所有检索完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并结果
        all_chunks = []
        for result in results:
            if isinstance(result, list):
                all_chunks.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"并行检索任务失败: {result}")

        return all_chunks

    async def _sequential_retrieve(
        self,
        query_expansion: QueryExpansion,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        knowledge_bases: Optional[List[str]],
        config: RetrievalConfig
    ) -> List[DocumentChunk]:
        """串行检索。"""
        all_chunks = []

        # 原始查询
        chunks = await self._retrieve_by_modes(
            query_expansion.original_query,
            top_k,
            filters,
            knowledge_bases,
            config.retrieval_modes
        )
        all_chunks.extend(chunks)

        # 扩展查询（如果需要更多结果）
        if len(all_chunks) < top_k and query_expansion.expanded_queries:
            for expanded_query in query_expansion.expanded_queries[:2]:
                additional_chunks = await self._retrieve_by_modes(
                    expanded_query,
                    top_k - len(all_chunks),
                    filters,
                    knowledge_bases,
                    config.retrieval_modes
                )
                all_chunks.extend(additional_chunks)

                if len(all_chunks) >= top_k:
                    break

        return all_chunks

    async def _retrieve_by_modes(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        knowledge_bases: Optional[List[str]],
        modes: List[RetrievalMode]
    ) -> List[DocumentChunk]:
        """按模式检索。"""
        chunks_by_mode = {}

        for mode in modes:
            try:
                if mode == RetrievalMode.VECTOR:
                    chunks = await self._vector_retrieve(query, top_k, filters, knowledge_bases)
                elif mode == RetrievalMode.HYBRID:
                    chunks = await self._hybrid_retrieve(query, top_k, filters, knowledge_bases)
                elif mode == RetrievalMode.SEMANTIC:
                    chunks = await self._semantic_retrieve(query, top_k, filters, knowledge_bases)
                elif mode == RetrievalMode.FULLTEXT:
                    chunks = await self._fulltext_retrieve(query, top_k, filters, knowledge_bases)
                else:
                    continue

                chunks_by_mode[mode] = chunks

            except Exception as e:
                self.logger.error(f"检索模式 {mode.value} 失败: {e}")

        # 融合不同模式的结果
        return self._fuse_mode_results(chunks_by_mode)

    async def _vector_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        knowledge_bases: Optional[List[str]]
    ) -> List[DocumentChunk]:
        """向量检索。"""
        if not self.vector_store or not self.embedding_service:
            return []

        try:
            # 生成查询向量
            query_vector = await self.embedding_service.embed_single(query)

            # 执行向量搜索
            all_results = []
            for kb_id in (knowledge_bases or []):
                try:
                    collection_name = f"kb_{kb_id}"
                    search_results = await self.vector_store.search(
                        query_vector=query_vector,
                        top_k=top_k,
                        filters=filters,
                        collection_name=collection_name
                    )

                    for doc_id, score, metadata in search_results:
                        chunk = DocumentChunk(
                            chunk_id=doc_id,
                            content=metadata.get("content", ""),
                            document_id=metadata.get("document_id", ""),
                            chunk_index=metadata.get("chunk_index", 0),
                            title=metadata.get("title", ""),
                            score=score,
                            metadata=metadata
                        )
                        all_results.append(chunk)

                except Exception as e:
                    self.logger.warning(f"知识库 {kb_id} 向量检索失败: {e}")

            # 按分数排序
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:top_k]

        except Exception as e:
            self.logger.error(f"向量检索失败: {e}")
            return []

    async def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        knowledge_bases: Optional[List[str]]
    ) -> List[DocumentChunk]:
        """混合检索。"""
        if not self.vector_store or not self.embedding_service:
            return await self._vector_retrieve(query, top_k, filters, knowledge_bases)

        try:
            # 生成查询向量
            query_vector = await self.embedding_service.embed_single(query)

            all_results = []
            for kb_id in (knowledge_bases or []):
                try:
                    collection_name = f"kb_{kb_id}"
                    search_results = await self.vector_store.hybrid_search(
                        query_text=query,
                        query_vector=query_vector,
                        top_k=top_k,
                        filters=filters,
                        collection_name=collection_name
                    )

                    for doc_id, score, metadata in search_results:
                        chunk = DocumentChunk(
                            chunk_id=doc_id,
                            content=metadata.get("content", ""),
                            document_id=metadata.get("document_id", ""),
                            chunk_index=metadata.get("chunk_index", 0),
                            title=metadata.get("title", ""),
                            score=score,
                            metadata=metadata
                        )
                        all_results.append(chunk)

                except Exception as e:
                    self.logger.warning(f"知识库 {kb_id} 混合检索失败: {e}")

            # 按分数排序
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:top_k]

        except Exception as e:
            self.logger.error(f"混合检索失败: {e}")
            return []

    async def _semantic_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        knowledge_bases: Optional[List[str]]
    ) -> List[DocumentChunk]:
        """语义检索。"""
        # 语义检索可以基于向量检索，加上语义理解的增强
        return await self._vector_retrieve(query, top_k, filters, knowledge_bases)

    async def _fulltext_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        knowledge_bases: Optional[List[str]]
    ) -> List[DocumentChunk]:
        """全文检索。"""
        # 全文检索需要集成搜索引擎，这里简化实现
        if not self.knowledge_manager:
            return []

        try:
            all_chunks = []
            for kb_id in (knowledge_bases or []):
                chunks = await self.knowledge_manager.search_documents(
                    kb_id=kb_id,
                    query=query,
                    top_k=top_k,
                    filters=filters
                )
                all_chunks.extend(chunks)

            return all_chunks[:top_k]

        except Exception as e:
            self.logger.error(f"全文检索失败: {e}")
            return []

    def _fuse_mode_results(self, chunks_by_mode: Dict[RetrievalMode, List[DocumentChunk]]) -> List[DocumentChunk]:
        """融合不同模式的检索结果。"""
        if not chunks_by_mode:
            return []

        # 简单的分数融合策略
        all_chunks = []
        mode_weights = self.default_config.mode_weights

        for mode, chunks in chunks_by_mode.items():
            mode_weight = mode_weights.get(mode.value, 1.0)
            for chunk in chunks:
                # 调整分数
                chunk.score *= mode_weight
                # 添加模式标记
                chunk.metadata["retrieval_mode"] = mode.value
                all_chunks.append(chunk)

        # 去重（基于document_id和chunk_index）
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            key = (chunk.document_id, chunk.chunk_index)
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)

        # 按分数排序
        unique_chunks.sort(key=lambda x: x.score, reverse=True)
        return unique_chunks

    async def _postprocess_results(
        self,
        chunks: List[DocumentChunk],
        original_query: str,
        config: RetrievalConfig
    ) -> List[DocumentChunk]:
        """后处理检索结果。"""
        if not chunks:
            return []

        # 过滤低分数结果
        if config.enable_filtering:
            chunks = [c for c in chunks if c.score >= config.similarity_threshold]

        # 去重
        if config.enable_deduplication:
            chunks = self._deduplicate_chunks(chunks)

        # 重排（如果启用）
        if config.enable_reranking:
            chunks = await self._rerank_chunks(chunks, original_query)

        return chunks

    def _deduplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """去重文档块。"""
        seen_content = set()
        unique_chunks = []

        for chunk in chunks:
            # 使用内容的哈希作为去重依据
            content_hash = hash(chunk.content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)

        return unique_chunks

    async def _rerank_chunks(self, chunks: List[DocumentChunk], query: str) -> List[DocumentChunk]:
        """重排文档块。"""
        # 简单的重排策略：基于分数和内容长度
        for chunk in chunks:
            # 根据内容长度调整分数
            length_bonus = min(len(chunk.content) / 1000, 1.0) * 0.1
            chunk.score += length_bonus

        # 重新排序
        chunks.sort(key=lambda x: x.score, reverse=True)
        return chunks

    # 查询处理器方法
    async def _basic_query_preprocessing(self, query: str) -> str:
        """基础查询预处理。"""
        # 转小写
        processed = query.lower().strip()

        # 移除特殊字符
        processed = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', processed)

        # 合并多个空格
        processed = re.sub(r'\s+', ' ', processed)

        return processed.strip()

    async def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词。"""
        # 简单的关键词提取
        words = query.split()
        # 过滤停用词（简化实现）
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords

    async def _expand_query(self, query: str) -> QueryExpansion:
        """查询扩展。"""
        # 简单的查询扩展：添加同义词和相关词
        keywords = await self._extract_keywords(query)
        expanded_queries = [query]

        # 为关键词添加简单的扩展（实际应用中可以使用同义词词典）
        for keyword in keywords[:3]:  # 限制扩展数量
            if keyword == "人工智能":
                expanded_queries.append(query.replace(keyword, "AI"))
            elif keyword == "机器学习":
                expanded_queries.append(query.replace(keyword, "ML"))
            elif keyword == "深度学习":
                expanded_queries.append(query.replace(keyword, "DL"))

        return QueryExpansion(
            original_query=query,
            expanded_queries=expanded_queries[1:],  # 排除原始查询
            expansion_method="synonym"
        )

    async def _generate_hyde_documents(self, query: str) -> List[str]:
        """生成HyDE文档。"""
        # 简化的HyDE实现：生成假设性文档
        # 实际应用中应该使用LLM生成
        hyde_docs = []

        # 基于查询生成假设性答案
        if "如何" in query or "怎么" in query:
            hyde_docs.append(f"要{query}，需要采取以下步骤：首先，明确目标；其次，制定计划；然后，逐步执行；最后，评估结果。")
        elif "什么是" in query:
            hyde_docs.append(f"{query.split('什么是')[-1].strip()}是一个重要的概念，它具有以下特点：1. 基本定义；2. 主要特征；3. 应用场景；4. 相关技术。")
        else:
            hyde_docs.append(f"关于{query}的详细信息包括：背景介绍、核心技术、实际应用、发展趋势等。")

        return hyde_docs

    async def _decompose_query(self, query: str) -> List[str]:
        """查询分解。"""
        # 简单的查询分解：按逗号、分号等分割
        delimiters = [',', ';', '，', '；', '和', '与', '以及']
        decomposed_queries = [query]

        for delimiter in delimiters:
            if delimiter in query:
                parts = query.split(delimiter)
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 3:
                        decomposed_queries.append(part)
                break

        return decomposed_queries[1:]  # 排除原始查询

    def _get_cache_key(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        knowledge_bases: Optional[List[str]]
    ) -> str:
        """生成缓存键。"""
        key_parts = [query, str(top_k)]
        if filters:
            key_parts.append(str(sorted(filters.items())))
        if knowledge_bases:
            key_parts.append(str(sorted(knowledge_bases)))
        return hash(":".join(key_parts))

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        knowledge_base_id: str
    ) -> List[str]:
        """添加文档到知识库。"""
        if not self.knowledge_manager:
            raise RetrievalException("知识管理服务未初始化")

        document_ids = []
        for doc in documents:
            doc_id = await self.knowledge_manager.add_document(
                kb_id=knowledge_base_id,
                title=doc.get("title", ""),
                content=doc.get("content", ""),
                file_path=doc.get("file_path"),
                file_type=doc.get("file_type"),
                metadata=doc.get("metadata", {})
            )
            document_ids.append(doc_id)

        return document_ids

    async def delete_documents(
        self,
        document_ids: List[str],
        knowledge_base_id: str
    ) -> bool:
        """从知识库删除文档。"""
        # 这里需要实现文档删除逻辑
        # 包括从向量存储和数据库中删除
        self.logger.info(f"删除文档功能待实现: {document_ids}")
        return False

    async def get_statistics(self) -> Dict[str, Any]:
        """获取检索统计信息。"""
        return {
            **self.stats,
            "cache_size": len(self.retrieval_cache),
            "supported_modes": [mode.value for mode in RetrievalMode],
            "available_processors": list(self.query_processors.keys())
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查。"""
        health_status = {
            "status": "healthy",
            "vector_store": False,
            "embedding_service": False,
            "knowledge_manager": False,
            "cache_enabled": len(self.retrieval_cache) > 0,
            "errors": []
        }

        # 检查依赖服务
        try:
            if self.vector_store:
                vs_health = await self.vector_store.health_check()
                health_status["vector_store"] = vs_health.get("milvus", False) or vs_health.get("elasticsearch", False)
        except Exception as e:
            health_status["errors"].append(f"Vector store: {str(e)}")

        try:
            if self.embedding_service:
                embed_health = await self.embedding_service.health_check()
                health_status["embedding_service"] = embed_health.get("status") == "healthy"
        except Exception as e:
            health_status["errors"].append(f"Embedding service: {str(e)}")

        try:
            if self.knowledge_manager:
                km_health = await self.knowledge_manager.health_check()
                health_status["knowledge_manager"] = km_health.get("status") == "healthy"
        except Exception as e:
            health_status["errors"].append(f"Knowledge manager: {str(e)}")

        # 总体状态
        if health_status["errors"]:
            health_status["status"] = "degraded"

        return health_status