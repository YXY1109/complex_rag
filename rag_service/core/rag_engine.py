"""
核心RAG引擎

整合所有RAG组件，提供统一的检索增强生成服务。
基于RAGFlow架构设计，支持复杂的RAG工作流程。
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import logging
from dataclasses import dataclass, field

from ..interfaces.rag_interface import (
    RAGInterface, RAGConfig, RAGQuery, RAGResult, RetrievalResult, GenerationResult,
    RetrievalMode, GenerationMode, ContextStrategy, RAGException
)
from ..services.retrieval_engine import RetrievalEngine
from ..services.context_builder import ContextBuilder
from ..services.generation_service import GenerationService
from ..services.document_ranker import DocumentRanker
from ..services.knowledge_manager import KnowledgeManager


@dataclass
class RAGEngineConfig:
    """RAG引擎配置。"""

    # 组件配置
    retrieval_config: Dict[str, Any] = field(default_factory=dict)
    context_config: Dict[str, Any] = field(default_factory=dict)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    rerank_config: Dict[str, Any] = field(default_factory=dict)

    # 工作流配置
    enable_reranking: bool = True
    enable_context_optimization: bool = True
    enable_query_processing: bool = True
    enable_result_caching: bool = True

    # 性能配置
    parallel_processing: bool = True
    timeout_seconds: int = 60
    max_concurrent_queries: int = 10

    # 质量控制
    min_confidence_threshold: float = 0.5
    max_context_length: int = 8000
    enable_quality_check: bool = True


class RAGEngine(RAGInterface):
    """核心RAG引擎。"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化RAG引擎。

        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 引擎配置
        self.engine_config = RAGEngineConfig(**config.get("rag_engine", {}))

        # 组件实例
        self.retrieval_engine: Optional[RetrievalEngine] = None
        self.context_builder: Optional[ContextBuilder] = None
        self.generation_service: Optional[GenerationService] = None
        self.document_ranker: Optional[DocumentRanker] = None
        self.knowledge_manager: Optional[KnowledgeManager] = None

        # 缓存
        self.result_cache: Dict[str, RAGResult] = {}
        self.cache_ttl = config.get("cache_ttl", 3600)

        # 查询处理器
        self.query_processors: Dict[str, callable] = {}
        self._init_query_processors()

        # 结果后处理器
        self.result_processors: List[callable] = []
        self._init_result_processors()

        # 统计信息
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_query_time": 0.0,
            "cache_hit_rate": 0.0,
            "component_usage": {},
            "query_types": {}
        }

    def _init_query_processors(self) -> None:
        """初始化查询处理器。"""
        self.query_processors = {
            "preprocessing": self._preprocess_query,
            "expansion": self._expand_query,
            "decomposition": self._decompose_query,
            "hyde": self._generate_hyde_query
        }

    def _init_result_processors(self) -> None:
        """初始化结果后处理器。"""
        self.result_processors = [
            self._validate_result_quality,
            self._enhance_result_metadata,
            self._format_result_output
        ]

    async def initialize(
        self,
        retrieval_engine: RetrievalEngine,
        context_builder: ContextBuilder,
        generation_service: GenerationService,
        document_ranker: DocumentRanker,
        knowledge_manager: KnowledgeManager
    ) -> bool:
        """
        初始化RAG引擎。

        Args:
            retrieval_engine: 检索引擎
            context_builder: 上下文构建器
            generation_service: 生成服务
            document_ranker: 文档重排服务
            knowledge_manager: 知识管理服务

        Returns:
            bool: 初始化是否成功
        """
        try:
            self.retrieval_engine = retrieval_engine
            self.context_builder = context_builder
            self.generation_service = generation_service
            self.document_ranker = document_ranker
            self.knowledge_manager = knowledge_manager

            self.logger.info("RAG引擎初始化成功")
            return True

        except Exception as e:
            self.logger.error(f"RAG引擎初始化失败: {e}")
            return False

    async def cleanup(self) -> None:
        """清理RAG引擎资源。"""
        try:
            self.result_cache.clear()
            self.query_processors.clear()
            self.result_processors.clear()
            self.logger.info("RAG引擎资源清理完成")

        except Exception as e:
            self.logger.error(f"RAG引擎清理失败: {e}")

    async def initialize_config(self, config: RAGConfig) -> bool:
        """
        初始化RAG配置。

        Args:
            config: RAG配置

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 验证配置
            validation_errors = self._validate_config(config)
            if validation_errors:
                self.logger.error(f"RAG配置验证失败: {validation_errors}")
                return False

            # 更新组件配置
            if self.retrieval_engine and self.engine_config.retrieval_config:
                # 这里可以配置检索引擎参数
                pass

            if self.generation_service and self.engine_config.generation_config:
                # 这里可以配置生成服务参数
                pass

            self.logger.info("RAG配置初始化成功")
            return True

        except Exception as e:
            self.logger.error(f"RAG配置初始化失败: {e}")
            return False

    async def query(self, query: RAGQuery) -> RAGResult:
        """
        执行RAG查询。

        Args:
            query: RAG查询

        Returns:
            RAGResult: 查询结果
        """
        start_time = datetime.now()
        self.stats["total_queries"] += 1

        try:
            # 检查缓存
            if self.engine_config.enable_result_caching:
                cache_key = self._get_cache_key(query)
                if cache_key in self.result_cache:
                    cached_result = self.result_cache[cache_key]
                    self.logger.info(f"RAG查询缓存命中: {query.query[:50]}...")
                    return cached_result

            # 更新统计
            query_type = self._classify_query(query)
            if query_type not in self.stats["query_types"]:
                self.stats["query_types"][query_type] = 0
            self.stats["query_types"][query_type] += 1

            # 执行RAG流程
            if self.engine_config.parallel_processing:
                result = await self._execute_parallel_rag(query)
            else:
                result = await self._execute_sequential_rag(query)

            # 后处理结果
            for processor in self.result_processors:
                try:
                    result = await processor(result, query)
                except Exception as e:
                    self.logger.error(f"结果后处理器失败: {e}")

            # 缓存结果
            if self.engine_config.enable_result_caching:
                self.result_cache[cache_key] = result

            # 更新统计
            query_time = (datetime.now() - start_time).total_seconds()
            self.stats["successful_queries"] += 1
            self.stats["average_query_time"] = (
                (self.stats["average_query_time"] * (self.stats["total_queries"] - 1) + query_time) /
                self.stats["total_queries"]
            )

            # 更新组件使用统计
            self._update_component_usage(query)

            self.logger.info(f"RAG查询完成，耗时: {query_time:.3f}s，类型: {query_type}")
            return result

        except Exception as e:
            self.stats["failed_queries"] += 1
            self.logger.error(f"RAG查询失败: {e}")
            raise RAGException(f"RAG查询失败: {str(e)}")

    async def _execute_sequential_rag(self, query: RAGQuery) -> RAGResult:
        """顺序执行RAG流程。"""
        # 1. 查询预处理
        processed_query = query
        if self.engine_config.enable_query_processing:
            processed_query = await self._process_query(query)

        # 2. 检索
        retrieval_result = await self._execute_retrieval(processed_query)

        # 3. 文档重排
        if self.engine_config.enable_reranking and self.document_ranker:
            reranked_chunks = await self.document_ranker.rerank(
                query=processed_query.query,
                documents=retrieval_result.chunks,
                top_k=processed_query.top_k or 10
            )
            retrieval_result.chunks = reranked_chunks

        # 4. 上下文构建
        context_result = await self._execute_context_building(
            retrieval_result, processed_query
        )

        # 5. 生成回答
        generation_result = await self._execute_generation(
            processed_query, context_result
        )

        # 6. 创建最终结果
        result = RAGResult(
            query_id=processed_query.query_id,
            query=processed_query.query,
            retrieval_result=retrieval_result,
            generation_result=generation_result,
            total_time=(datetime.now() - processed_query.created_at).total_seconds(),
            success=True,
            metadata={
                "execution_mode": "sequential",
                "processed_query": processed_query.query,
                "retrieval_time": retrieval_result.search_time,
                "generation_time": generation_result.generation_time
            }
        )

        return result

    async def _execute_parallel_rag(self, query: RAGQuery) -> RAGResult:
        """并行执行RAG流程。"""
        # 1. 查询预处理（必须串行）
        processed_query = query
        if self.engine_config.enable_query_processing:
            processed_query = await self._process_query(query)

        # 2. 并行执行检索和上下文准备
        retrieval_task = asyncio.create_task(self._execute_retrieval(processed_query))

        # 3. 等待检索完成
        retrieval_result = await retrieval_task

        # 4. 文档重排（如果启用）
        if self.engine_config.enable_reranking and self.document_ranker:
            rerank_task = asyncio.create_task(
                self.document_ranker.rerank(
                    query=processed_query.query,
                    documents=retrieval_result.chunks,
                    top_k=processed_query.top_k or 10
                )
            )
            reranked_chunks = await rerank_task
            retrieval_result.chunks = reranked_chunks

        # 5. 上下文构建
        context_task = asyncio.create_task(
            self._execute_context_building(retrieval_result, processed_query)
        )
        context_result = await context_task

        # 6. 生成回答
        generation_result = await self._execute_generation(
            processed_query, context_result
        )

        # 7. 创建最终结果
        result = RAGResult(
            query_id=processed_query.query_id,
            query=processed_query.query,
            retrieval_result=retrieval_result,
            generation_result=generation_result,
            total_time=(datetime.now() - processed_query.created_at).total_seconds(),
            success=True,
            metadata={
                "execution_mode": "parallel",
                "processed_query": processed_query.query,
                "retrieval_time": retrieval_result.search_time,
                "generation_time": generation_result.generation_time
            }
        )

        return result

    async def _execute_retrieval(self, query: RAGQuery) -> RetrievalResult:
        """执行检索。"""
        if not self.retrieval_engine:
            raise RAGException("检索引擎未初始化")

        retrieval_mode = query.retrieval_mode or RetrievalMode.HYBRID
        top_k = query.top_k or 10

        return await self.retrieval_engine.retrieve(
            query=query.query,
            top_k=top_k,
            filters=query.filters,
            knowledge_bases=query.knowledge_bases
        )

    async def _execute_context_building(
        self,
        retrieval_result: RetrievalResult,
        query: RAGQuery
    ) -> Any:
        """执行上下文构建。"""
        if not self.context_builder:
            # 简单的上下文构建
            context_text = "\n\n".join([chunk.content for chunk in retrieval_result.chunks])
            return {
                "context_chunks": retrieval_result.chunks,
                "formatted_context": context_text,
                "context_length": len(context_text)
            }

        context_strategy = query.context_strategy or ContextStrategy.STUFFING
        return await self.context_builder.build_context(
            chunks=retrieval_result.chunks,
            query=query.query,
            config=None,  # 使用默认配置
            rag_query=query
        )

    async def _execute_generation(
        self,
        query: RAGQuery,
        context_result: Any
    ) -> GenerationResult:
        """执行生成。"""
        if not self.generation_service:
            raise RAGException("生成服务未初始化")

        context_text = context_result.formatted_context if hasattr(context_result, 'formatted_context') else str(context_result)

        return await self.generation_service.generate(
            query=query.query,
            context=context_text,
            conversation_history=query.conversation_history,
            rag_query=query
        )

    async def _process_query(self, query: RAGQuery) -> RAGQuery:
        """处理查询。"""
        processed_query = query

        for processor_name, processor_func in self.query_processors.items():
            try:
                processed_query = await processor_func(processed_query)
            except Exception as e:
                self.logger.error(f"查询处理器 {processor_name} 失败: {e}")

        return processed_query

    async def _preprocess_query(self, query: RAGQuery) -> RAGQuery:
        """查询预处理。"""
        # 基础清理
        cleaned_query = query.query.strip()

        # 移除多余空格
        cleaned_query = " ".join(cleaned_query.split())

        return RAGQuery(
            query_id=query.query_id,
            query=cleaned_query,
            context=query.context,
            conversation_history=query.conversation_history,
            user_id=query.user_id,
            tenant_id=query.tenant_id,
            session_id=query.session_id,
            retrieval_mode=query.retrieval_mode,
            top_k=query.top_k,
            filters=query.filters,
            include_metadata=query.include_metadata,
            include_scores=query.include_scores,
            generation_mode=query.generation_mode,
            temperature=query.temperature,
            max_tokens=query.max_tokens,
            system_prompt=query.system_prompt,
            user_prompt_template=query.user_prompt_template,
            created_at=query.created_at,
            metadata={**query.metadata, "preprocessed": True}
        )

    async def _expand_query(self, query: RAGQuery) -> RAGQuery:
        """查询扩展。"""
        # 简化实现：添加同义词
        expansions = []
        original_query = query.query.lower()

        # 简单的同义词扩展
        synonym_map = {
            "人工智能": ["AI", "机器学习", "深度学习"],
            "机器学习": ["AI", "人工智能", "ML"],
            "深度学习": ["AI", "人工智能", "DL", "神经网络"],
            "如何": ["怎么", "怎样", "方式"],
            "什么": ["定义", "含义", "解释"]
        }

        for term, synonyms in synonym_map.items():
            if term in original_query:
                expansions.extend(synonyms)

        if expansions:
            expanded_query = query.query + " " + " ".join(expansions[:3])  # 限制扩展数量
            metadata = {**query.metadata, "expanded": True, "expansions": expansions}
        else:
            expanded_query = query.query
            metadata = query.metadata

        return RAGQuery(
            query_id=query.query_id,
            query=expanded_query,
            context=query.context,
            conversation_history=query.conversation_history,
            user_id=query.user_id,
            tenant_id=query.tenant_id,
            session_id=query.session_id,
            retrieval_mode=query.retrieval_mode,
            top_k=query.top_k,
            filters=query.filters,
            include_metadata=query.include_metadata,
            include_scores=query.include_scores,
            generation_mode=query.generation_mode,
            temperature=query.temperature,
            max_tokens=query.max_tokens,
            system_prompt=query.system_prompt,
            user_prompt_template=query.user_prompt_template,
            created_at=query.created_at,
            metadata=metadata
        )

    async def _decompose_query(self, query: RAGQuery) -> RAGQuery:
        """查询分解。"""
        # 简化实现：按逗号分号分割
        delimiters = [',', ';', '，', '；', '和', '与']
        parts = [query.query]

        for delimiter in delimiters:
            if delimiter in query.query:
                parts = query.query.split(delimiter)
                break

        if len(parts) > 1:
            # 使用第一部分作为主查询
            main_query = parts[0].strip()
            metadata = {**query.metadata, "decomposed": True, "parts": parts}
        else:
            main_query = query.query
            metadata = query.metadata

        return RAGQuery(
            query_id=query.query_id,
            query=main_query,
            context=query.context,
            conversation_history=query.conversation_history,
            user_id=query.user_id,
            tenant_id=query.tenant_id,
            session_id=query.session_id,
            retrieval_mode=query.retrieval_mode,
            top_k=query.top_k,
            filters=query.filters,
            include_metadata=query.include_metadata,
            include_scores=query.include_scores,
            generation_mode=query.generation_mode,
            temperature=query.temperature,
            max_tokens=query.max_tokens,
            system_prompt=query.system_prompt,
            user_prompt_template=query.user_prompt_template,
            created_at=query.created_at,
            metadata=metadata
        )

    async def _generate_hyde_query(self, query: RAGQuery) -> RAGQuery:
        """生成HyDE查询。"""
        # 简化的HyDE实现：生成假设性文档
        hypothetical_doc = f"关于{query.query}的详细信息包括：背景介绍、核心概念、应用场景、相关技术、发展趋势等。"

        metadata = {**query.metadata, "hyde_generated": True}

        return RAGQuery(
            query_id=query.query_id,
            query=hypothetical_doc,
            context=query.context,
            conversation_history=query.conversation_history,
            user_id=query.user_id,
            tenant_id=query.tenant_id,
            session_id=query.session_id,
            retrieval_mode=query.retrieval_mode,
            top_k=query.top_k,
            filters=query.filters,
            include_metadata=query.include_metadata,
            include_scores=query.include_scores,
            generation_mode=query.generation_mode,
            temperature=query.temperature,
            max_tokens=query.max_tokens,
            system_prompt=query.system_prompt,
            user_prompt_template=query.user_prompt_template,
            created_at=query.created_at,
            metadata=metadata
        )

    async def _validate_result_quality(self, result: RAGResult, query: RAGQuery) -> RAGResult:
        """验证结果质量。"""
        if not self.engine_config.enable_quality_check:
            return result

        quality_score = 0.0
        quality_factors = []

        # 检查检索质量
        if result.retrieval_result:
            retrieval_score = sum(chunk.score for chunk in result.retrieval_result.chunks) / len(result.retrieval_result.chunks)
            quality_factors.append(retrieval_score)
            quality_score += retrieval_score * 0.4

        # 检查生成质量
        if result.generation_result:
            # 简化的生成质量评估
            answer_length = len(result.generation_result.answer)
            length_score = min(answer_length / 100, 1.0)  # 理想长度100字符
            quality_factors.append(length_score)
            quality_score += length_score * 0.3

            # 检查token使用合理性
            if result.generation_result.token_usage:
                token_efficiency = result.generation_result.token_usage.get("completion_tokens", 0) / max(result.generation_result.token_usage.get("total_tokens", 1), 1)
                quality_factors.append(token_efficiency)
                quality_score += token_efficiency * 0.3

        # 添加质量信息到元数据
        result.metadata.update({
            "quality_score": quality_score,
            "quality_factors": quality_factors,
            "meets_threshold": quality_score >= self.engine_config.min_confidence_threshold
        })

        # 如果质量低于阈值，标记结果
        if quality_score < self.engine_config.min_confidence_threshold:
            result.success = False
            result.metadata["quality_issue"] = "Low confidence score"

        return result

    async def _enhance_result_metadata(self, result: RAGResult, query: RAGQuery) -> RAGResult:
        """增强结果元数据。"""
        # 添加执行时间信息
        result.metadata.update({
            "query_timestamp": query.created_at.isoformat(),
            "response_timestamp": datetime.now().isoformat(),
            "component_times": {
                "retrieval": result.retrieval_result.search_time,
                "generation": result.generation_result.generation_time
            }
        })

        # 添加来源信息
        if result.retrieval_result and result.retrieval_result.chunks:
            sources = []
            for chunk in result.retrieval_result.chunks:
                source_info = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "score": chunk.score,
                    "metadata": chunk.metadata
                }
                sources.append(source_info)
            result.metadata["sources"] = sources

        # 添加性能指标
        if result.retrieval_result and result.generation_result:
            result.metadata["performance"] = {
                "total_time": result.total_time,
                "retrieval_time": result.retrieval_result.search_time,
                "generation_time": result.generation_result.generation_time,
                "retrieval_efficiency": len(result.retrieval_result.chunks) / result.retrieval_result.total_found if result.retrieval_result.total_found > 0 else 1.0
            }

        return result

    async def _format_result_output(self, result: RAGResult, query: RAGQuery) -> RAGResult:
        """格式化结果输出。"""
        # 这里可以根据需要格式化输出
        # 例如添加引用格式、结构化输出等
        return result

    def _classify_query(self, query: RAGQuery) -> str:
        """分类查询类型。"""
        query_text = query.query.lower()

        if any(keyword in query_text for keyword in ["如何", "怎么", "怎样", "方式", "方法"]):
            return "how_to"
        elif any(keyword in query_text for keyword in ["什么", "定义", "解释", "含义"]):
            return "what_is"
        elif any(keyword in query_text for keyword in ["为什么", "原因", "理由"]):
            return "why"
        elif any(keyword in query_text for keyword in ["比较", "对比", "区别", "差异"]):
            return "comparison"
        elif any(keyword in query_text for keyword in ["列表", "包含", "有哪些", "总结"]):
            return "list"
        else:
            return "general"

    def _update_component_usage(self, query: RAGQuery) -> None:
        """更新组件使用统计。"""
        components = []

        if query.retrieval_mode:
            components.append(f"retrieval_{query.retrieval_mode.value}")

        if query.generation_mode:
            components.append(f"generation_{query.generation_mode.value}")

        if query.context_strategy:
            components.append(f"context_{query.context_strategy.value}")

        for component in components:
            if component not in self.stats["component_usage"]:
                self.stats["component_usage"][component] = 0
            self.stats["component_usage"][component] += 1

    def _get_cache_key(self, query: RAGQuery) -> str:
        """生成缓存键。"""
        key_parts = [
            query.query,
            str(query.retrieval_mode.value) if query.retrieval_mode else "",
            str(query.generation_mode.value) if query.generation_mode else "",
            str(query.top_k or 10),
            str(sorted(query.filters.items())) if query.filters else ""
        ]
        return hash(":".join(key_parts))

    def _validate_config(self, config: RAGConfig) -> List[str]:
        """验证RAG配置。"""
        errors = []

        if config.top_k <= 0:
            errors.append("top_k 必须大于 0")

        if config.similarity_threshold < 0 or config.similarity_threshold > 1:
            errors.append("similarity_threshold 必须在 0-1 之间")

        if config.max_tokens <= 0:
            errors.append("max_tokens 必须大于 0")

        if config.temperature < 0 or config.temperature > 2:
            errors.append("temperature 必须在 0-2 之间")

        return errors

    async def query_stream(
        self,
        query: RAGQuery
    ) -> AsyncGenerator[str, None]:
        """
        流式RAG查询。

        Args:
            query: RAG查询

        Yields:
            str: 流式输出内容
        """
        try:
            # 检查缓存
            if self.engine_config.enable_result_caching:
                cache_key = self._get_cache_key(query)
                if cache_key in self.result_cache:
                    cached_result = self.result_cache[cache_key]
                    yield cached_result.answer
                    return

            # 执行RAG流程（简化版本，只到最后生成阶段流式输出）
            processed_query = query
            if self.engine_config.enable_query_processing:
                processed_query = await self._process_query(query)

            # 检索和上下文构建
            retrieval_result = await self._execute_retrieval(processed_query)

            if self.engine_config.enable_reranking and self.document_ranker:
                reranked_chunks = await self.document_ranker.rerank(
                    query=processed_query.query,
                    documents=retrieval_result.chunks,
                    top_k=processed_query.top_k or 10
                )
                retrieval_result.chunks = reranked_chunks

            context_result = await self._execute_context_building(
                retrieval_result, processed_query
            )

            # 流式生成
            context_text = context_result.formatted_context if hasattr(context_result, 'formatted_context') else str(context_result)

            async for chunk in self.generation_service.generate_stream(
                query=processed_query.query,
                context=context_text,
                conversation_history=processed_query.conversation_history,
                rag_query=processed_query
            ):
                yield chunk

        except Exception as e:
            self.logger.error(f"流式RAG查询失败: {e}")
            yield f"抱歉，处理过程中出现错误: {str(e)}"

    async def batch_query(
        self,
        queries: List[RAGQuery],
        max_concurrent: Optional[int] = None
    ) -> List[RAGResult]:
        """
        批量RAG查询。

        Args:
            queries: RAG查询列表
            max_concurrent: 最大并发数

        Returns:
            List[RAGResult]: 查询结果列表
        """
        if not queries:
            return []

        max_concurrent = max_concurrent or self.engine_config.max_concurrent_queries

        # 分批处理
        results = []
        for i in range(0, len(queries), max_concurrent):
            batch = queries[i:i + max_concurrent]

            # 并行执行批量查询
            tasks = [self.query(query) for query in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for result in batch_results:
                if isinstance(result, RAGResult):
                    results.append(result)
                else:
                    self.logger.error(f"批量查询中某个任务失败: {result}")
                    # 创建错误结果
                    error_result = RAGResult(
                        query_id="error",
                        query="",
                        retrieval_result=None,
                        generation_result=None,
                        total_time=0.0,
                        success=False,
                        error_message=str(result)
                    )
                    results.append(error_result)

        return results

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息。"""
        success_rate = 0.0
        if self.stats["total_queries"] > 0:
            success_rate = self.stats["successful_queries"] / self.stats["total_queries"]

        cache_hit_rate = 0.0
        if self.engine_config.enable_result_caching and self.stats["total_queries"] > 0:
            cache_hit_rate = self.stats.get("cache_hits", 0) / self.stats["total_queries"]

        return {
            **self.stats,
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.result_cache),
            "engine_config": self.engine_config.__dict__,
            "supported_modes": {
                "retrieval": [mode.value for mode in RetrievalMode],
                "generation": [mode.value for mode in GenerationMode],
                "context": [strategy.value for strategy in ContextStrategy]
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查。"""
        health_status = {
            "status": "healthy",
            "components": {},
            "cache_enabled": self.engine_config.enable_result_caching,
            "cache_size": len(self.result_cache),
            "errors": []
        }

        # 检查组件状态
        components_to_check = {
            "retrieval_engine": self.retrieval_engine,
            "context_builder": self.context_builder,
            "generation_service": self.generation_service,
            "document_ranker": self.document_ranker,
            "knowledge_manager": self.knowledge_manager
        }

        for component_name, component in components_to_check.items():
            try:
                if component and hasattr(component, 'health_check'):
                    component_health = await component.health_check()
                    health_status["components"][component_name] = component_health.get("status") == "healthy"
                else:
                    health_status["components"][component_name] = component is not None
            except Exception as e:
                health_status["errors"].append(f"{component_name}: {str(e)}")
                health_status["components"][component_name] = False

        # 总体状态
        if health_status["errors"] or any(not status for status in health_status["components"].values()):
            health_status["status"] = "degraded"

        return health_status