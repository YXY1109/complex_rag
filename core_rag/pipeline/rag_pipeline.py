"""
RAG处理流水线

整合查询理解、检索、上下文构建和答案生成的完整流水线。
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator

from .interfaces.pipeline_interface import (
    RAGPipelineInterface,
    PipelineResponse,
    PipelineConfig,
    QueryRequest,
    QueryUnderstanding,
    RetrievalResult,
    Context,
    Answer,
    PipelineStage,
    GenerationStrategy,
)
from .stages.query_understanding import QueryUnderstandingEngine
from .stages.context_building import ContextBuilder
from .stages.answer_generation import AnswerGenerator
from ..retriever.factory import RetrieverFactory
from ..retriever.interfaces.retriever_interface import RetrievalQuery, RetrievalStrategy
from ..infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.pipeline.rag_pipeline")


class RAGPipeline(RAGPipelineInterface):
    """
    RAG处理流水线

    整合查询理解、检索、上下文构建和答案生成的完整流程。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化RAG流水线

        Args:
            config: 流水线配置
        """
        self.config = PipelineConfig(**config)

        # 组件实例
        self.query_understanding_engine: Optional[QueryUnderstandingEngine] = None
        self.context_builder: Optional[ContextBuilder] = None
        self.answer_generator: Optional[AnswerGenerator] = None
        self.retriever = None

        # 性能统计
        self.stage_times: Dict[str, List[float]] = {
            stage.value: [] for stage in PipelineStage
        }
        self.total_requests = 0
        self.successful_requests = 0

        # 缓存
        self._response_cache = {} if self.config.enable_caching else None
        self._cache_timestamps = {}

        self._initialized = False

    async def initialize(self) -> bool:
        """
        初始化RAG流水线

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "初始化RAG流水线",
                extra={
                    "enable_caching": self.config.enable_caching,
                    "retrieval_strategies": self.config.retrieval_strategies,
                    "enable_parallel_processing": self.config.enable_parallel_processing,
                }
            )

            # 初始化查询理解引擎
            self.query_understanding_engine = QueryUnderstandingEngine(
                {
                    "enable_query_rewrite": self.config.enable_query_rewrite,
                    "enable_query_expansion": self.config.enable_query_expansion,
                    "enable_intent_detection": self.config.enable_intent_detection,
                    "max_rewrite_suggestions": self.config.max_rewrite_suggestions,
                }
            )
            await self.query_understanding_engine.initialize()

            # 初始化上下文构建器
            self.context_builder = ContextBuilder(
                {
                    "max_context_tokens": self.config.max_context_tokens,
                    "context_window_overlap": self.config.context_window_overlap,
                    "enable_context_compression": self.config.enable_context_compression,
                    "enable_context_ranking": self.config.enable_context_ranking,
                    "building_strategy": "relevance_ranked",
                }
            )
            await self.context_builder.initialize()

            # 初始化答案生成器
            self.answer_generator = AnswerGenerator(
                {
                    "generation_model": self.config.generation_model,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "enable_citation_generation": self.config.enable_citation_generation,
                    "enable_source_attribution": self.config.enable_source_attribution,
                }
            )
            await self.answer_generator.initialize()

            # 初始化检索器
            await self._initialize_retriever()

            self._initialized = True
            structured_logger.info("RAG流水线初始化成功")
            return True

        except Exception as e:
            structured_logger.error(f"RAG流水线初始化失败: {e}")
            return False

    async def _initialize_retriever(self) -> None:
        """初始化检索器"""
        try:
            # 创建多策略检索器配置
            retriever_config = {
                "enable_adaptive": True,
                "enable_caching": self.config.enable_caching,
                "fusion": {
                    "method": "weighted_sum",
                    "top_k": self.config.max_retrieval_results,
                },
                "strategies": {},
            }

            # 配置各种检索策略
            for strategy in self.config.retrieval_strategies:
                if strategy == "vector":
                    retriever_config["strategies"]["vector"] = {
                        "enabled": True,
                        "weight": 0.7,
                        "config": {
                            "top_k": self.config.max_retrieval_results,
                            "min_score": self.config.min_relevance_score,
                        }
                    }
                elif strategy == "bm25":
                    retriever_config["strategies"]["bm25"] = {
                        "enabled": True,
                        "weight": 0.3,
                        "config": {
                            "top_k": self.config.max_retrieval_results,
                            "min_score": self.config.min_relevance_score,
                        }
                    }

            # 创建检索器
            from ..retriever.factory import RetrieverFactory
            self.retriever = RetrieverFactory.create_multi_strategy_retriever(retriever_config)
            await self.retriever.initialize()

            structured_logger.info("检索器初始化成功")

        except Exception as e:
            structured_logger.error(f"检索器初始化失败: {e}")
            raise

    async def process(self, request: QueryRequest) -> PipelineResponse:
        """
        处理RAG查询

        Args:
            request: 查询请求

        Returns:
            PipelineResponse: 处理结果
        """
        if not self._initialized:
            raise RuntimeError("RAG流水线未初始化")

        start_time = time.time()
        stage_times = {}

        try:
            self.total_requests += 1

            structured_logger.info(
                "开始处理RAG查询",
                extra={
                    "query_length": len(request.query),
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "generation_strategy": request.generation_strategy.value,
                }
            )

            # 检查缓存
            cache_key = self._get_cache_key(request)
            if self._response_cache and cache_key in self._response_cache:
                cached_response = self._response_cache[cache_key]
                structured_logger.debug(f"使用缓存结果: {request.query[:50]}...")
                return cached_response

            # 阶段1: 查询理解
            stage_start = time.time()
            query_understanding = await self.understand_query(
                request.query,
                request.context,
                request.conversation_history
            )
            stage_times[PipelineStage.QUERY_UNDERSTANDING.value] = (time.time() - stage_start) * 1000

            # 阶段2: 文档检索
            stage_start = time.time()
            retrieval_results = await self.retrieve_documents(
                query_understanding.processed_query,
                request.filters,
                self.config.max_retrieval_results
            )
            stage_times[PipelineStage.RETRIEVAL.value] = (time.time() - stage_start) * 1000

            # 阶段3: 上下文构建
            stage_start = time.time()
            context = await self.build_context(
                retrieval_results,
                request.max_context_length
            )
            stage_times[PipelineStage.CONTEXT_BUILDING.value] = (time.time() - stage_start) * 1000

            # 阶段4: 答案生成
            stage_start = time.time()
            answer = await self.generate_answer(
                request.query,
                context,
                request.generation_strategy,
                request.enable_citations
            )
            stage_times[PipelineStage.ANSWER_GENERATION.value] = (time.time() - stage_start) * 1000

            # 阶段5: 后处理（如果有）
            stage_start = time.time()
            post_processed_answer = await self._post_process_answer(answer, request)
            stage_times[PipelineStage.POST_PROCESSING.value] = (time.time() - stage_start) * 1000

            total_processing_time = (time.time() - start_time) * 1000

            # 构建响应
            response = PipelineResponse(
                query=request.query,
                answer=post_processed_answer,
                query_understanding=query_understanding,
                retrieval_results=retrieval_results,
                context=context,
                total_processing_time_ms=total_processing_time,
                stage_times=stage_times,
                success=True,
                metadata={
                    "request_id": f"req_{int(time.time())}",
                    "config_used": {
                        "max_context_length": request.max_context_length,
                        "generation_strategy": request.generation_strategy.value,
                    }
                },
                created_at=datetime.utcnow().isoformat(),
            )

            # 缓存结果
            if self._response_cache:
                self._response_cache[cache_key] = response
                self._cache_timestamps[cache_key] = time.time()

            # 更新统计
            self.successful_requests += 1
            self._update_stage_times(stage_times)

            structured_logger.info(
                "RAG查询处理完成",
                extra={
                    "total_processing_time_ms": total_processing_time,
                    "retrieved_documents_count": len(retrieval_results.chunks),
                    "context_documents_count": len(context.documents),
                    "answer_length": len(post_processed_answer.content),
                    "confidence": post_processed_answer.confidence,
                }
            )

            return response

        except Exception as e:
            structured_logger.error(f"RAG查询处理失败: {e}")

            # 返回错误响应
            error_response = PipelineResponse(
                query=request.query,
                answer=Answer(
                    content=f"抱歉，处理您的问题时出现了错误: {str(e)}",
                    confidence=0.0,
                    generation_time_ms=0.0,
                    token_usage={},
                ),
                query_understanding=QueryUnderstanding(
                    original_query=request.query,
                    processed_query=request.query,
                    query_type="unknown",
                    query_intent="处理失败",
                    confidence=0.0,
                    processing_time_ms=0.0,
                ),
                retrieval_results=RetrievalResult(
                    chunks=[],
                    query=request.query,
                    strategy="unknown",
                    total_found=0,
                    search_time_ms=0.0,
                ),
                context=Context(
                    documents=[],
                    formatted_context="",
                    total_length=0,
                    relevance_score=0.0,
                    construction_time_ms=0.0,
                ),
                total_processing_time_ms=(time.time() - start_time) * 1000,
                stage_times=stage_times,
                success=False,
                error_message=str(e),
                created_at=datetime.utcnow().isoformat(),
            )

            return error_response

    async def process_stream(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        """
        流式处理RAG查询

        Args:
            request: 查询请求

        Yields:
            str: 流式生成的答案片段
        """
        if not self._initialized:
            raise RuntimeError("RAG流水线未初始化")

        try:
            structured_logger.info(f"开始流式处理RAG查询: {request.query[:50]}...")

            # 执行前几个阶段（非流式）
            query_understanding = await self.understand_query(
                request.query,
                request.context,
                request.conversation_history
            )

            retrieval_results = await self.retrieve_documents(
                query_understanding.processed_query,
                request.filters,
                self.config.max_retrieval_results
            )

            context = await self.build_context(
                retrieval_results,
                request.max_context_length
            )

            # 流式生成答案
            async for chunk in self.answer_generator.generate_answer_stream(
                request.query,
                context,
                request.generation_strategy,
                request.enable_citations
            ):
                yield chunk

            structured_logger.info("流式RAG查询处理完成")

        except Exception as e:
            structured_logger.error(f"流式RAG查询处理失败: {e}")
            yield f"处理查询时出错: {str(e)}"

    async def understand_query(
        self,
        query: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> QueryUnderstanding:
        """查询理解"""
        if not self.query_understanding_engine:
            raise RuntimeError("查询理解引擎未初始化")

        request = QueryRequest(
            query=query,
            context=context,
            conversation_history=history or []
        )

        return await self.query_understanding_engine.understand_query(request)

    async def retrieve_documents(
        self,
        processed_query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 20
    ) -> RetrievalResult:
        """文档检索"""
        if not self.retriever:
            raise RuntimeError("检索器未初始化")

        retrieval_query = RetrievalQuery(
            text=processed_query,
            top_k=max_results,
            min_score=self.config.min_relevance_score,
            filters=filters or {},
            strategy=RetrievalStrategy.HYBRID,
            mode="multi",
        )

        result = await self.retriever.retrieve_multi_strategy(retrieval_query)

        # 转换为标准格式
        return RetrievalResult(
            chunks=[
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "score": chunk.score,
                    "metadata": chunk.metadata,
                }
                for chunk in result.combined_chunks
            ],
            query=processed_query,
            strategy=result.best_strategy.value if result.best_strategy else "hybrid",
            total_found=len(result.combined_chunks),
            search_time_ms=result.total_processing_time_ms,
            scores=[chunk.score for chunk in result.combined_chunks],
            metadata=result.metadata,
        )

    async def build_context(
        self,
        retrieval_results: RetrievalResult,
        max_length: int = 4000
    ) -> Context:
        """构建上下文"""
        if not self.context_builder:
            raise RuntimeError("上下文构建器未初始化")

        return await self.context_builder.build_context(retrieval_results, max_length)

    async def generate_answer(
        self,
        query: str,
        context: Context,
        generation_strategy: GenerationStrategy = GenerationStrategy.DETAILED,
        enable_citations: bool = True
    ) -> Answer:
        """生成答案"""
        if not self.answer_generator:
            raise RuntimeError("答案生成器未初始化")

        return await self.answer_generator.generate_answer(
            query, context, generation_strategy, enable_citations
        )

    async def _post_process_answer(self, answer: Answer, request: QueryRequest) -> Answer:
        """后处理答案"""
        # 这里可以添加各种后处理逻辑
        processed_answer = answer

        # 根据生成策略调整答案格式
        if request.generation_strategy == GenerationStrategy.CONCISE:
            # 确保简洁性
            if len(processed_answer.content.split()) > 200:
                processed_answer.content = ' '.join(processed_answer.content.split()[:200]) + "..."

        elif request.generation_strategy == GenerationStrategy.STRUCTURED:
            # 确保结构化
            if not any(marker in processed_answer.content for marker in ["#", "*", "1.", "-"]):
                # 添加简单的结构化标记
                processed_answer.content = "## 主要回答\n\n" + processed_answer.content

        return processed_answer

    async def batch_process(self, requests: List[QueryRequest]) -> List[PipelineResponse]:
        """批量处理查询"""
        if not self._initialized:
            raise RuntimeError("RAG流水线未初始化")

        try:
            structured_logger.info(f"开始批量处理RAG查询，数量: {len(requests)}")

            if self.config.enable_parallel_processing:
                # 并行处理
                tasks = [self.process(request) for request in requests]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 处理异常
                valid_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        structured_logger.error(f"查询 {i} 处理失败: {result}")
                        # 创建错误响应
                        valid_results.append(PipelineResponse(
                            query=requests[i].query,
                            answer=Answer(
                                content=f"批量处理时出现错误: {str(result)}",
                                confidence=0.0,
                                generation_time_ms=0.0,
                                token_usage={},
                            ),
                            query_understanding=QueryUnderstanding(
                                original_query=requests[i].query,
                                processed_query=requests[i].query,
                                query_type="unknown",
                                query_intent="处理失败",
                                confidence=0.0,
                                processing_time_ms=0.0,
                            ),
                            retrieval_results=RetrievalResult(
                                chunks=[],
                                query=requests[i].query,
                                strategy="unknown",
                                total_found=0,
                                search_time_ms=0.0,
                            ),
                            context=Context(
                                documents=[],
                                formatted_context="",
                                total_length=0,
                                relevance_score=0.0,
                                construction_time_ms=0.0,
                            ),
                            total_processing_time_ms=0.0,
                            success=False,
                            error_message=str(result),
                            created_at=datetime.utcnow().isoformat(),
                        ))
                    else:
                        valid_results.append(result)
            else:
                # 顺序处理
                valid_results = []
                for request in requests:
                    try:
                        result = await self.process(request)
                        valid_results.append(result)
                    except Exception as e:
                        structured_logger.error(f"查询处理失败: {e}")
                        # 继续处理下一个查询

            structured_logger.info(f"批量RAG查询处理完成，成功处理 {len(valid_results)} 个")
            return valid_results

        except Exception as e:
            structured_logger.error(f"批量RAG查询处理失败: {e}")
            raise Exception(f"Batch RAG processing failed: {e}")

    def _get_cache_key(self, request: QueryRequest) -> str:
        """生成缓存键"""
        key_parts = [
            request.query,
            str(request.generation_strategy),
            str(request.max_context_length),
            str(request.enable_citations),
        ]
        return str(hash("_".join(key_parts)))

    def _update_stage_times(self, stage_times: Dict[str, float]) -> None:
        """更新阶段时间统计"""
        for stage, time_ms in stage_times.items():
            if stage in self.stage_times:
                self.stage_times[stage].append(time_ms)

    async def get_pipeline_statistics(self) -> Dict[str, Any]:
        """获取流水线统计信息"""
        stage_stats = {}
        for stage, times in self.stage_times.items():
            if times:
                stage_stats[stage] = {
                    "avg_time_ms": sum(times) / len(times),
                    "min_time_ms": min(times),
                    "max_time_ms": max(times),
                    "total_requests": len(times),
                }

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "stage_statistics": stage_stats,
            "cache_size": len(self._response_cache) if self._response_cache else 0,
            "config": self.config.dict(),
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查各组件健康状态
            component_health = {}

            if self.query_understanding_engine:
                component_health["query_understanding"] = await self.query_understanding_engine.health_check()

            if self.context_builder:
                component_health["context_builder"] = await self.context_builder.health_check()

            if self.answer_generator:
                component_health["answer_generator"] = await self.answer_generator.health_check()

            if self.retriever:
                component_health["retriever"] = await self.retriever.health_check()

            # 检查整体健康状态
            all_healthy = all(
                health.get("status") == "healthy"
                for health in component_health.values()
            )

            # 执行端到端测试
            test_start = time.time()
            test_request = QueryRequest(query="什么是AI？")
            test_response = await self.process(test_request)
            test_time = (time.time() - test_start) * 1000

            return {
                "status": "healthy" if all_healthy and test_response.success else "degraded",
                "initialized": self._initialized,
                "component_health": component_health,
                "test_processing_time_ms": test_time,
                "test_success": test_response.success,
                "success_rate": self.successful_requests / max(self.total_requests, 1),
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "initialized": self._initialized,
            }

    async def cleanup(self) -> None:
        """清理资源"""
        try:
            # 清理各组件
            if self.query_understanding_engine:
                await self.query_understanding_engine.cleanup()

            if self.context_builder:
                await self.context_builder.cleanup()

            if self.answer_generator:
                await self.answer_generator.cleanup()

            if self.retriever:
                await self.retriever.cleanup()

            # 清理缓存
            if self._response_cache:
                self._response_cache.clear()
            self._cache_timestamps.clear()

            self._initialized = False
            structured_logger.info("RAG流水线清理完成")

        except Exception as e:
            structured_logger.error(f"RAG流水线清理失败: {e}")