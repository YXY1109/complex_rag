"""
多策略检索器

支持多种检索策略的组合、优化和自适应选择。
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import statistics

from .interfaces.retriever_interface import (
    MultiStrategyRetrieverInterface,
    RetrieverInterface,
    RetrievalQuery,
    RetrievalResult,
    MultiStrategyResult,
    DocumentChunk,
    RetrievalStrategy,
    RetrievalMode,
    RetrieverConfig,
)
from .strategies.vector_retriever import VectorRetriever
from .strategies.keyword_retriever import BM25Retriever
from .strategies.graph_retriever import GraphRetriever
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.retriever.multi_strategy_retriever")


@dataclass
class StrategyConfig:
    """策略配置"""
    strategy: RetrievalStrategy
    weight: float = 1.0
    enabled: bool = True
    config: Optional[Dict[str, Any]] = None
    retriever: Optional[RetrieverInterface] = None


@dataclass
class FusionConfig:
    """结果融合配置"""
    method: str = "weighted_sum"  # weighted_sum, rank_fusion, score_fusion, adaptive
    normalize_scores: bool = True
    top_k: int = 10
    diversity_threshold: float = 0.3
    max_results_per_strategy: int = 50


class MultiStrategyRetriever(MultiStrategyRetrieverInterface):
    """
    多策略检索器

    支持多种检索策略的并行执行和结果融合。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化多策略检索器

        Args:
            config: 配置参数
        """
        self.config = config
        self.fusion_config = FusionConfig(**config.get("fusion", {}))

        # 策略配置
        self.strategies: Dict[RetrievalStrategy, StrategyConfig] = {}
        self.retrievers: Dict[RetrievalStrategy, RetrieverInterface] = {}

        # 性能统计
        self.strategy_performance: Dict[RetrievalStrategy, Dict[str, float]] = defaultdict(lambda: {
            "avg_time": 0.0,
            "avg_score": 0.0,
            "success_rate": 1.0,
            "usage_count": 0,
        })

        # 自适应优化
        self.enable_adaptive = config.get("enable_adaptive", True)
        self.performance_history: List[Dict[str, Any]] = []

        # 缓存
        self._result_cache = {} if config.get("enable_caching", True) else None
        self._cache_ttl = config.get("cache_ttl", 3600)

        self._initialized = False

    async def initialize(self) -> bool:
        """
        初始化多策略检索器

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "初始化多策略检索器",
                extra={
                    "fusion_method": self.fusion_config.method,
                    "enable_adaptive": self.enable_adaptive,
                }
            )

            # 初始化各种策略
            await self._initialize_strategies()

            # 并行初始化所有检索器
            init_tasks = [retriever.initialize() for retriever in self.retrievers.values()]
            results = await asyncio.gather(*init_tasks, return_exceptions=True)

            # 检查初始化结果
            success_count = sum(1 for result in results if result is True)
            total_count = len(results)

            if success_count == 0:
                structured_logger.error("所有策略初始化失败")
                return False
            elif success_count < total_count:
                failed_strategies = [
                    strategy.value for strategy, result in zip(self.retrievers.keys(), results)
                    if result is not True
                ]
                structured_logger.warning(f"部分策略初始化失败: {failed_strategies}")

            self._initialized = True
            structured_logger.info(
                f"多策略检索器初始化完成，成功: {success_count}/{total_count}"
            )
            return True

        except Exception as e:
            structured_logger.error(f"多策略检索器初始化失败: {e}")
            return False

    async def _initialize_strategies(self) -> None:
        """初始化各种检索策略"""
        strategy_configs = self.config.get("strategies", {})

        # 初始化向量检索器
        if "vector" in strategy_configs and strategy_configs["vector"].get("enabled", True):
            vector_config = RetrieverConfig(**strategy_configs["vector"].get("config", {}))
            vector_retriever = VectorRetriever(
                config=vector_config,
                embedding_service=strategy_configs["vector"].get("embedding_service")
            )
            self.strategies[RetrievalStrategy.VECTOR] = StrategyConfig(
                strategy=RetrievalStrategy.VECTOR,
                weight=strategy_configs["vector"].get("weight", 1.0),
                enabled=True,
                config=strategy_configs["vector"].get("config", {}),
                retriever=vector_retriever
            )
            self.retrievers[RetrievalStrategy.VECTOR] = vector_retriever

        # 初始化BM25检索器
        if "keyword" in strategy_configs and strategy_configs["keyword"].get("enabled", True):
            keyword_config = RetrieverConfig(**strategy_configs["keyword"].get("config", {}))
            keyword_retriever = BM25Retriever(config=keyword_config)
            self.strategies[RetrievalStrategy.BM25] = StrategyConfig(
                strategy=RetrievalStrategy.BM25,
                weight=strategy_configs["keyword"].get("weight", 1.0),
                enabled=True,
                config=strategy_configs["keyword"].get("config", {}),
                retriever=keyword_retriever
            )
            self.retrievers[RetrievalStrategy.BM25] = keyword_retriever

        # 初始化图检索器
        if "graph" in strategy_configs and strategy_configs["graph"].get("enabled", True):
            graph_config = RetrieverConfig(**strategy_configs["graph"].get("config", {}))
            graph_retriever = GraphRetriever(
                config=graph_config,
                graph_service=strategy_configs["graph"].get("graph_service")
            )
            self.strategies[RetrievalStrategy.GRAPH] = StrategyConfig(
                strategy=RetrievalStrategy.GRAPH,
                weight=strategy_configs["graph"].get("weight", 0.8),
                enabled=True,
                config=strategy_configs["graph"].get("config", {}),
                retriever=graph_retriever
            )
            self.retrievers[RetrievalStrategy.GRAPH] = graph_retriever

        structured_logger.info(f"初始化了 {len(self.retrievers)} 个检索策略")

    async def add_strategy(
        self,
        strategy: RetrievalStrategy,
        retriever: RetrieverInterface,
        weight: float = 1.0
    ) -> bool:
        """
        添加检索策略

        Args:
            strategy: 检索策略
            retriever: 检索器实例
            weight: 策略权重

        Returns:
            bool: 添加是否成功
        """
        try:
            if not self._initialized:
                raise RuntimeError("多策略检索器未初始化")

            # 初始化检索器
            if not await retriever.initialize():
                structured_logger.error(f"策略 {strategy.value} 初始化失败")
                return False

            # 添加策略
            self.strategies[strategy] = StrategyConfig(
                strategy=strategy,
                weight=weight,
                enabled=True,
                retriever=retriever
            )
            self.retrievers[strategy] = retriever

            structured_logger.info(f"成功添加策略: {strategy.value}")
            return True

        except Exception as e:
            structured_logger.error(f"添加策略失败: {e}")
            return False

    async def retrieve_multi_strategy(
        self,
        query: RetrievalQuery
    ) -> MultiStrategyResult:
        """
        多策略检索

        Args:
            query: 检索查询

        Returns:
            MultiStrategyResult: 多策略检索结果
        """
        if not self._initialized:
            raise RuntimeError("多策略检索器未初始化")

        start_time = time.time()

        try:
            # 检查缓存
            cache_key = self._get_cache_key(query)
            if self._result_cache and cache_key in self._result_cache:
                cached_result = self._result_cache[cache_key]
                structured_logger.debug(f"使用缓存的多策略结果: {query.text[:50]}...")
                return cached_result

            structured_logger.info(
                f"开始多策略检索",
                extra={
                    "query_length": len(query.text),
                    "mode": query.mode,
                    "strategies_count": len(self.retrievers),
                }
            )

            # 根据查询模式选择策略
            strategies_to_use = self._select_strategies(query)

            # 执行多策略检索
            if query.mode == RetrievalMode.SINGLE:
                results = await self._single_strategy_retrieval(query, strategies_to_use)
            elif query.mode == RetrievalMode.MULTI:
                results = await self._parallel_strategy_retrieval(query, strategies_to_use)
            elif query.mode == RetrievalMode.SEQUENTIAL:
                results = await self._sequential_strategy_retrieval(query, strategies_to_use)
            else:
                # 默认并行执行
                results = await self._parallel_strategy_retrieval(query, strategies_to_use)

            # 融合结果
            combined_chunks = await self._fuse_results(results, query)

            # 计算策略评分
            strategy_scores = await self._calculate_strategy_scores(results)

            # 选择最佳策略
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0] if strategy_scores else None

            processing_time = (time.time() - start_time) * 1000

            result = MultiStrategyResult(
                query=query.text,
                results=results,
                combined_chunks=combined_chunks,
                strategy_scores=strategy_scores,
                combined_score=sum(strategy_scores.values()),
                total_processing_time_ms=processing_time,
                best_strategy=best_strategy,
                created_at=datetime.utcnow().isoformat(),
            )

            # 缓存结果
            if self._result_cache:
                self._result_cache[cache_key] = result

            # 更新性能统计
            await self._update_performance_stats(query, results)

            structured_logger.info(
                f"多策略检索完成",
                extra={
                    "strategies_used": list(results.keys()),
                    "combined_results_count": len(combined_chunks),
                    "best_strategy": best_strategy.value if best_strategy else None,
                    "processing_time_ms": processing_time,
                }
            )

            return result

        except Exception as e:
            structured_logger.error(f"多策略检索失败: {e}")
            raise Exception(f"Multi-strategy retrieval failed: {e}")

    def _select_strategies(self, query: RetrievalQuery) -> List[RetrievalStrategy]:
        """选择要使用的策略"""
        strategies_to_use = []

        # 如果查询指定了策略
        if query.strategy != RetrievalStrategy.HYBRID:
            if query.strategy in self.retrievers:
                strategies_to_use.append(query.strategy)
        else:
            # 使用所有启用的策略
            strategies_to_use = [
                strategy for strategy, config in self.strategies.items()
                if config.enabled
            ]

        # 限制策略数量
        max_strategies = self.config.get("max_parallel_strategies", 3)
        if len(strategies_to_use) > max_strategies:
            # 根据性能权重排序
            strategies_to_use.sort(
                key=lambda s: self.strategy_performance[s]["avg_score"],
                reverse=True
            )
            strategies_to_use = strategies_to_use[:max_strategies]

        return strategies_to_use

    async def _single_strategy_retrieval(
        self,
        query: RetrievalQuery,
        strategies: List[RetrievalStrategy]
    ) -> Dict[RetrievalStrategy, RetrievalResult]:
        """单策略检索"""
        results = {}

        if strategies:
            strategy = strategies[0]  # 使用第一个策略
            retriever = self.retrievers[strategy]
            result = await retriever.retrieve(query)
            results[strategy] = result

        return results

    async def _parallel_strategy_retrieval(
        self,
        query: RetrievalQuery,
        strategies: List[RetrievalStrategy]
    ) -> Dict[RetrievalStrategy, RetrievalResult]:
        """并行策略检索"""
        tasks = []
        strategy_names = []

        for strategy in strategies:
            retriever = self.retrievers[strategy]
            task = asyncio.create_task(retriever.retrieve(query))
            tasks.append(task)
            strategy_names.append(strategy)

        # 等待所有任务完成
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # 构建结果字典
        results = {}
        for strategy_name, result in zip(strategy_names, results_list):
            if isinstance(result, Exception):
                structured_logger.error(f"策略 {strategy_name.value} 执行失败: {result}")
                # 创建空结果
                results[strategy_name] = RetrievalResult(
                    chunks=[],
                    query=query.text,
                    strategy=strategy_name,
                    total_found=0,
                    search_time_ms=0.0,
                    created_at=datetime.utcnow().isoformat(),
                )
            else:
                results[strategy_name] = result

        return results

    async def _sequential_strategy_retrieval(
        self,
        query: RetrievalQuery,
        strategies: List[RetrievalStrategy]
    ) -> Dict[RetrievalStrategy, RetrievalResult]:
        """顺序策略检索"""
        results = {}

        for strategy in strategies:
            try:
                retriever = self.retrievers[strategy]
                result = await retriever.retrieve(query)
                results[strategy] = result

                # 如果已有足够好的结果，可以提前终止
                if result.chunks and result.chunks[0].score > 0.9:
                    structured_logger.info(f"策略 {strategy.value} 找到高质量结果，提前终止")
                    break

            except Exception as e:
                structured_logger.error(f"策略 {strategy.value} 执行失败: {e}")

        return results

    async def _fuse_results(
        self,
        results: Dict[RetrievalStrategy, RetrievalResult],
        query: RetrievalQuery
    ) -> List[DocumentChunk]:
        """融合多个策略的结果"""
        if not results:
            return []

        if len(results) == 1:
            # 单策略，直接返回
            return list(results.values())[0].chunks

        # 多策略融合
        if self.fusion_config.method == "weighted_sum":
            return await self._weighted_sum_fusion(results, query)
        elif self.fusion_config.method == "rank_fusion":
            return await self._rank_fusion(results, query)
        elif self.fusion_config.method == "score_fusion":
            return await self._score_fusion(results, query)
        elif self.fusion_config.method == "adaptive":
            return await self._adaptive_fusion(results, query)
        else:
            # 默认使用加权求和
            return await self._weighted_sum_fusion(results, query)

    async def _weighted_sum_fusion(
        self,
        results: Dict[RetrievalStrategy, RetrievalResult],
        query: RetrievalQuery
    ) -> List[DocumentChunk]:
        """加权求和融合"""
        # 收集所有文档片段
        all_chunks = defaultdict(lambda: {"chunk": None, "scores": [], "weights": []})

        for strategy, result in results.items():
            strategy_config = self.strategies.get(strategy)
            if not strategy_config or not strategy_config.enabled:
                continue

            weight = strategy_config.weight
            max_results_per_strategy = self.fusion_config.max_results_per_strategy

            for chunk in result.chunks[:max_results_per_strategy]:
                chunk_id = chunk.id
                if all_chunks[chunk_id]["chunk"] is None:
                    all_chunks[chunk_id]["chunk"] = chunk

                all_chunks[chunk_id]["scores"].append(chunk.score)
                all_chunks[chunk_id]["weights"].append(weight)

        # 计算加权平均分数
        fused_chunks = []
        for chunk_data in all_chunks.values():
            chunk = chunk_data["chunk"]
            scores = chunk_data["scores"]
            weights = chunk_data["weights"]

            if scores and weights:
                # 加权平均分数
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                chunk.score = weighted_score
                fused_chunks.append(chunk)

        # 排序并返回top-k
        fused_chunks.sort(key=lambda x: x.score, reverse=True)
        return fused_chunks[:self.fusion_config.top_k]

    async def _rank_fusion(
        self,
        results: Dict[RetrievalStrategy, RetrievalResult],
        query: RetrievalQuery
    ) -> List[DocumentChunk]:
        """排名融合（Reciprocal Rank Fusion）"""
        k = 60  # RRF常数
        chunk_scores = defaultdict(float)
        chunk_map = {}

        for strategy, result in results.items():
            strategy_config = self.strategies.get(strategy)
            if not strategy_config or not strategy_config.enabled:
                continue

            weight = strategy_config.weight

            for rank, chunk in enumerate(result.chunks):
                chunk_id = chunk.id
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = chunk

                # RRF分数
                rrf_score = weight / (k + rank + 1)
                chunk_scores[chunk_id] += rrf_score

        # 排序并返回
        fused_chunks = []
        for chunk_id, score in sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True):
            chunk = chunk_map[chunk_id]
            chunk.score = score
            fused_chunks.append(chunk)

        return fused_chunks[:self.fusion_config.top_k]

    async def _score_fusion(
        self,
        results: Dict[RetrievalStrategy, RetrievalResult],
        query: RetrievalQuery
    ) -> List[DocumentChunk]:
        """分数融合（标准化后的分数融合）"""
        chunk_scores = defaultdict(float)
        chunk_map = {}
        strategy_max_scores = {}

        # 计算每个策略的最大分数用于标准化
        for strategy, result in results.items():
            if result.chunks:
                strategy_max_scores[strategy] = max(chunk.score for chunk in result.chunks)
            else:
                strategy_max_scores[strategy] = 1.0

        for strategy, result in results.items():
            strategy_config = self.strategies.get(strategy)
            if not strategy_config or not strategy_config.enabled:
                continue

            weight = strategy_config.weight
            max_score = strategy_max_scores[strategy]

            for chunk in result.chunks:
                chunk_id = chunk.id
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = chunk

                # 标准化分数
                normalized_score = chunk.score / max_score if max_score > 0 else 0.0
                chunk_scores[chunk_id] += normalized_score * weight

        # 排序并返回
        fused_chunks = []
        for chunk_id, score in sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True):
            chunk = chunk_map[chunk_id]
            chunk.score = score
            fused_chunks.append(chunk)

        return fused_chunks[:self.fusion_config.top_k]

    async def _adaptive_fusion(
        self,
        results: Dict[RetrievalStrategy, RetrievalResult],
        query: RetrievalQuery
    ) -> List[DocumentChunk]:
        """自适应融合"""
        # 根据历史性能动态调整权重
        adaptive_weights = {}

        for strategy in results.keys():
            performance = self.strategy_performance[strategy]
            # 基于成功率和平均分数调整权重
            adaptive_weight = performance["success_rate"] * performance["avg_score"]
            adaptive_weights[strategy] = adaptive_weight

        # 使用自适应权重进行加权融合
        chunk_scores = defaultdict(float)
        chunk_map = {}

        for strategy, result in results.items():
            weight = adaptive_weights.get(strategy, 1.0)

            for chunk in result.chunks:
                chunk_id = chunk.id
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = chunk

                chunk_scores[chunk_id] += chunk.score * weight

        # 排序并返回
        fused_chunks = []
        for chunk_id, score in sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True):
            chunk = chunk_map[chunk_id]
            chunk.score = score
            fused_chunks.append(chunk)

        return fused_chunks[:self.fusion_config.top_k]

    async def _calculate_strategy_scores(
        self,
        results: Dict[RetrievalStrategy, RetrievalResult]
    ) -> Dict[RetrievalStrategy, float]:
        """计算策略评分"""
        strategy_scores = {}

        for strategy, result in results.items():
            if not result.chunks:
                strategy_scores[strategy] = 0.0
                continue

            # 基于结果质量计算评分
            avg_score = sum(chunk.score for chunk in result.chunks) / len(result.chunks)
            result_count = len(result.chunks)
            response_time = result.search_time_ms

            # 综合评分（分数 × 数量 ÷ 响应时间）
            strategy_score = (avg_score * result_count) / (response_time + 1)
            strategy_scores[strategy] = strategy_score

        return strategy_scores

    async def adaptive_retrieve(
        self,
        query: RetrievalQuery
    ) -> MultiStrategyResult:
        """
        自适应检索

        Args:
            query: 检索查询

        Returns:
            MultiStrategyResult: 自适应检索结果
        """
        if not self.enable_adaptive:
            return await self.retrieve_multi_strategy(query)

        # 基于查询类型和历史性能选择最佳策略组合
        optimal_strategies = await self._select_optimal_strategies(query)

        # 修改查询以使用最优策略
        original_mode = query.mode
        query.mode = RetrievalMode.MULTI

        # 临时限制策略
        temp_strategies = self.strategies.copy()
        self.strategies = {
            strategy: config for strategy, config in temp_strategies.items()
            if strategy in optimal_strategies
        }

        try:
            result = await self.retrieve_multi_strategy(query)
            return result
        finally:
            # 恢复原始策略配置
            self.strategies = temp_strategies
            query.mode = original_mode

    async def _select_optimal_strategies(self, query: RetrievalQuery) -> List[RetrievalStrategy]:
        """选择最优策略组合"""
        # 基于查询特征的策略选择（简化实现）
        query_lower = query.text.lower()

        selected_strategies = []

        # 检测查询类型
        if any(word in query_lower for word in ["who", "what", "where", "when", "how"]):
            # 事实性查询，优先使用关键词检索
            if RetrievalStrategy.BM25 in self.retrievers:
                selected_strategies.append(RetrievalStrategy.BM25)

        if len(query.text.split()) > 5:
            # 长查询，使用向量检索
            if RetrievalStrategy.VECTOR in self.retrievers:
                selected_strategies.append(RetrievalStrategy.VECTOR)

        # 如果包含实体名称，使用图检索
        for entity_id, entity in getattr(self.retrievers.get(RetrievalStrategy.GRAPH), 'entities', {}).items():
            if entity.name.lower() in query_lower:
                if RetrievalStrategy.GRAPH in self.retrievers:
                    selected_strategies.append(RetrievalStrategy.GRAPH)
                break

        # 如果没有选择到策略，使用默认策略
        if not selected_strategies:
            selected_strategies = list(self.retrievers.keys())[:2]  # 使用前两个策略

        return selected_strategies

    async def optimize_strategy_weights(
        self,
        training_queries: List[RetrievalQuery],
        ground_truth: List[List[str]]
    ) -> Dict[RetrievalStrategy, float]:
        """
        优化策略权重

        Args:
            training_queries: 训练查询
            ground_truth: 真实相关文档

        Returns:
            Dict[RetrievalStrategy, float]: 优化后的权重
        """
        if len(training_queries) != len(ground_truth):
            raise ValueError("训练查询和真实答案数量不匹配")

        structured_logger.info(f"开始优化策略权重，训练样本数: {len(training_queries)}")

        best_weights = {}
        best_score = 0.0

        # 网格搜索优化权重
        weight_combinations = self._generate_weight_combinations()

        for weight_comb in weight_combinations:
            # 更新策略权重
            for strategy, weight in weight_comb.items():
                if strategy in self.strategies:
                    self.strategies[strategy].weight = weight

            # 评估性能
            total_score = 0.0
            for query, truth_docs in zip(training_queries, ground_truth):
                try:
                    result = await self.retrieve_multi_strategy(query)
                    score = self._evaluate_result(result, truth_docs)
                    total_score += score
                except Exception as e:
                    structured_logger.warning(f"评估查询失败: {e}")
                    continue

            avg_score = total_score / len(training_queries)

            if avg_score > best_score:
                best_score = avg_score
                best_weights = weight_comb.copy()

        # 应用最佳权重
        for strategy, weight in best_weights.items():
            if strategy in self.strategies:
                self.strategies[strategy].weight = weight

        structured_logger.info(f"策略权重优化完成，最佳分数: {best_score:.3f}")
        return best_weights

    def _generate_weight_combinations(self) -> List[Dict[RetrievalStrategy, float]]:
        """生成权重组合"""
        strategies = list(self.strategies.keys())
        combinations = []

        # 简化的权重组合生成
        if len(strategies) == 2:
            weights = [0.1, 0.3, 0.5, 0.7, 0.9]
            for w1 in weights:
                w2 = 1.0 - w1
                if w2 >= 0.1:
                    combinations.append({
                        strategies[0]: w1,
                        strategies[1]: w2
                    })
        elif len(strategies) == 3:
            # 简化的三元组合
            base_weights = [0.2, 0.4, 0.6, 0.8]
            for w1 in base_weights:
                for w2 in base_weights:
                    w3 = 1.0 - w1 - w2
                    if w3 >= 0.1:
                        combinations.append({
                            strategies[0]: w1,
                            strategies[1]: w2,
                            strategies[2]: w3
                        })

        # 如果没有生成组合，使用默认等权重
        if not combinations:
            equal_weight = 1.0 / len(strategies)
            combinations.append({
                strategy: equal_weight for strategy in strategies
            })

        return combinations

    def _evaluate_result(self, result: MultiStrategyResult, truth_docs: List[str]) -> float:
        """评估检索结果"""
        if not result.combined_chunks or not truth_docs:
            return 0.0

        # 计算准确率@k
        retrieved_ids = {chunk.id for chunk in result.combined_chunks[:10]}
        truth_ids = set(truth_docs)

        if not retrieved_ids:
            return 0.0

        intersection = retrieved_ids & truth_ids
        precision = len(intersection) / len(retrieved_ids)
        recall = len(intersection) / len(truth_ids) if truth_ids else 0.0

        # F1分数
        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    async def _update_performance_stats(
        self,
        query: RetrievalQuery,
        results: Dict[RetrievalStrategy, RetrievalResult]
    ) -> None:
        """更新性能统计"""
        for strategy, result in results.items():
            stats = self.strategy_performance[strategy]

            # 更新使用计数
            stats["usage_count"] += 1

            # 更新平均时间
            stats["avg_time"] = (
                (stats["avg_time"] * (stats["usage_count"] - 1) + result.search_time_ms) /
                stats["usage_count"]
            )

            # 更新平均分数
            if result.chunks:
                avg_score = sum(chunk.score for chunk in result.chunks) / len(result.chunks)
                stats["avg_score"] = (
                    (stats["avg_score"] * (stats["usage_count"] - 1) + avg_score) /
                    stats["usage_count"]
                )
            else:
                # 没有结果，成功率下降
                stats["success_rate"] = (
                    (stats["success_rate"] * (stats["usage_count"] - 1)) /
                    stats["usage_count"]
                )

    def _get_cache_key(self, query: RetrievalQuery) -> str:
        """生成缓存键"""
        key_parts = [
            query.text,
            str(query.mode),
            query.strategy.value,
            str(query.top_k),
            str(query.min_score)
        ]
        return str(hash("_".join(key_parts)))

    async def cleanup(self) -> None:
        """清理资源"""
        try:
            # 清理所有检索器
            cleanup_tasks = [retriever.cleanup() for retriever in self.retrievers.values()]
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            # 清理缓存
            if self._result_cache:
                self._result_cache.clear()

            self._initialized = False
            structured_logger.info("多策略检索器清理完成")

        except Exception as e:
            structured_logger.error(f"多策略检索器清理失败: {e}")