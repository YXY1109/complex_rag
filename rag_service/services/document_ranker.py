"""
文档重排服务

基于RAGFlow架构的智能文档重排服务，
支持多种重排算法、语义相似度计算、相关性优化等功能。
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import json
from dataclasses import dataclass, field
from enum import Enum

from ..interfaces.rag_interface import DocumentChunk, RerankInterface


class RerankMethod(Enum):
    """重排方法。"""

    CROSS_ENCODER = "cross_encoder"
    BM25 = "bm25"
    TF_IDF = "tfidf"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    LEARNT_RANKING = "learnt_ranking"
    MULTISTAGE = "multistage"
    HYBRID = "hybrid"
    RECIPROCAL_RANK = "reciprocal_rank"


class ScoringMethod(Enum):
    """评分方法。"""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class RerankConfig:
    """重排配置。"""

    # 基础配置
    method: RerankMethod = RerankMethod.CROSS_ENCODER
    top_k: int = 10
    scoring_method: ScoringMethod = ScoringMethod.COSINE

    # 重排参数
    normalize_scores: bool = True
    boost_recent: bool = False
    boost_long_content: bool = False
    penalty_duplicates: bool = True

    # 多阶段配置
    stages: List[RerankMethod] = field(default_factory=lambda: [RerankMethod.SEMANTIC_SIMILARITY, RerankMethod.CROSS_ENCODER])
    stage_weights: List[float] = field(default_factory=lambda: [0.3, 0.7])

    # 过滤配置
    min_score_threshold: float = 0.1
    max_duplicates_ratio: float = 0.3
    content_length_range: Optional[Tuple[int, int]] = None

    # 语义配置
    semantic_threshold: float = 0.5
    diversity_boost: float = 0.1
    temporal_decay: float = 0.05


@dataclass
class RerankResult:
    """重排结果。"""

    original_chunks: List[DocumentChunk]
    reranked_chunks: List[DocumentChunk]
    method: RerankMethod
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

    @property
    def improvement_score(self) -> float:
        """计算重排改进分数。"""
        if not self.original_chunks or not self.reranked_chunks:
            return 0.0

        # 简化的改进分数计算
        original_scores = [chunk.score for chunk in self.original_chunks]
        reranked_scores = self.scores

        # 计算平均相关性分数的提升
        original_avg = sum(original_scores) / len(original_scores)
        reranked_avg = sum(reranked_scores) / len(reranked_scores)

        return (reranked_avg - original_avg) / max(original_avg, 0.01)


class DocumentRanker(RerankInterface):
    """文档重排服务。"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化文档重排服务。

        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 默认配置
        self.default_config = RerankConfig(**config.get("rerank", {}))

        # 重排模型缓存
        self.models: Dict[str, Any] = {}
        self._init_models()

        # 统计信息
        self.stats = {
            "total_rerank_requests": 0,
            "method_usage": {},
            "average_processing_time": 0.0,
            "average_improvement": 0.0
        }

    def _init_models(self) -> None:
        """初始化重排模型。"""
        # 这里可以初始化各种重排模型
        # 例如交叉编码器、学习排序模型等
        try:
            # 尝试加载sentence-transformers的交叉编码器
            from sentence_transformers import CrossEncoder
            self.models["cross_encoder"] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.logger.info("交叉编码器模型加载成功")
        except ImportError:
            self.logger.warning("sentence_transformers 未安装，交叉编码器功能不可用")

    async def rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        top_k: int = 10,
        config: Optional[RerankConfig] = None
    ) -> List[DocumentChunk]:
        """
        重排文档。

        Args:
            query: 查询字符串
            documents: 文档列表
            top_k: 返回结果数量
            config: 重排配置

        Returns:
            List[DocumentChunk]: 重排后的文档列表
        """
        start_time = datetime.now()
        self.stats["total_rerank_requests"] += 1

        try:
            # 使用配置
            rerank_config = config or self.default_config

            if not documents:
                return []

            # 更新统计
            method_name = rerank_config.method.value
            if method_name not in self.stats["method_usage"]:
                self.stats["method_usage"][method_name] = 0
            self.stats["method_usage"][method_name] += 1

            # 执行重排
            if rerank_config.method == RerankMethod.CROSS_ENCODER:
                reranked_docs, scores = await self._cross_encoder_rerank(query, documents, rerank_config)
            elif rerank_config.method == RerankMethod.BM25:
                reranked_docs, scores = await self._bm25_rerank(query, documents, rerank_config)
            elif rerank_config.method == RerankMethod.SEMANTIC_SIMILARITY:
                reranked_docs, scores = await self._semantic_similarity_rerank(query, documents, rerank_config)
            elif rerank_config.method == RerankMethod.MULTISTAGE:
                reranked_docs, scores = await self._multistage_rerank(query, documents, rerank_config)
            elif rerank_config.method == RerankMethod.HYBRID:
                reranked_docs, scores = await self._hybrid_rerank(query, documents, rerank_config)
            elif rerank_config.method == RerankMethod.RECIPROCAL_RANK:
                reranked_docs, scores = await self._reciprocal_rank_rerank(query, documents, rerank_config)
            else:
                # 默认使用语义相似度重排
                reranked_docs, scores = await self._semantic_similarity_rerank(query, documents, rerank_config)

            # 后处理
            reranked_docs = await self._postprocess_reranked_documents(
                reranked_docs, scores, query, rerank_config
            )

            # 限制返回数量
            final_docs = reranked_docs[:top_k]

            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["average_processing_time"] = (
                (self.stats["average_processing_time"] * (self.stats["total_rerank_requests"] - 1) + processing_time) /
                self.stats["total_rerank_requests"]
            )

            # 计算改进分数
            if len(reranked_docs) > 0:
                improvement = await self._calculate_improvement(documents, reranked_docs[:len(documents)], scores)
                self.stats["average_improvement"] = (
                    (self.stats["average_improvement"] * (self.stats["total_rerank_requests"] - 1) + improvement) /
                    self.stats["total_rerank_requests"]
                )

            self.logger.info(f"文档重排完成，方法: {method_name}，耗时: {processing_time:.3f}s")
            return final_docs

        except Exception as e:
            self.logger.error(f"文档重排失败: {e}")
            # 返回原始文档作为后备
            return documents[:top_k]

    async def _cross_encoder_rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        config: RerankConfig
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """交叉编码器重排。"""
        try:
            model = self.models.get("cross_encoder")
            if not model:
                self.logger.warning("交叉编码器模型不可用，使用语义相似度重排")
                return await self._semantic_similarity_rerank(query, documents, config)

            # 准备输入对
            input_pairs = [(query, doc.content) for doc in documents]

            # 预测相关性分数
            scores = model.predict(input_pairs)

            # 标准化分数
            if config.normalize_scores:
                scores = self._normalize_scores(scores)

            # 排序文档
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            reranked_docs = [doc for doc, _ in scored_docs]
            final_scores = [score for _, score in scored_docs]

            return reranked_docs, final_scores

        except Exception as e:
            self.logger.error(f"交叉编码器重排失败: {e}")
            return await self._semantic_similarity_rerank(query, documents, config)

    async def _bm25_rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        config: RerankConfig
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """BM25重排。"""
        try:
            # 简化的BM25实现
            query_terms = query.lower().split()
            doc_scores = []

            avg_doc_length = sum(len(doc.content.split()) for doc in documents) / len(documents)
            k1 = 1.5  # 控制词频饱和度
            b = 0.75  # 控制文档长度归一化程度

            for doc in documents:
                doc_terms = doc.content.lower().split()
                doc_length = len(doc_terms)

                score = 0.0
                for term in query_terms:
                    # 词频
                    tf = doc_terms.count(term)
                    if tf == 0:
                        continue

                    # 逆文档频率
                    df = sum(1 for d in documents if term in d.content.lower().split())
                    idf = np.log((len(documents) - df + 0.5) / (df + 0.5))

                    # BM25公式
                    normalized_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
                    score += idf * normalized_tf

                doc_scores.append(score)

            # 标准化分数
            if config.normalize_scores:
                doc_scores = self._normalize_scores(doc_scores)

            # 排序
            scored_docs = list(zip(documents, doc_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            reranked_docs = [doc for doc, _ in scored_docs]
            final_scores = [score for _, score in scored_docs]

            return reranked_docs, final_scores

        except Exception as e:
            self.logger.error(f"BM25重排失败: {e}")
            return await self._semantic_similarity_rerank(query, documents, config)

    async def _semantic_similarity_rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        config: RerankConfig
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """语义相似度重排。"""
        try:
            # 简化的语义相似度计算（基于TF-IDF向量）
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            # 构建文档集
            corpus = [doc.content for doc in documents]
            corpus.insert(0, query)  # 查询作为第一个文档

            # 计算TF-IDF向量
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)

            # 计算查询与文档的相似度
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]

            similarities = cosine_similarity(query_vector, doc_vectors).flatten()

            # 标准化分数
            if config.normalize_scores:
                similarities = self._normalize_scores(similarities)

            # 排序
            scored_docs = list(zip(documents, similarities))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            reranked_docs = [doc for doc, _ in scored_docs]
            final_scores = [score for _, score in scored_docs]

            return reranked_docs, final_scores

        except ImportError:
            # 如果sklearn不可用，使用简单的词重叠度
            return await self._simple_overlap_rerank(query, documents, config)
        except Exception as e:
            self.logger.error(f"语义相似度重排失败: {e}")
            return await self._simple_overlap_rerank(query, documents, config)

    async def _simple_overlap_rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        config: RerankConfig
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """简单重叠度重排（后备方案）。"""
        query_terms = set(query.lower().split())
        doc_scores = []

        for doc in documents:
            doc_terms = set(doc.content.lower().split())

            # 计算Jaccard相似度
            intersection = query_terms.intersection(doc_terms)
            union = query_terms.union(doc_terms)
            similarity = len(intersection) / len(union) if union else 0.0

            doc_scores.append(similarity)

        # 排序
        scored_docs = list(zip(documents, doc_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked_docs = [doc for doc, _ in scored_docs]
        final_scores = [score for _, score in scored_docs]

        return reranked_docs, final_scores

    async def _multistage_rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        config: RerankConfig
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """多阶段重排。"""
        if not config.stages:
            return await self._semantic_similarity_rerank(query, documents, config)

        current_docs = documents.copy()
        all_scores = [0.0] * len(documents)
        stage_weights = config.stage_weights or [1.0] * len(config.stages)

        for i, (stage, weight) in enumerate(zip(config.stages, stage_weights)):
            # 为每个阶段创建临时配置
            stage_config = RerankConfig(
                method=stage,
                top_k=len(current_docs),
                normalize_scores=True
            )

            # 执行阶段重排
            if stage == RerankMethod.CROSS_ENCODER:
                stage_docs, stage_scores = await self._cross_encoder_rerank(query, current_docs, stage_config)
            elif stage == RerankMethod.BM25:
                stage_docs, stage_scores = await self._bm25_rerank(query, current_docs, stage_config)
            elif stage == RerankMethod.SEMANTIC_SIMILARITY:
                stage_docs, stage_scores = await self._semantic_similarity_rerank(query, current_docs, stage_config)
            else:
                stage_docs, stage_scores = await self._semantic_similarity_rerank(query, current_docs, stage_config)

            # 更新分数（加权平均）
            for j, doc in enumerate(stage_docs):
                # 找到原始文档索引
                original_index = current_docs.index(doc)
                all_scores[original_index] += stage_scores[j] * weight

            # 更新当前文档列表（保持排序）
            current_docs = stage_docs

        # 重新排序最终结果
        scored_docs = list(zip(documents, all_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked_docs = [doc for doc, _ in scored_docs]
        final_scores = [score for _, score in scored_docs]

        return reranked_docs, final_scores

    async def _hybrid_rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        config: RerankConfig
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """混合重排（结合多种方法）。"""
        try:
            # 并行执行多种重排方法
            tasks = []

            # 语义相似度
            tasks.append(asyncio.create_task(
                self._semantic_similarity_rerank(query, documents, config)
            ))

            # BM25
            tasks.append(asyncio.create_task(
                self._bm25_rerank(query, documents, config)
            ))

            # 如果交叉编码器可用，也加入
            if "cross_encoder" in self.models:
                tasks.append(asyncio.create_task(
                    self._cross_encoder_rerank(query, documents, config)
                ))

            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 收集有效结果
            valid_results = []
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    valid_results.append(result)

            if not valid_results:
                return await self._semantic_similarity_rerank(query, documents, config)

            # 融合多种重排结果
            doc_score_map = {}
            doc_info = {}

            for stage_docs, stage_scores in valid_results:
                for doc, score in zip(stage_docs, stage_scores):
                    doc_id = doc.chunk_id or hash(doc.content)
                    if doc_id not in doc_score_map:
                        doc_score_map[doc_id] = []
                        doc_info[doc_id] = doc
                    doc_score_map[doc_id].append(score)

            # 计算平均分数
            final_scores = []
            reranked_docs = []

            for doc_id, scores in doc_score_map.items():
                avg_score = sum(scores) / len(scores)
                final_scores.append(avg_score)
                reranked_docs.append(doc_info[doc_id])

            # 按分数排序
            scored_docs = list(zip(reranked_docs, final_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            reranked_docs = [doc for doc, _ in scored_docs]
            final_scores = [score for _, score in scored_docs]

            return reranked_docs, final_scores

        except Exception as e:
            self.logger.error(f"混合重排失败: {e}")
            return await self._semantic_similarity_rerank(query, documents, config)

    async def _reciprocal_rank_rerank(
        self,
        query: str,
        documents: List[DocumentChunk],
        config: RerankConfig
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """互惠排名重排（Reciprocal Rank Fusion）。"""
        try:
            # 使用多种方法获取排名
            methods = [
                (self._semantic_similarity_rerank, "semantic"),
                (self._bm25_rerank, "bm25")
            ]

            if "cross_encoder" in self.models:
                methods.append((self._cross_encoder_rerank, "cross_encoder"))

            # 收集每种方法的排名
            rankings = {}
            for method_func, method_name in methods:
                try:
                    method_config = RerankConfig(
                        method=RerankMethod.SEMANTIC_SIMILARITY,
                        normalize_scores=True
                    )
                    ranked_docs, _ = await method_func(query, documents, method_config)
                    rankings[method_name] = [doc.chunk_id for doc in ranked_docs]
                except Exception as e:
                    self.logger.warning(f"方法 {method_name} 在互惠排名中失败: {e}")

            if not rankings:
                return await self._semantic_similarity_rerank(query, documents, config)

            # 计算互惠排名分数
            k = 60  # RRF常数
            doc_scores = {}

            for method_name, ranked_docs in rankings.items():
                for rank, doc_id in enumerate(ranked_docs):
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0
                    doc_scores[doc_id] += 1.0 / (k + rank + 1)

            # 找到对应的文档对象
            final_docs = []
            final_scores = []

            for doc in documents:
                doc_id = doc.chunk_id or hash(doc.content)
                if doc_id in doc_scores:
                    final_docs.append(doc)
                    final_scores.append(doc_scores[doc_id])

            # 按分数排序
            scored_docs = list(zip(final_docs, final_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            reranked_docs = [doc for doc, _ in scored_docs]
            final_scores = [score for _, score in scored_docs]

            return reranked_docs, final_scores

        except Exception as e:
            self.logger.error(f"互惠排名重排失败: {e}")
            return await self._semantic_similarity_rerank(query, documents, config)

    async def _postprocess_reranked_documents(
        self,
        documents: List[DocumentChunk],
        scores: List[float],
        query: str,
        config: RerankConfig
    ) -> List[DocumentChunk]:
        """后处理重排后的文档。"""
        processed_docs = []

        for i, (doc, score) in enumerate(zip(documents, scores)):
            # 过滤低分文档
            if score < config.min_score_threshold:
                continue

            # 内容长度过滤
            if config.content_length_range:
                min_len, max_len = config.content_length_range
                if not (min_len <= len(doc.content) <= max_len):
                    continue

            # 更新文档分数
            doc.score = score

            # 添加重排元数据
            doc.metadata.update({
                "rerank_score": score,
                "rerank_rank": i + 1,
                "rerank_method": config.method.value,
                "rerank_timestamp": datetime.now().isoformat()
            })

            processed_docs.append(doc)

        # 去重处理
        if config.penalty_duplicates:
            processed_docs = self._remove_duplicates(processed_docs, config)

        # 时间衰减
        if config.boost_recent:
            processed_docs = self._apply_temporal_decay(processed_docs, config)

        # 长度提升
        if config.boost_long_content:
            processed_docs = self._boost_long_content(processed_docs, config)

        # 多样性提升
        if config.diversity_boost > 0:
            processed_docs = self._boost_diversity(processed_docs, query, config)

        return processed_docs

    def _remove_duplicates(
        self,
        documents: List[DocumentChunk],
        config: RerankConfig
    ) -> List[DocumentChunk]:
        """移除重复文档。"""
        unique_docs = []
        seen_content = set()

        for doc in documents:
            # 使用内容的哈希作为去重依据
            content_hash = hash(doc.content[:200])  # 使用前200字符的哈希
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
            else:
                # 重复文档降低分数
                doc.score *= (1 - config.max_duplicates_ratio)

        return unique_docs

    def _apply_temporal_decay(
        self,
        documents: List[DocumentChunk],
        config: RerankConfig
    ) -> List[DocumentChunk]:
        """应用时间衰减。"""
        current_time = datetime.now()

        for doc in documents:
            created_time = doc.metadata.get("created_at")
            if created_time:
                if isinstance(created_time, str):
                    created_time = datetime.fromisoformat(created_time)

                # 计算时间差（天）
                days_old = (current_time - created_time).days
                decay_factor = np.exp(-config.temporal_decay * days_old)
                doc.score *= decay_factor

        return documents

    def _boost_long_content(
        self,
        documents: List[DocumentChunk],
        config: RerankConfig
    ) -> List[DocumentChunk]:
        """提升长内容分数。"""
        avg_length = sum(len(doc.content) for doc in documents) / len(documents)

        for doc in documents:
            content_length = len(doc.content)
            if content_length > avg_length:
                boost = min(content_length / avg_length, 2.0)  # 最大提升2倍
                doc.score *= boost

        # 重新排序
        documents.sort(key=lambda x: x.score, reverse=True)
        return documents

    def _boost_diversity(
        self,
        documents: List[DocumentChunk],
        query: str,
        config: RerankConfig
    ) -> List[DocumentChunk]:
        """提升多样性。"""
        if len(documents) <= 1:
            return documents

        # 简化的多样性提升：确保不同文档类型或来源的文档有更好的分布
        diversified_docs = [documents[0]]  # 总是保留最高分的文档

        for doc in documents[1:]:
            # 检查与已选文档的相似性
            is_diverse = True
            for selected_doc in diversified_docs:
                similarity = self._calculate_content_similarity(doc.content, selected_doc.content)
                if similarity > 0.8:  # 相似度阈值
                    is_diverse = False
                    break

            if is_diverse:
                doc.score += config.diversity_boost
                diversified_docs.append(doc)

        return diversified_docs

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度。"""
        # 简化的相似度计算
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """标准化分数到0-1范围。"""
        if not scores:
            return scores

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [0.5] * len(scores)

        normalized = [(score - min_score) / (max_score - min_score) for score in scores]
        return normalized

    async def _calculate_improvement(
        self,
        original_docs: List[DocumentChunk],
        reranked_docs: List[DocumentChunk],
        reranked_scores: List[float]
    ) -> float:
        """计算重排改进分数。"""
        if len(original_docs) == 0 or len(reranked_docs) == 0:
            return 0.0

        # 计算原始平均分数
        original_scores = [doc.score for doc in original_docs]
        original_avg = sum(original_scores) / len(original_scores)

        # 计算重排后平均分数
        reranked_avg = sum(reranked_scores) / len(reranked_scores)

        # 计算改进分数
        improvement = (reranked_avg - original_avg) / max(abs(original_avg), 0.01)
        return improvement

    async def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[DocumentChunk]],
        config: Optional[RerankConfig] = None
    ) -> List[List[DocumentChunk]]:
        """批量重排。"""
        if len(queries) != len(documents_list):
            raise ValueError("查询数量和文档列表数量不匹配")

        results = []
        rerank_config = config or self.default_config

        # 并行处理
        tasks = []
        for query, documents in zip(queries, documents_list):
            task = asyncio.create_task(
                self.rerank(query, documents, rerank_config.top_k, rerank_config)
            )
            tasks.append(task)

        # 等待所有任务完成
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        for result in batch_results:
            if isinstance(result, list):
                results.append(result)
            else:
                self.logger.error(f"批量重排中某个任务失败: {result}")
                results.append([])  # 空结果作为后备

        return results

    async def get_rerank_statistics(self) -> Dict[str, Any]:
        """获取重排统计信息。"""
        return {
            **self.stats,
            "available_methods": [method.value for method in RerankMethod],
            "available_models": list(self.models.keys()),
            "default_config": self.default_config.__dict__
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查。"""
        health_status = {
            "status": "healthy",
            "models_loaded": len(self.models),
            "supported_methods": [method.value for method in RerankMethod],
            "errors": []
        }

        # 检查模型状态
        try:
            for model_name, model in self.models.items():
                # 简单的健康检查
                if hasattr(model, 'predict'):
                    health_status["models_loaded"] += 1
        except Exception as e:
            health_status["errors"].append(f"Model check failed: {str(e)}")

        return health_status