"""
上下文构建算法

实现检索结果的上下文组织、压缩和优化。
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import heapq
import math

from ..interfaces.pipeline_interface import (
    Context,
    ContextDocument,
    RetrievalResult,
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.pipeline.stages.context_building")


class ContextBuilder:
    """
    上下文构建器

    负责将检索结果组织成适合大模型理解的上下文。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化上下文构建器

        Args:
            config: 配置参数
        """
        self.config = config
        self.max_context_tokens = config.get("max_context_tokens", 4000)
        self.context_window_overlap = config.get("context_window_overlap", 50)
        self.enable_context_compression = config.get("enable_context_compression", True)
        self.enable_context_ranking = config.get("enable_context_ranking", True)
        self.min_document_relevance = config.get("min_document_relevance", 0.3)

        # 上下文构建策略
        self.building_strategy = config.get("building_strategy", "relevance_ranked")

        # 文档压缩配置
        self.compression_ratio = config.get("compression_ratio", 0.8)
        self.preserve_key_info = config.get("preserve_key_info", True)

        self._initialized = False

    async def initialize(self) -> bool:
        """
        初始化上下文构建器

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "初始化上下文构建器",
                extra={
                    "max_context_tokens": self.max_context_tokens,
                    "building_strategy": self.building_strategy,
                    "enable_compression": self.enable_context_compression,
                    "enable_ranking": self.enable_context_ranking,
                }
            )

            self._initialized = True
            structured_logger.info("上下文构建器初始化成功")
            return True

        except Exception as e:
            structured_logger.error(f"上下文构建器初始化失败: {e}")
            return False

    async def build_context(
        self,
        retrieval_results: RetrievalResult,
        max_length: Optional[int] = None
    ) -> Context:
        """
        构建上下文

        Args:
            retrieval_results: 检索结果
            max_length: 最大长度（token数）

        Returns:
            Context: 构建的上下文
        """
        if not self._initialized:
            raise RuntimeError("上下文构建器未初始化")

        start_time = time.time()
        max_length = max_length or self.max_context_tokens

        try:
            structured_logger.info(
                "开始构建上下文",
                extra={
                    "input_chunks_count": len(retrieval_results.chunks),
                    "max_length": max_length,
                    "building_strategy": self.building_strategy,
                }
            )

            # 转换检索结果为上下文文档
            context_docs = await self._convert_to_context_documents(retrieval_results)

            # 过滤低相关性文档
            if self.enable_context_ranking:
                context_docs = await self._filter_by_relevance(context_docs)

            # 文档重排序
            if self.enable_context_ranking:
                context_docs = await self._rerank_documents(context_docs, retrieval_results)

            # 根据策略构建上下文
            if self.building_strategy == "relevance_ranked":
                formatted_context = await self._build_relevance_ranked_context(
                    context_docs, max_length
                )
            elif self.building_strategy == "diversified":
                formatted_context = await self._build_diversified_context(
                    context_docs, max_length
                )
            elif self.building_strategy == "chronological":
                formatted_context = await self._build_chronological_context(
                    context_docs, max_length
                )
            else:
                # 默认使用相关性排序
                formatted_context = await self._build_relevance_ranked_context(
                    context_docs, max_length
                )

            # 计算上下文统计信息
            total_length = len(formatted_context.split())
            relevance_score = self._calculate_context_relevance(context_docs)

            construction_time = (time.time() - start_time) * 1000

            result = Context(
                documents=context_docs,
                formatted_context=formatted_context,
                total_length=total_length,
                relevance_score=relevance_score,
                construction_time_ms=construction_time,
                metadata={
                    "building_strategy": self.building_strategy,
                    "original_chunks_count": len(retrieval_results.chunks),
                    "filtered_docs_count": len(context_docs),
                    "compression_applied": self.enable_context_compression,
                },
            )

            structured_logger.info(
                "上下文构建完成",
                extra={
                    "documents_count": len(context_docs),
                    "total_length": total_length,
                    "relevance_score": relevance_score,
                    "construction_time_ms": construction_time,
                }
            )

            return result

        except Exception as e:
            structured_logger.error(f"上下文构建失败: {e}")
            raise Exception(f"Context building failed: {e}")

    async def _convert_to_context_documents(
        self,
        retrieval_results: RetrievalResult
    ) -> List[ContextDocument]:
        """转换检索结果为上下文文档"""
        context_docs = []

        for i, chunk in enumerate(retrieval_results.chunks):
            # 提取文档内容
            content = chunk.get("content", "") if isinstance(chunk, dict) else getattr(chunk, "content", "")

            # 提取元数据
            metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else getattr(chunk, "metadata", {})
            source = metadata.get("source") or chunk.get("source")
            title = metadata.get("title") or chunk.get("title")
            url = metadata.get("url") or chunk.get("url")

            # 计算相关性分数
            score = chunk.get("score", 0.0) if isinstance(chunk, dict) else getattr(chunk, "score", 0.0)

            # 生成引用ID
            citation_id = f"doc_{i+1}"

            context_doc = ContextDocument(
                id=chunk.get("id", f"doc_{i}") if isinstance(chunk, dict) else getattr(chunk, "id", f"doc_{i}"),
                content=content,
                score=score,
                source=source,
                title=title,
                url=url,
                metadata=metadata,
                relevance_score=score,
                citation_id=citation_id,
            )

            context_docs.append(context_doc)

        return context_docs

    async def _filter_by_relevance(
        self,
        documents: List[ContextDocument]
    ) -> List[ContextDocument]:
        """基于相关性过滤文档"""
        filtered_docs = []

        for doc in documents:
            if doc.relevance_score >= self.min_document_relevance:
                filtered_docs.append(doc)

        structured_logger.debug(
            f"相关性过滤: {len(documents)} -> {len(filtered_docs)}"
        )
        return filtered_docs

    async def _rerank_documents(
        self,
        documents: List[ContextDocument],
        retrieval_results: RetrievalResult
    ) -> List[ContextDocument]:
        """重新排序文档"""
        if not documents:
            return documents

        # 多因子排序
        def calculate_final_score(doc: ContextDocument) -> float:
            base_score = doc.relevance_score

            # 长度惩罚（太长或太短的文档）
            content_length = len(doc.content.split())
            if content_length < 10:
                length_penalty = 0.5
            elif content_length > 500:
                length_penalty = 0.8
            else:
                length_penalty = 1.0

            # 来源权重
            source_weight = 1.0
            if doc.source:
                if "wiki" in doc.source.lower():
                    source_weight = 1.1
                elif "official" in doc.source.lower():
                    source_weight = 1.2
                elif "blog" in doc.source.lower():
                    source_weight = 0.9

            # 多样性奖励（基于内容差异）
            diversity_bonus = 1.0

            return base_score * length_penalty * source_weight * diversity_bonus

        # 计算最终分数
        for doc in documents:
            doc.metadata["rerank_score"] = calculate_final_score(doc)

        # 按最终分数排序
        documents.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

        return documents

    async def _build_relevance_ranked_context(
        self,
        documents: List[ContextDocument],
        max_length: int
    ) -> str:
        """构建基于相关性排序的上下文"""
        if not documents:
            return ""

        context_parts = []
        current_length = 0

        for doc in documents:
            content = doc.content

            # 应用压缩
            if self.enable_context_compression:
                content = await self._compress_content(content)

            # 检查长度限制
            content_length = len(content.split())
            if current_length + content_length > max_length:
                # 截断内容
                remaining_tokens = max_length - current_length
                if remaining_tokens > 10:  # 至少保留10个词
                    content = self._truncate_content(content, remaining_tokens)
                else:
                    break

            # 格式化文档片段
            formatted_doc = self._format_document(doc, content)
            context_parts.append(formatted_doc)
            current_length += content_length

            if current_length >= max_length:
                break

        return "\n\n".join(context_parts)

    async def _build_diversified_context(
        self,
        documents: List[ContextDocument],
        max_length: int
    ) -> str:
        """构建多样化的上下文"""
        if not documents:
            return ""

        # 使用贪心算法选择多样化的文档
        selected_docs = []
        current_length = 0

        # 首先选择最相关的文档
        if documents:
            selected_docs.append(documents[0])
            current_length += len(documents[0].content.split())

        # 然后选择与已选文档差异最大的文档
        remaining_docs = documents[1:]

        while remaining_docs and current_length < max_length:
            best_doc = None
            best_diversity_score = -1

            for doc in remaining_docs:
                # 计算与已选文档的平均差异度
                diversity_score = self._calculate_diversity_score(doc, selected_docs)

                # 结合相关性和多样性
                combined_score = doc.relevance_score * 0.7 + diversity_score * 0.3

                if combined_score > best_diversity_score:
                    best_diversity_score = combined_score
                    best_doc = doc

            if best_doc:
                content = best_doc.content
                if self.enable_context_compression:
                    content = await self._compress_content(content)

                content_length = len(content.split())
                if current_length + content_length <= max_length:
                    selected_docs.append(best_doc)
                    remaining_docs.remove(best_doc)
                    current_length += content_length
                else:
                    break
            else:
                break

        # 格式化上下文
        context_parts = []
        for doc in selected_docs:
            content = doc.content
            if self.enable_context_compression:
                content = await self._compress_content(content)

            formatted_doc = self._format_document(doc, content)
            context_parts.append(formatted_doc)

        return "\n\n".join(context_parts)

    async def _build_chronological_context(
        self,
        documents: List[ContextDocument],
        max_length: int
    ) -> str:
        """构建按时间排序的上下文"""
        if not documents:
            return ""

        # 尝试从元数据中提取时间信息
        def extract_time(doc: ContextDocument) -> float:
            time_str = doc.metadata.get("created_at") or doc.metadata.get("date") or doc.metadata.get("time")
            if time_str:
                try:
                    # 简单的时间解析
                    if isinstance(time_str, str):
                        return float(time_str.replace("-", "").replace(":", "").replace(" ", ""))
                except:
                    pass
            return 0.0  # 默认时间

        # 按时间排序
        documents.sort(key=extract_time)

        # 构建上下文
        context_parts = []
        current_length = 0

        for doc in documents:
            content = doc.content
            if self.enable_context_compression:
                content = await self._compress_content(content)

            content_length = len(content.split())
            if current_length + content_length > max_length:
                break

            formatted_doc = self._format_document(doc, content)
            context_parts.append(formatted_doc)
            current_length += content_length

        return "\n\n".join(context_parts)

    async def _compress_content(self, content: str) -> str:
        """压缩内容"""
        if not self.enable_context_compression:
            return content

        try:
            # 简单的压缩策略：保留关键句子
            sentences = content.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= 3:
                return content

            # 计算句子重要性
            sentence_scores = []
            for sentence in sentences:
                score = len(sentence.split())  # 基于长度
                # 包含关键词的句子加分
                keywords = ["重要", "关键", "主要", "核心", "important", "key", "main"]
                for keyword in keywords:
                    if keyword in sentence.lower():
                        score += 5
                sentence_scores.append((sentence, score))

            # 选择重要句子
            sentences.sort(key=lambda x: x[1], reverse=True)
            target_count = max(3, int(len(sentences) * self.compression_ratio))
            selected_sentences = [s[0] for s in sentences[:target_count]]

            # 按原始顺序重新排列
            result_sentences = []
            for original_sentence in sentences:
                if original_sentence in selected_sentences:
                    result_sentences.append(original_sentence)

            return '. '.join(result_sentences) + '.'

        except Exception as e:
            structured_logger.warning(f"内容压缩失败: {e}")
            return content

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """截断内容"""
        words = content.split()
        if len(words) <= max_tokens:
            return content

        truncated_words = words[:max_tokens]
        # 尝试在句子边界截断
        result = ' '.join(truncated_words)
        last_period = result.rfind('.')
        if last_period > len(result) * 0.8:  # 如果句号位置合理
            result = result[:last_period + 1]

        return result

    def _format_document(self, doc: ContextDocument, content: str) -> str:
        """格式化文档片段"""
        formatted_parts = []

        # 添加引用标记
        if doc.citation_id:
            formatted_parts.append(f"[{doc.citation_id}]")

        # 添加标题
        if doc.title:
            formatted_parts.append(f"**{doc.title}**")

        # 添加内容
        formatted_parts.append(content)

        # 添加来源
        if doc.source:
            formatted_parts.append(f"来源: {doc.source}")

        return '\n'.join(formatted_parts)

    def _calculate_diversity_score(
        self,
        doc: ContextDocument,
        selected_docs: List[ContextDocument]
    ) -> float:
        """计算文档多样性分数"""
        if not selected_docs:
            return 1.0

        # 计算与已选文档的词汇差异度
        doc_words = set(doc.content.lower().split())
        total_similarity = 0.0

        for selected_doc in selected_docs:
            selected_words = set(selected_doc.content.lower().split())

            # Jaccard相似度
            intersection = doc_words & selected_words
            union = doc_words | selected_words
            if union:
                similarity = len(intersection) / len(union)
                total_similarity += similarity

        # 平均相似度
        avg_similarity = total_similarity / len(selected_docs)

        # 多样性分数 = 1 - 相似度
        diversity_score = 1.0 - avg_similarity
        return diversity_score

    def _calculate_context_relevance(self, documents: List[ContextDocument]) -> float:
        """计算上下文整体相关性"""
        if not documents:
            return 0.0

        # 加权平均相关性
        total_weight = 0.0
        weighted_score = 0.0

        for i, doc in enumerate(documents):
            # 位置权重（前面的文档权重更高）
            position_weight = 1.0 / (i + 1)
            weight = doc.relevance_score * position_weight

            weighted_score += weight
            total_weight += position_weight

        if total_weight == 0:
            return 0.0

        return weighted_score / total_weight

    async def batch_build_context(
        self,
        retrieval_results_list: List[RetrievalResult]
    ) -> List[Context]:
        """批量构建上下文"""
        if not self._initialized:
            raise RuntimeError("上下文构建器未初始化")

        try:
            structured_logger.info(f"开始批量构建上下文，数量: {len(retrieval_results_list)}")

            # 并行处理
            tasks = [self.build_context(results) for results in retrieval_results_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理异常
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    structured_logger.error(f"上下文 {i} 构建失败: {result}")
                    # 创建空上下文
                    valid_results.append(Context(
                        documents=[],
                        formatted_context="",
                        total_length=0,
                        relevance_score=0.0,
                        construction_time_ms=0.0,
                    ))
                else:
                    valid_results.append(result)

            structured_logger.info(f"批量上下文构建完成，成功处理 {len(valid_results)} 个")
            return valid_results

        except Exception as e:
            structured_logger.error(f"批量上下文构建失败: {e}")
            raise Exception(f"Batch context building failed: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "initialized": self._initialized,
            "max_context_tokens": self.max_context_tokens,
            "building_strategy": self.building_strategy,
            "enable_context_compression": self.enable_context_compression,
            "enable_context_ranking": self.enable_context_ranking,
            "compression_ratio": self.compression_ratio,
            "min_document_relevance": self.min_document_relevance,
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 创建测试数据
            from ..interfaces.pipeline_interface import RetrievalResult
            test_results = RetrievalResult(
                chunks=[
                    {"content": "这是一个测试文档", "score": 0.8, "id": "test1"},
                    {"content": "这是另一个测试文档", "score": 0.6, "id": "test2"},
                ],
                query="测试查询",
                strategy="test",
                total_found=2,
                search_time_ms=10.0,
            )

            result = await self.build_context(test_results, 100)

            return {
                "status": "healthy",
                "initialized": self._initialized,
                "test_construction_time_ms": result.construction_time_ms,
                "test_documents_count": len(result.documents),
                "test_context_length": result.total_length,
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
            self._initialized = False
            structured_logger.info("上下文构建器清理完成")

        except Exception as e:
            structured_logger.error(f"上下文构建器清理失败: {e}")