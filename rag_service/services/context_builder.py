"""
上下文构建服务

基于RAGFlow架构的智能上下文构建服务，
支持多种上下文策略、内容优化、语义连贯性保证等功能。
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from ..interfaces.rag_interface import (
    DocumentChunk, GenerationContext, ContextStrategy,
    RAGConfig, RAGQuery
)


class ContextCompressionMode(Enum):
    """上下文压缩模式。"""

    NONE = "none"
    TRUNCATION = "truncation"
    SUMMARIZATION = "summarization"
    RELEVANCE_FILTERING = "relevance_filtering"
    SEMANTIC_COMPRESSION = "semantic_compression"


class ContextReorderingMode(Enum):
    """上下文重排模式。"""

    ORIGINAL = "original"
    RELEVANCE = "relevance"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class ContextBuilderConfig:
    """上下文构建配置。"""

    # 基础配置
    context_strategy: ContextStrategy = ContextStrategy.STUFFING
    max_context_length: int = 4000
    min_context_length: int = 100
    chunk_separator: str = "\n\n"
    include_metadata: bool = False

    # 压缩配置
    compression_mode: ContextCompressionMode = ContextCompressionMode.NONE
    compression_ratio: float = 0.8
    preserve_key_info: bool = True

    # 重排配置
    reordering_mode: ContextReorderingMode = ContextReorderingMode.RELEVANCE
    max_chunks_per_context: int = 20

    # 质量控制
    min_relevance_score: float = 0.5
    max_duplicate_ratio: float = 0.3
    coherence_threshold: float = 0.7

    # 增强功能
    enable_context_enhancement: bool = True
    enable_cross_reference: bool = True
    enable_temporal_ordering: bool = True


@dataclass
class ContextSection:
    """上下文段落。"""

    section_id: str
    title: str
    content: str
    chunks: List[DocumentChunk]
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.section_id:
            self.section_id = str(hash(self.title + self.content))


class ContextBuilder:
    """上下文构建服务。"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化上下文构建服务。

        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 默认配置
        self.default_config = ContextBuilderConfig(**config.get("context_builder", {}))

        # 上下文模板
        self.templates = config.get("templates", {})
        self._load_default_templates()

        # 统计信息
        self.stats = {
            "total_contexts_built": 0,
            "average_context_length": 0.0,
            "strategy_usage": {},
            "compression_count": 0,
            "reordering_count": 0
        }

    def _load_default_templates(self) -> None:
        """加载默认模板。"""
        self.templates.update({
            "qa_context": """
请根据以下上下文信息回答问题：

上下文：
{context}

问题：{query}

回答：
""",
            "summarization_context": """
请根据以下信息进行总结：

信息：
{context}

请提供简洁准确的总结：
""",
            "conversation_context": """
对话历史：
{history}

相关背景信息：
{context}

当前问题：{query}

请基于以上信息回答：
"""
        })

    async def build_context(
        self,
        chunks: List[DocumentChunk],
        query: str,
        config: Optional[ContextBuilderConfig] = None,
        rag_query: Optional[RAGQuery] = None
    ) -> GenerationContext:
        """
        构建生成上下文。

        Args:
            chunks: 文档块列表
            query: 查询字符串
            config: 上下文构建配置
            rag_query: RAG查询对象

        Returns:
            GenerationContext: 生成上下文
        """
        try:
            # 使用配置
            context_config = config or self.default_config

            self.stats["total_contexts_built"] += 1
            strategy_name = context_config.context_strategy.value
            if strategy_name not in self.stats["strategy_usage"]:
                self.stats["strategy_usage"][strategy_name] = 0
            self.stats["strategy_usage"][strategy_name] += 1

            # 根据策略构建上下文
            if context_config.context_strategy == ContextStrategy.STUFFING:
                context = await self._build_stuffing_context(chunks, query, context_config)
            elif context_config.context_strategy == ContextStrategy.MAP_REDUCE:
                context = await self._build_map_reduce_context(chunks, query, context_config)
            elif context_config.context_strategy == ContextStrategy.REFINE:
                context = await self._build_refine_context(chunks, query, context_config)
            elif context_config.context_strategy == ContextStrategy.COMPRESSION:
                context = await self._build_compression_context(chunks, query, context_config)
            else:
                context = await self._build_stuffing_context(chunks, query, context_config)

            # 后处理
            context = await self._postprocess_context(context, query, context_config)

            # 更新统计
            context_length = len(context.formatted_context)
            total_contexts = self.stats["total_contexts_built"]
            self.stats["average_context_length"] = (
                (self.stats["average_context_length"] * (total_contexts - 1) + context_length) /
                total_contexts
            )

            self.logger.info(f"上下文构建完成，策略: {strategy_name}，长度: {context_length}")

            return context

        except Exception as e:
            self.logger.error(f"上下文构建失败: {e}")
            # 返回简化上下文作为后备
            return self._create_fallback_context(chunks, query)

    async def _build_stuffing_context(
        self,
        chunks: List[DocumentChunk],
        query: str,
        config: ContextBuilderConfig
    ) -> GenerationContext:
        """构建填充式上下文。"""
        # 重排文档块
        reordered_chunks = await self._reorder_chunks(chunks, query, config)

        # 压缩内容
        compressed_chunks = await self._compress_chunks(reordered_chunks, config)

        # 格式化上下文
        formatted_context = await self._format_chunks(compressed_chunks, config)

        # 创建上下文段落
        context_sections = await self._create_context_sections(compressed_chunks, query, config)

        return GenerationContext(
            context_chunks=compressed_chunks,
            formatted_context=formatted_context,
            context_length=len(formatted_context),
            truncation_info={
                "original_chunks": len(chunks),
                "final_chunks": len(compressed_chunks),
                "compression_applied": config.compression_mode != ContextCompressionMode.NONE
            }
        )

    async def _build_map_reduce_context(
        self,
        chunks: List[DocumentChunk],
        query: str,
        config: ContextBuilderConfig
    ) -> GenerationContext:
        """构建Map-Reduce上下文。"""
        # Map阶段：处理每个文档块
        mapped_results = []
        for chunk in chunks[:config.max_chunks_per_context]:
            chunk_context = await self._process_chunk_map(chunk, query, config)
            mapped_results.append(chunk_context)

        # Reduce阶段：合并处理结果
        reduced_context = await self._reduce_mapped_results(mapped_results, config)

        return GenerationContext(
            context_chunks=chunks[:config.max_chunks_per_context],
            formatted_context=reduced_context,
            context_length=len(reduced_context),
            truncation_info={
                "strategy": "map_reduce",
                "mapped_chunks": len(mapped_results)
            }
        )

    async def _build_refine_context(
        self,
        chunks: List[DocumentChunk],
        query: str,
        config: ContextBuilderConfig
    ) -> GenerationContext:
        """构建精炼式上下文。"""
        refined_context = ""
        processed_chunks = []

        # 逐步精炼上下文
        for i, chunk in enumerate(chunks[:config.max_chunks_per_context]):
            if i == 0:
                # 第一个块直接使用
                refined_context = chunk.content
                processed_chunks.append(chunk)
            else:
                # 精炼处理
                refined_context = await self._refine_context_with_chunk(
                    refined_context, chunk, query, config
                )
                processed_chunks.append(chunk)

            # 检查长度限制
            if len(refined_context) > config.max_context_length:
                break

        return GenerationContext(
            context_chunks=processed_chunks,
            formatted_context=refined_context,
            context_length=len(refined_context),
            truncation_info={
                "strategy": "refine",
                "refinement_steps": len(processed_chunks)
            }
        )

    async def _build_compression_context(
        self,
        chunks: List[DocumentChunk],
        query: str,
        config: ContextBuilderConfig
    ) -> GenerationContext:
        """构建压缩式上下文。"""
        # 提取关键信息
        key_chunks = await self._extract_key_chunks(chunks, query, config)

        # 压缩内容
        compressed_content = await self._compress_content(key_chunks, config)

        return GenerationContext(
            context_chunks=key_chunks,
            formatted_context=compressed_content,
            context_length=len(compressed_content),
            truncation_info={
                "strategy": "compression",
                "compression_ratio": config.compression_ratio,
                "original_chunks": len(chunks),
                "key_chunks": len(key_chunks)
            }
        )

    async def _reorder_chunks(
        self,
        chunks: List[DocumentChunk],
        query: str,
        config: ContextBuilderConfig
    ) -> List[DocumentChunk]:
        """重排文档块。"""
        if config.reordering_mode == ContextReorderingMode.ORIGINAL:
            return chunks

        self.stats["reordering_count"] += 1

        if config.reordering_mode == ContextReorderingMode.RELEVANCE:
            # 按相关性分数排序
            return sorted(chunks, key=lambda x: x.score, reverse=True)

        elif config.reordering_mode == ContextReorderingMode.TEMPORAL:
            # 按时间排序
            return sorted(chunks, key=lambda x: x.metadata.get("created_at", 0), reverse=True)

        elif config.reordering_mode == ContextReorderingMode.SEMANTIC:
            # 语义相似性排序
            return await self._semantic_reorder(chunks, query)

        elif config.reordering_mode == ContextReorderingMode.HYBRID:
            # 混合排序：相关性 + 语义相似性
            return await self._hybrid_reorder(chunks, query)

        return chunks

    async def _semantic_reorder(self, chunks: List[DocumentChunk], query: str) -> List[DocumentChunk]:
        """语义相似性排序。"""
        try:
            # 简化实现：基于内容长度和分数的综合排序
            for chunk in chunks:
                # 计算语义分数
                content_words = len(chunk.content.split())
                semantic_score = chunk.score * 0.7 + min(content_words / 200, 1.0) * 0.3
                chunk.metadata["semantic_score"] = semantic_score

            return sorted(chunks, key=lambda x: x.metadata.get("semantic_score", 0), reverse=True)

        except Exception as e:
            self.logger.error(f"语义排序失败: {e}")
            return chunks

    async def _hybrid_reorder(self, chunks: List[DocumentChunk], query: str) -> List[DocumentChunk]:
        """混合排序。"""
        # 结合相关性、时间、内容质量等因素
        for chunk in chunks:
            # 计算综合分数
            relevance_score = chunk.score * 0.6

            # 内容质量分数（基于长度和结构）
            content_quality = min(len(chunk.content) / 100, 1.0) * 0.2

            # 时间新鲜度分数
            time_score = 0.2  # 简化实现

            hybrid_score = relevance_score + content_quality + time_score
            chunk.metadata["hybrid_score"] = hybrid_score

        return sorted(chunks, key=lambda x: x.metadata.get("hybrid_score", 0), reverse=True)

    async def _compress_chunks(
        self,
        chunks: List[DocumentChunk],
        config: ContextBuilderConfig
    ) -> List[DocumentChunk]:
        """压缩文档块。"""
        if config.compression_mode == ContextCompressionMode.NONE:
            return chunks

        self.stats["compression_count"] += 1

        if config.compression_mode == ContextCompressionMode.TRUNCATION:
            return await self._truncate_chunks(chunks, config)

        elif config.compression_mode == ContextCompressionMode.SUMMARIZATION:
            return await self._summarize_chunks(chunks, config)

        elif config.compression_mode == ContextCompressionMode.RELEVANCE_FILTERING:
            return await self._filter_by_relevance(chunks, config)

        elif config.compression_mode == ContextCompressionMode.SEMANTIC_COMPRESSION:
            return await self._semantic_compress_chunks(chunks, config)

        return chunks

    async def _truncate_chunks(
        self,
        chunks: List[DocumentChunk],
        config: ContextBuilderConfig
    ) -> List[DocumentChunk]:
        """截断压缩。"""
        total_length = 0
        compressed_chunks = []

        for chunk in chunks:
            chunk_length = len(chunk.content)
            if total_length + chunk_length <= config.max_context_length:
                compressed_chunks.append(chunk)
                total_length += chunk_length
            else:
                # 部分截断最后一个块
                remaining_length = config.max_context_length - total_length
                if remaining_length > 100:  # 至少保留100字符
                    truncated_content = chunk.content[:remaining_length - 3] + "..."
                    truncated_chunk = DocumentChunk(
                        chunk_id=chunk.chunk_id,
                        content=truncated_content,
                        document_id=chunk.document_id,
                        chunk_index=chunk.chunk_index,
                        title=chunk.title,
                        score=chunk.score,
                        metadata={**chunk.metadata, "truncated": True}
                    )
                    compressed_chunks.append(truncated_chunk)
                break

        return compressed_chunks

    async def _summarize_chunks(
        self,
        chunks: List[DocumentChunk],
        config: ContextBuilderConfig
    ) -> List[DocumentChunk]:
        """摘要压缩。"""
        # 简化实现：提取每段的前几句话作为摘要
        summarized_chunks = []
        for chunk in chunks:
            sentences = re.split(r'[.!?。！？]', chunk.content)
            summary = sentences[0] + "." if sentences else chunk.content[:100]

            if len(summary) > len(chunk.content) * config.compression_ratio:
                summary = chunk.content[:int(len(chunk.content) * config.compression_ratio)] + "..."

            summarized_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                content=summary,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                title=chunk.title,
                score=chunk.score,
                metadata={**chunk.metadata, "summarized": True}
            )
            summarized_chunks.append(summarized_chunk)

        return summarized_chunks

    async def _filter_by_relevance(
        self,
        chunks: List[DocumentChunk],
        config: ContextBuilderConfig
    ) -> List[DocumentChunk]:
        """基于相关性过滤。"""
        filtered_chunks = [
            chunk for chunk in chunks
            if chunk.score >= config.min_relevance_score
        ]

        # 如果过滤后太少了，降低阈值
        if len(filtered_chunks) < 3:
            filtered_chunks = chunks[:5]  # 至少保留5个块

        return filtered_chunks

    async def _semantic_compress_chunks(
        self,
        chunks: List[DocumentChunk],
        config: ContextBuilderConfig
    ) -> List[DocumentChunk]:
        """语义压缩。"""
        # 移除重复和相似的块
        unique_chunks = []
        seen_content = set()

        for chunk in chunks:
            content_hash = hash(chunk.content[:200])  # 使用前200字符的哈希
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)

        # 基于重要性选择块
        if len(unique_chunks) > config.max_chunks_per_context:
            # 按分数排序并选择top块
            unique_chunks.sort(key=lambda x: x.score, reverse=True)
            unique_chunks = unique_chunks[:config.max_chunks_per_context]

        return unique_chunks

    async def _format_chunks(
        self,
        chunks: List[DocumentChunk],
        config: ContextBuilderConfig
    ) -> str:
        """格式化文档块。"""
        if not chunks:
            return ""

        formatted_parts = []

        for i, chunk in enumerate(chunks):
            # 构建块标题
            title = chunk.title or f"文档片段 {i+1}"
            if config.include_metadata and chunk.metadata.get("source"):
                title += f" (来源: {chunk.metadata['source']})"

            # 格式化内容
            content = chunk.content.strip()

            # 添加块标题
            formatted_parts.append(f"### {title}")
            formatted_parts.append(content)

            # 添加分隔符
            if i < len(chunks) - 1:
                formatted_parts.append(config.chunk_separator)

        formatted_context = "\n".join(formatted_parts)

        # 检查长度限制
        if len(formatted_context) > config.max_context_length:
            # 截断
            truncated = formatted_context[:config.max_context_length - 50] + "\n...[内容已截断]"
            formatted_context = truncated

        return formatted_context

    async def _create_context_sections(
        self,
        chunks: List[DocumentChunk],
        query: str,
        config: ContextBuilderConfig
    ) -> List[ContextSection]:
        """创建上下文段落。"""
        sections = []

        # 按文档ID分组
        doc_groups = {}
        for chunk in chunks:
            doc_id = chunk.document_id
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(chunk)

        # 为每个文档组创建段落
        for doc_id, doc_chunks in doc_groups.items():
            if not doc_chunks:
                continue

            # 使用第一个块的标题
            title = doc_chunks[0].title or f"文档 {doc_id[:8]}"

            # 合并内容
            content_parts = []
            for chunk in doc_chunks:
                content_parts.append(chunk.content)

            content = config.chunk_separator.join(content_parts)

            # 计算平均相关性分数
            avg_score = sum(chunk.score for chunk in doc_chunks) / len(doc_chunks)

            section = ContextSection(
                title=title,
                content=content,
                chunks=doc_chunks,
                relevance_score=avg_score,
                metadata={
                    "document_id": doc_id,
                    "chunk_count": len(doc_chunks)
                }
            )
            sections.append(section)

        # 按相关性排序
        sections.sort(key=lambda x: x.relevance_score, reverse=True)
        return sections

    async def _process_chunk_map(
        self,
        chunk: DocumentChunk,
        query: str,
        config: ContextBuilderConfig
    ) -> str:
        """Map阶段处理单个文档块。"""
        # 简化实现：为每个块添加查询相关性说明
        return f"相关信息：{chunk.content}\n相关性分数：{chunk.score:.2f}"

    async def _reduce_mapped_results(
        self,
        mapped_results: List[str],
        config: ContextBuilderConfig
    ) -> str:
        """Reduce阶段合并处理结果。"""
        # 简单合并
        combined = "\n\n".join(mapped_results)

        # 检查长度限制
        if len(combined) > config.max_context_length:
            combined = combined[:config.max_context_length - 50] + "\n...[内容已截断]"

        return combined

    async def _refine_context_with_chunk(
        self,
        existing_context: str,
        new_chunk: DocumentChunk,
        query: str,
        config: ContextBuilderConfig
    ) -> str:
        """用新文档块精炼现有上下文。"""
        # 简化实现：检查新块是否提供新信息
        if len(existing_context) + len(new_chunk.content) <= config.max_context_length:
            # 直接添加
            return existing_context + "\n\n" + f"补充信息：{new_chunk.content}"
        else:
            # 精炼现有内容，为新信息腾出空间
            max_existing_length = config.max_context_length - len(new_chunk.content) - 100
            if max_existing_length > 200:
                refined_existing = existing_context[:max_existing_length] + "\n...[部分内容省略]"
                return refined_existing + "\n\n" + f"补充信息：{new_chunk.content}"
            else:
                # 只保留新信息
                return f"关键信息：{new_chunk.content}"

    async def _extract_key_chunks(
        self,
        chunks: List[DocumentChunk],
        query: str,
        config: ContextBuilderConfig
    ) -> List[DocumentChunk]:
        """提取关键文档块。"""
        # 基于分数和长度提取关键块
        key_chunks = []

        # 优先选择高分数的块
        high_score_chunks = [c for c in chunks if c.score > 0.8]
        key_chunks.extend(high_score_chunks)

        # 补充中等分数的块
        if len(key_chunks) < config.max_chunks_per_context:
            medium_score_chunks = [c for c in chunks if 0.5 <= c.score <= 0.8]
            key_chunks.extend(medium_score_chunks[:config.max_chunks_per_context - len(key_chunks)])

        return key_chunks[:config.max_chunks_per_context]

    async def _compress_content(
        self,
        chunks: List[DocumentChunk],
        config: ContextBuilderConfig
    ) -> str:
        """压缩内容。"""
        # 提取关键句子和概念
        key_sentences = []
        for chunk in chunks:
            sentences = re.split(r'[.!?。！？]', chunk.content)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and sentence not in key_sentences:
                    key_sentences.append(sentence)

        # 按重要性排序
        key_sentences.sort(key=lambda x: len(x), reverse=True)

        # 组合压缩内容
        compressed = "。".join(key_sentences[:10])  # 最多10个关键句

        if len(compressed) > config.max_context_length:
            compressed = compressed[:config.max_context_length - 3] + "..."

        return compressed

    async def _postprocess_context(
        self,
        context: GenerationContext,
        query: str,
        config: ContextBuilderConfig
    ) -> GenerationContext:
        """后处理上下文。"""
        # 检查最小长度要求
        if len(context.formatted_context) < config.min_context_length:
            # 添加默认说明
            context.formatted_context += "\n\n注意：基于查询内容的相关信息有限，回答时请谨慎处理。"

        # 增强上下文连贯性
        if config.enable_context_enhancement:
            context.formatted_context = await self._enhance_context_coherence(
                context.formatted_context, query
            )

        return context

    async def _enhance_context_coherence(self, context: str, query: str) -> str:
        """增强上下文连贯性。"""
        # 简化实现：添加过渡词和连接
        context = re.sub(r'\n\n+', '\n\n', context)  # 合并多个换行

        # 添加适当的过渡
        parts = context.split('\n\n')
        enhanced_parts = []

        for i, part in enumerate(parts):
            if i > 0 and not part.startswith(('此外', '另外', '同时', '因此', '所以')):
                # 添加过渡词
                enhanced_parts.append('此外，' + part)
            else:
                enhanced_parts.append(part)

        return '\n\n'.join(enhanced_parts)

    def _create_fallback_context(self, chunks: List[DocumentChunk], query: str) -> GenerationContext:
        """创建后备上下文。"""
        if not chunks:
            formatted_context = "未找到相关信息，请基于一般知识回答。"
        else:
            # 使用前3个最相关的块
            top_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)[:3]
            formatted_context = "\n\n".join([chunk.content for chunk in top_chunks])

        return GenerationContext(
            context_chunks=chunks[:3],
            formatted_context=formatted_context,
            context_length=len(formatted_context),
            truncation_info={"fallback": True}
        )

    async def build_prompt_context(
        self,
        context: GenerationContext,
        query: str,
        template_name: str = "qa_context",
        rag_query: Optional[RAGQuery] = None
    ) -> str:
        """构建提示词上下文。"""
        template = self.templates.get(template_name, self.templates["qa_context"])

        # 准备模板变量
        template_vars = {
            "context": context.formatted_context,
            "query": query
        }

        # 添加对话历史
        if rag_query and rag_query.conversation_history:
            history_text = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in rag_query.conversation_history[-5:]  # 最近5轮对话
            ])
            template_vars["history"] = history_text

        # 替换模板变量
        try:
            formatted_prompt = template.format(**template_vars)
        except KeyError as e:
            self.logger.warning(f"模板变量缺失: {e}")
            formatted_prompt = template.format(context=context.formatted_context, query=query)

        return formatted_prompt

    async def evaluate_context_quality(
        self,
        context: GenerationContext,
        query: str
    ) -> Dict[str, float]:
        """评估上下文质量。"""
        quality_metrics = {}

        # 长度分数
        ideal_length = 2000
        length_score = min(context.context_length / ideal_length, 1.0)
        quality_metrics["length_score"] = length_score

        # 相关性分数
        if context.context_chunks:
            avg_relevance = sum(chunk.score for chunk in context.context_chunks) / len(context.context_chunks)
            quality_metrics["relevance_score"] = avg_relevance
        else:
            quality_metrics["relevance_score"] = 0.0

        # 多样性分数
        if len(context.context_chunks) > 1:
            doc_ids = set(chunk.document_id for chunk in context.context_chunks)
            diversity_score = len(doc_ids) / len(context.context_chunks)
            quality_metrics["diversity_score"] = diversity_score
        else:
            quality_metrics["diversity_score"] = 0.0

        # 完整性分数
        if context.context_chunks:
            completeness = min(len(context.context_chunks) / 5, 1.0)  # 理想5个块
            quality_metrics["completeness_score"] = completeness
        else:
            quality_metrics["completeness_score"] = 0.0

        # 总体质量分数
        quality_metrics["overall_score"] = sum(quality_metrics.values()) / len(quality_metrics)

        return quality_metrics

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return {
            **self.stats,
            "available_templates": list(self.templates.keys()),
            "supported_strategies": [strategy.value for strategy in ContextStrategy],
            "supported_compression_modes": [mode.value for mode in ContextCompressionMode],
            "supported_reordering_modes": [mode.value for mode in ContextReorderingMode]
        }

    def add_template(self, name: str, template: str) -> None:
        """添加模板。"""
        self.templates[name] = template
        self.logger.info(f"添加模板: {name}")

    def remove_template(self, name: str) -> bool:
        """移除模板。"""
        if name in self.templates:
            del self.templates[name]
            self.logger.info(f"移除模板: {name}")
            return True
        return False