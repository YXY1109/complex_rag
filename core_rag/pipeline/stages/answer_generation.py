"""
答案生成和引用标注

实现基于上下文的答案生成、引用生成和来源标注。
"""

import asyncio
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass

from ..interfaces.pipeline_interface import (
    Answer,
    Context,
    GenerationStrategy,
    QueryRequest,
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.pipeline.stages.answer_generation")


@dataclass
class GenerationPrompt:
    """生成提示"""
    system_prompt: str
    user_prompt: str
    include_citations: bool = True
    max_tokens: int = 1000


class AnswerGenerator:
    """
    答案生成器

    负责基于上下文生成答案并添加引用标注。
    """

    def __init__(self, config: Dict[str, Any], llm_service=None):
        """
        初始化答案生成器

        Args:
            config: 配置参数
            llm_service: LLM服务实例
        """
        self.config = config
        self.llm_service = llm_service
        self.generation_model = config.get("generation_model", "default")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        self.enable_citation_generation = config.get("enable_citation_generation", True)
        self.enable_source_attribution = config.get("enable_source_attribution", True)

        # 生成策略提示模板
        self.prompt_templates = self._load_prompt_templates()

        # 引用模式
        self.citation_pattern = r'\[(\w+)\]'
        self.citation_replacement = r'[citation:\1]'

        self._initialized = False

    def _load_prompt_templates(self) -> Dict[GenerationStrategy, str]:
        """加载生成策略的提示模板"""
        return {
            GenerationStrategy.CONCISE: """
你是一个专业的AI助手。请基于提供的上下文，给出简洁、准确的回答。

要求：
- 回答要简洁明了，直击要点
- 基于上下文信息，不要添加无关内容
- 如果上下文不足，明确说明
- 在相关内容后添加引用标记 [citation:X]

上下文：
{context}

问题：{query}

简洁回答：""",

            GenerationStrategy.DETAILED: """
你是一个专业的AI助手。请基于提供的上下文，给出详细、全面的回答。

要求：
- 回答要详细全面，涵盖重要信息
- 基于上下文信息，可以进行适当推理
- 结构清晰，逻辑性强
- 在相关内容后添加引用标记 [citation:X]
- 如果上下文不足，明确说明局限性

上下文：
{context}

问题：{query}

详细回答：""",

            GenerationStrategy.STEP_BY_STEP: """
你是一个专业的AI助手。请基于提供的上下文，按步骤回答问题。

要求：
- 将回答分解为清晰的步骤
- 每个步骤都要基于上下文
- 逻辑递进，易于理解
- 在相关步骤后添加引用标记 [citation:X]
- 如果某些步骤缺乏信息，明确说明

上下文：
{context}

问题：{query}

分步回答：""",

            GenerationStrategy.STRUCTURED: """
你是一个专业的AI助手。请基于提供的上下文，给出结构化的回答。

要求：
- 使用标题、列表等结构化格式
- 信息组织清晰，层次分明
- 重要信息突出显示
- 在相关内容后添加引用标记 [citation:X]
- 包含要点总结和详细说明

上下文：
{context}

问题：{query}

结构化回答：""",

            GenerationStrategy.CONVERSATIONAL: """
你是一个友好的AI助手。请基于提供的上下文，以对话的方式回答问题。

要求：
- 语气自然友好，像真人对话
- 回答流畅易懂
- 可以适当加入解释和例子
- 在相关内容后添加引用标记 [citation:X]
- 保持专业性的同时亲和力强

上下文：
{context}

问题：{query}

对话式回答：""",
        }

    async def initialize(self) -> bool:
        """
        初始化答案生成器

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "初始化答案生成器",
                extra={
                    "generation_model": self.generation_model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "enable_citations": self.enable_citation_generation,
                }
            )

            # 初始化LLM服务
            if not self.llm_service:
                structured_logger.warning("未提供LLM服务，将使用模拟实现")

            self._initialized = True
            structured_logger.info("答案生成器初始化成功")
            return True

        except Exception as e:
            structured_logger.error(f"答案生成器初始化失败: {e}")
            return False

    async def generate_answer(
        self,
        query: str,
        context: Context,
        generation_strategy: GenerationStrategy = GenerationStrategy.DETAILED,
        enable_citations: bool = True
    ) -> Answer:
        """
        生成答案

        Args:
            query: 查询问题
            context: 上下文
            generation_strategy: 生成策略
            enable_citations: 是否启用引用

        Returns:
            Answer: 生成的答案
        """
        if not self._initialized:
            raise RuntimeError("答案生成器未初始化")

        start_time = time.time()

        try:
            structured_logger.info(
                "开始生成答案",
                extra={
                    "query_length": len(query),
                    "context_length": context.total_length,
                    "generation_strategy": generation_strategy.value,
                    "enable_citations": enable_citations,
                }
            )

            # 构建生成提示
            prompt = self._build_generation_prompt(query, context, generation_strategy)

            # 调用LLM生成答案
            if self.llm_service:
                raw_answer = await self._call_llm(prompt, generation_strategy)
                token_usage = await self._estimate_token_usage(prompt, raw_answer)
            else:
                # 模拟实现
                raw_answer = await self._simulate_answer_generation(query, context, generation_strategy)
                token_usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": len(raw_answer.split()), "total_tokens": len(prompt.split()) + len(raw_answer.split())}

            # 后处理答案
            processed_answer = await self._post_process_answer(raw_answer, enable_citations)

            # 生成引用信息
            citations = []
            sources = []
            if enable_citations and self.enable_citation_generation:
                citations = await self._extract_citations(processed_answer, context)
                sources = await self._generate_sources(context.documents, citations)

            # 计算置信度
            confidence = self._calculate_answer_confidence(
                processed_answer, context, citations
            )

            generation_time = (time.time() - start_time) * 1000

            result = Answer(
                content=processed_answer,
                citations=citations,
                sources=sources,
                confidence=confidence,
                generation_time_ms=generation_time,
                token_usage=token_usage,
                metadata={
                    "generation_strategy": generation_strategy.value,
                    "model": self.generation_model,
                    "temperature": self.temperature,
                    "context_documents_count": len(context.documents),
                    "citations_count": len(citations),
                },
            )

            structured_logger.info(
                "答案生成完成",
                extra={
                    "answer_length": len(processed_answer),
                    "citations_count": len(citations),
                    "confidence": confidence,
                    "generation_time_ms": generation_time,
                    "token_usage": token_usage,
                }
            )

            return result

        except Exception as e:
            structured_logger.error(f"答案生成失败: {e}")
            raise Exception(f"Answer generation failed: {e}")

    def _build_generation_prompt(
        self,
        query: str,
        context: Context,
        generation_strategy: GenerationStrategy
    ) -> str:
        """构建生成提示"""
        template = self.prompt_templates.get(generation_strategy, self.prompt_templates[GenerationStrategy.DETAILED])

        # 格式化上下文
        formatted_context = context.formatted_context

        # 替换模板变量
        prompt = template.format(
            context=formatted_context,
            query=query
        )

        return prompt

    async def _call_llm(self, prompt: str, generation_strategy: GenerationStrategy) -> str:
        """调用LLM生成答案"""
        try:
            # 根据策略调整参数
            temperature = self.temperature
            if generation_strategy == GenerationStrategy.CONCISE:
                temperature = 0.3  # 更保守
            elif generation_strategy == GenerationStrategy.CREATIVE:
                temperature = 0.9  # 更有创造性

            # 调用LLM服务
            if hasattr(self.llm_service, 'generate_text'):
                response = await self.llm_service.generate_text(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    model=self.generation_model
                )
                return response.content
            elif hasattr(self.llm_service, 'chat'):
                messages = [
                    {"role": "system", "content": "你是一个专业的AI助手。"},
                    {"role": "user", "content": prompt}
                ]
                response = await self.llm_service.chat(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    model=self.generation_model
                )
                return response.content
            else:
                raise AttributeError("LLM服务不支持文本生成方法")

        except Exception as e:
            structured_logger.error(f"调用LLM失败: {e}")
            raise Exception(f"LLM call failed: {e}")

    async def _simulate_answer_generation(
        self,
        query: str,
        context: Context,
        generation_strategy: GenerationStrategy
    ) -> str:
        """模拟答案生成（用于测试）"""
        # 这是一个简化的模拟实现
        if not context.documents:
            return "抱歉，我没有找到相关信息来回答您的问题。"

        # 基于上下文生成简单答案
        answer_parts = [f"根据提供的信息，我来回答您关于 '{query}' 的问题。\n\n"]

        for i, doc in enumerate(context.documents[:3]):  # 最多使用3个文档
            content = doc.content[:200]  # 限制长度
            answer_parts.append(f"[citation:{doc.citation_id or str(i+1)}] {content}\n")

        answer_parts.append("\n以上是基于可用信息给出的回答。")

        return "".join(answer_parts)

    async def _post_process_answer(self, raw_answer: str, enable_citations: bool) -> str:
        """后处理答案"""
        # 清理多余的换行和空格
        processed_answer = re.sub(r'\n\s*\n\s*\n', '\n\n', raw_answer)
        processed_answer = re.sub(r' +', ' ', processed_answer)

        # 确保引用格式正确
        if enable_citations:
            # 标准化引用格式
            processed_answer = re.sub(r'\[\s*(\w+)\s*\]', r'[citation:\1]', processed_answer)

        return processed_answer.strip()

    async def _extract_citations(self, answer: str, context: Context) -> List[Dict[str, Any]]:
        """提取引用信息"""
        if not self.enable_citation_generation:
            return []

        citations = []

        # 查找所有引用标记
        citation_matches = re.finditer(r'\[citation:(\w+)\]', answer)

        citation_ids = set()
        for match in citation_matches:
            citation_id = match.group(1)
            citation_ids.add(citation_id)

        # 为每个引用ID创建引用信息
        for citation_id in citation_ids:
            # 查找对应的文档
            cited_doc = None
            for doc in context.documents:
                if doc.citation_id == citation_id:
                    cited_doc = doc
                    break

            if cited_doc:
                citation_info = {
                    "id": citation_id,
                    "document_id": cited_doc.id,
                    "title": cited_doc.title,
                    "source": cited_doc.source,
                    "url": cited_doc.url,
                    "relevance_score": cited_doc.relevance_score,
                }
                citations.append(citation_info)

        return citations

    async def _generate_sources(
        self,
        documents: List[ContextDocument],
        citations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """生成来源信息"""
        if not self.enable_source_attribution:
            return []

        sources = []
        cited_doc_ids = {citation["document_id"] for citation in citations}

        for doc in documents:
            if doc.id in cited_doc_ids:
                source_info = {
                    "id": doc.id,
                    "title": doc.title,
                    "source": doc.source,
                    "url": doc.url,
                    "relevance_score": doc.relevance_score,
                    "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                }
                sources.append(source_info)

        return sources

    def _calculate_answer_confidence(
        self,
        answer: str,
        context: Context,
        citations: List[Dict[str, Any]]
    ) -> float:
        """计算答案置信度"""
        confidence = 0.5  # 基础置信度

        # 基于引用数量
        citation_count = len(citations)
        if citation_count > 0:
            confidence += min(0.3, citation_count * 0.1)

        # 基于上下文质量
        if context.relevance_score > 0.7:
            confidence += 0.2
        elif context.relevance_score > 0.5:
            confidence += 0.1

        # 基于答案长度
        answer_length = len(answer.split())
        if 20 <= answer_length <= 500:
            confidence += 0.1
        elif answer_length > 500:
            confidence += 0.05

        return min(1.0, confidence)

    async def _estimate_token_usage(self, prompt: str, answer: str) -> Dict[str, int]:
        """估算Token使用量"""
        # 简单的Token估算（实际应用中应使用精确的计算方法）
        prompt_tokens = len(prompt.split())
        completion_tokens = len(answer.split())
        total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    async def generate_answer_stream(
        self,
        query: str,
        context: Context,
        generation_strategy: GenerationStrategy = GenerationStrategy.DETAILED,
        enable_citations: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        流式生成答案

        Args:
            query: 查询问题
            context: 上下文
            generation_strategy: 生成策略
            enable_citations: 是否启用引用

        Yields:
            str: 流式生成的答案片段
        """
        if not self._initialized:
            raise RuntimeError("答案生成器未初始化")

        try:
            # 构建生成提示
            prompt = self._build_generation_prompt(query, context, generation_strategy)

            # 如果LLM服务支持流式生成
            if self.llm_service and hasattr(self.llm_service, 'generate_text_stream'):
                async for chunk in self.llm_service.generate_text_stream(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    model=self.generation_model
                ):
                    yield chunk
            else:
                # 不支持流式时，生成完整答案后分块返回
                full_answer = await self.generate_answer(query, context, generation_strategy, enable_citations)
                chunk_size = 50  # 每次返回50个字符

                for i in range(0, len(full_answer.content), chunk_size):
                    chunk = full_answer.content[i:i + chunk_size]
                    yield chunk

                    # 模拟流式延迟
                    await asyncio.sleep(0.1)

        except Exception as e:
            structured_logger.error(f"流式答案生成失败: {e}")
            yield f"生成答案时出错: {str(e)}"

    async def batch_generate_answers(
        self,
        queries: List[str],
        contexts: List[Context],
        generation_strategy: GenerationStrategy = GenerationStrategy.DETAILED,
        enable_citations: bool = True
    ) -> List[Answer]:
        """批量生成答案"""
        if not self._initialized:
            raise RuntimeError("答案生成器未初始化")

        if len(queries) != len(contexts):
            raise ValueError("查询和上下文数量不匹配")

        try:
            structured_logger.info(f"开始批量生成答案，数量: {len(queries)}")

            # 并行处理
            tasks = [
                self.generate_answer(query, context, generation_strategy, enable_citations)
                for query, context in zip(queries, contexts)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理异常
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    structured_logger.error(f"答案 {i} 生成失败: {result}")
                    # 创建空答案
                    valid_results.append(Answer(
                        content="抱歉，生成答案时出现问题。",
                        confidence=0.0,
                        generation_time_ms=0.0,
                        token_usage={},
                    ))
                else:
                    valid_results.append(result)

            structured_logger.info(f"批量答案生成完成，成功处理 {len(valid_results)} 个")
            return valid_results

        except Exception as e:
            structured_logger.error(f"批量答案生成失败: {e}")
            raise Exception(f"Batch answer generation failed: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "initialized": self._initialized,
            "generation_model": self.generation_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enable_citation_generation": self.enable_citation_generation,
            "enable_source_attribution": self.enable_source_attribution,
            "supported_strategies": [strategy.value for strategy in self.prompt_templates.keys()],
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 创建测试上下文
            from ..interfaces.pipeline_interface import ContextDocument
            test_context = Context(
                documents=[
                    ContextDocument(
                        id="test_doc",
                        content="这是一个测试文档，用于验证答案生成功能。",
                        score=0.8,
                        citation_id="doc1",
                    )
                ],
                formatted_context="[doc1] 这是一个测试文档，用于验证答案生成功能。",
                total_length=15,
                relevance_score=0.8,
            )

            result = await self.generate_answer(
                "这个文档是关于什么的？",
                test_context,
                GenerationStrategy.CONCISE
            )

            return {
                "status": "healthy",
                "initialized": self._initialized,
                "test_generation_time_ms": result.generation_time_ms,
                "test_confidence": result.confidence,
                "test_token_usage": result.token_usage,
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
            structured_logger.info("答案生成器清理完成")

        except Exception as e:
            structured_logger.error(f"答案生成器清理失败: {e}")