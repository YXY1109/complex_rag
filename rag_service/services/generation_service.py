"""
生成服务

基于RAGFlow架构的智能生成服务，
支持多种生成模式、模型集成、流式输出等功能。
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum

from ..interfaces.rag_interface import (
    GenerationInterface, GenerationResult, GenerationContext,
    GenerationMode, RAGQuery, GenerationException
)


class ModelProvider(Enum):
    """模型提供商。"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CUSTOM = "custom"


class OutputFormat(Enum):
    """输出格式。"""

    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class GenerationConfig:
    """生成配置。"""

    # 基础配置
    model_name: str = "gpt-3.5-turbo"
    model_provider: ModelProvider = ModelProvider.OPENAI
    generation_mode: GenerationMode = GenerationMode.RAG
    output_format: OutputFormat = OutputFormat.TEXT

    # 参数配置
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)

    # 高级配置
    enable_streaming: bool = False
    enable_function_calling: bool = False
    enable_json_mode: bool = False
    enable_caching: bool = True

    # 超时和重试
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0

    # 系统提示词
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None

    # 函数调用配置
    functions: List[Dict[str, Any]] = field(default_factory=list)
    function_call: Optional[Union[str, Dict[str, Any]]] = None


@dataclass
class ModelConfig:
    """模型配置。"""

    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    organization: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingChunk:
    """流式输出块。"""

    content: str
    delta: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    timestamp: datetime = field(default_factory=datetime.now)


class GenerationService(GenerationInterface):
    """生成服务。"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化生成服务。

        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 模型配置
        self.models: Dict[str, ModelConfig] = {}
        self.default_model: Optional[str] = None

        # 生成配置
        self.default_generation_config = GenerationConfig(**config.get("generation", {}))

        # 提示词模板
        self.prompts = config.get("prompts", {})
        self._load_default_prompts()

        # 缓存
        self.response_cache: Dict[str, GenerationResult] = {}
        self.cache_ttl = config.get("cache_ttl", 3600)

        # 统计信息
        self.stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "average_generation_time": 0.0,
            "model_usage": {},
            "streaming_generations": 0,
            "total_tokens": 0
        }

        # 初始化模型
        self._init_models()

    def _init_models(self) -> None:
        """初始化模型配置。"""
        models_config = self.config.get("models", {})

        for model_name, model_config in models_config.items():
            model = ModelConfig(
                provider=ModelProvider(model_config.get("provider", "openai")),
                model_name=model_name,
                **{k: v for k, v in model_config.items() if k != "provider"}
            )
            self.models[model_name] = model

            # 设置默认模型
            if not self.default_model or model_config.get("default", False):
                self.default_model = model_name

        if not self.default_model and self.models:
            self.default_model = list(self.models.keys())[0]

        self.logger.info(f"初始化了 {len(self.models)} 个生成模型，默认模型: {self.default_model}")

    def _load_default_prompts(self) -> None:
        """加载默认提示词。"""
        self.prompts.update({
            "qa_system": """
你是一个专业的问答助手。请基于提供的上下文信息准确回答用户的问题。

回答要求：
1. 严格基于提供的上下文信息
2. 如果上下文信息不足，请明确说明
3. 回答要准确、简洁、易懂
4. 使用中文回答

""",
            "conversation_system": """
你是一个智能对话助手。请基于对话历史和相关背景信息，自然地回应用户的查询。

对话要求：
1. 保持对话的连贯性和上下文关联
2. 基于提供的背景信息
3. 回答要友好、专业、准确
4. 使用中文回答

""",
            "summarization_system": """
你是一个专业的文档总结助手。请对提供的信息进行准确、简洁的总结。

总结要求：
1. 提取关键信息和要点
2. 保持逻辑清晰
3. 简洁明了，避免冗余
4. 使用中文总结

""",
            "creative_system": """
你是一个创意写作助手。请基于提供的信息，创作出有创意的内容。

创作要求：
1. 发挥创意，不拘泥于原文
2. 保持内容的逻辑性和可读性
3. 语言生动有趣
4. 使用中文创作

"""
        })

    async def generate(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        config: Optional[GenerationConfig] = None,
        rag_query: Optional[RAGQuery] = None
    ) -> GenerationResult:
        """
        生成回答。

        Args:
            query: 查询字符串
            context: 上下文
            conversation_history: 对话历史
            config: 生成配置
            rag_query: RAG查询对象

        Returns:
            GenerationResult: 生成结果
        """
        start_time = datetime.now()
        self.stats["total_generations"] += 1

        try:
            # 使用配置
            generation_config = config or self.default_generation_config

            # 选择模型
            model_name = generation_config.model_name or self.default_model
            model_config = self.models.get(model_name)
            if not model_config:
                raise GenerationException(f"模型 {model_name} 不存在")

            # 构建提示词
            messages = await self._build_messages(
                query, context, conversation_history, generation_config, rag_query
            )

            # 检查缓存
            if generation_config.enable_caching:
                cache_key = self._get_cache_key(messages, generation_config)
                if cache_key in self.response_cache:
                    cached_result = self.response_cache[cache_key]
                    self.logger.info(f"生成缓存命中: {query[:50]}...")
                    return cached_result

            # 执行生成
            if generation_config.enable_streaming:
                # 流式生成（收集完整结果）
                full_content = ""
                chunk_count = 0

                async for chunk in self._generate_stream(
                    messages, generation_config, model_config
                ):
                    full_content += chunk.delta
                    chunk_count += 1

                    if chunk.finish_reason:
                        break

                answer = full_content
                self.stats["streaming_generations"] += 1
            else:
                # 非流式生成
                answer = await self._generate_once(
                    messages, generation_config, model_config
                )

            # 处理输出格式
            answer = await self._process_output_format(answer, generation_config)

            # 计算token使用量（简化）
            token_usage = await self._estimate_token_usage(messages, answer)

            # 创建生成结果
            generation_time = (datetime.now() - start_time).total_seconds()
            result = GenerationResult(
                query_id=rag_query.query_id if rag_query else str(hash(query + str(start_time))),
                answer=answer,
                context=GenerationContext(
                    context_chunks=[],  # 这里可以传入实际的上下文块
                    formatted_context=context,
                    context_length=len(context)
                ),
                generation_time=generation_time,
                token_usage=token_usage,
                model_info={
                    "model_name": model_name,
                    "provider": model_config.provider.value,
                    "generation_mode": generation_config.generation_mode.value
                },
                metadata={
                    "config": generation_config.__dict__,
                    "cache_hit": False
                }
            )

            # 缓存结果
            if generation_config.enable_caching:
                self.response_cache[cache_key] = result

            # 更新统计
            self.stats["successful_generations"] += 1
            self.stats["total_tokens"] += sum(token_usage.values())
            self.stats["average_generation_time"] = (
                (self.stats["average_generation_time"] * (self.stats["total_generations"] - 1) + generation_time) /
                self.stats["total_generations"]
            )

            if model_name not in self.stats["model_usage"]:
                self.stats["model_usage"][model_name] = 0
            self.stats["model_usage"][model_name] += 1

            self.logger.info(f"生成完成，耗时 {generation_time:.3f}s，模型: {model_name}")

            return result

        except Exception as e:
            self.stats["failed_generations"] += 1
            self.logger.error(f"生成失败: {e}")
            raise GenerationException(f"生成失败: {str(e)}")

    async def generate_stream(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        config: Optional[GenerationConfig] = None,
        rag_query: Optional[RAGQuery] = None
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        流式生成。

        Args:
            query: 查询字符串
            context: 上下文
            conversation_history: 对话历史
            config: 生成配置
            rag_query: RAG查询对象

        Yields:
            StreamingChunk: 流式输出块
        """
        try:
            # 使用配置
            generation_config = config or self.default_generation_config
            generation_config.enable_streaming = True

            # 选择模型
            model_name = generation_config.model_name or self.default_model
            model_config = self.models.get(model_name)
            if not model_config:
                raise GenerationException(f"模型 {model_name} 不存在")

            # 构建提示词
            messages = await self._build_messages(
                query, context, conversation_history, generation_config, rag_query
            )

            # 流式生成
            async for chunk in self._generate_stream(messages, generation_config, model_config):
                yield chunk

        except Exception as e:
            self.logger.error(f"流式生成失败: {e}")
            yield StreamingChunk(
                content="",
                delta="[生成过程中发生错误]",
                finish_reason="error"
            )

    async def _build_messages(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]],
        config: GenerationConfig,
        rag_query: Optional[RAGQuery]
    ) -> List[Dict[str, str]]:
        """构建消息列表。"""
        messages = []

        # 系统提示词
        system_prompt = config.system_prompt
        if not system_prompt:
            # 根据生成模式选择系统提示词
            if config.generation_mode == GenerationMode.RAG:
                system_prompt = self.prompts.get("qa_system", "")
            elif config.generation_mode == GenerationMode.CHAIN_OF_THOUGHT:
                system_prompt = self.prompts.get("creative_system", "")
            else:
                system_prompt = "你是一个有用的AI助手。"

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 对话历史
        if conversation_history:
            for msg in conversation_history[-10:]:  # 限制历史记录数量
                if msg.get("role") and msg.get("content"):
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        # 构建用户提示词
        if config.user_prompt_template:
            user_prompt = config.user_prompt_template.format(
                context=context,
                query=query
            )
        else:
            user_prompt = await self._build_default_user_prompt(
                context, query, config, rag_query
            )

        messages.append({"role": "user", "content": user_prompt})

        return messages

    async def _build_default_user_prompt(
        self,
        context: str,
        query: str,
        config: GenerationConfig,
        rag_query: Optional[RAGQuery]
    ) -> str:
        """构建默认用户提示词。"""
        if config.generation_mode == GenerationMode.RAG:
            return f"""
请基于以下上下文信息回答问题：

上下文：
{context}

问题：{query}

请提供准确、简洁的回答：
"""
        elif config.generation_mode == GenerationMode.CHAIN_OF_THOUGHT:
            return f"""
请逐步思考并回答以下问题。请展示你的思考过程。

背景信息：
{context}

问题：{query}

请按以下格式回答：
思考过程：[你的分析过程]
最终答案：[你的答案]
"""
        elif config.generation_mode == GenerationMode.FEW_SHOT:
            return f"""
参考以下示例，基于上下文信息回答问题：

示例：
问：什么是机器学习？
答：机器学习是人工智能的一个分支...

现在请回答：

上下文：
{context}

问题：{query}

答：
"""
        else:
            return f"请根据以下信息回答：{query}\n\n背景：{context}"

    async def _generate_once(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        model_config: ModelConfig
    ) -> str:
        """执行单次生成。"""
        if model_config.provider == ModelProvider.OPENAI:
            return await self._generate_openai(messages, config, model_config)
        elif model_config.provider == ModelProvider.ANTHROPIC:
            return await self._generate_anthropic(messages, config, model_config)
        elif model_config.provider == ModelProvider.AZURE_OPENAI:
            return await self._generate_azure_openai(messages, config, model_config)
        elif model_config.provider == ModelProvider.HUGGINGFACE:
            return await self._generate_huggingface(messages, config, model_config)
        elif model_config.provider == ModelProvider.LOCAL:
            return await self._generate_local(messages, config, model_config)
        else:
            raise GenerationException(f"不支持的模型提供商: {model_config.provider}")

    async def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        model_config: ModelConfig
    ) -> AsyncGenerator[StreamingChunk, None]:
        """执行流式生成。"""
        if model_config.provider == ModelProvider.OPENAI:
            async for chunk in self._generate_openai_stream(messages, config, model_config):
                yield chunk
        elif model_config.provider == ModelProvider.ANTHROPIC:
            async for chunk in self._generate_anthropic_stream(messages, config, model_config):
                yield chunk
        elif model_config.provider == ModelProvider.AZURE_OPENAI:
            async for chunk in self._generate_azure_openai_stream(messages, config, model_config):
                yield chunk
        else:
            # 对于不支持流式的提供商，回退到单次生成
            answer = await self._generate_once(messages, config, model_config)
            yield StreamingChunk(
                content=answer,
                delta=answer,
                finish_reason="stop"
            )

    async def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        model_config: ModelConfig
    ) -> str:
        """使用OpenAI模型生成。"""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=model_config.api_key,
                base_url=model_config.api_base,
                organization=model_config.organization
            )

            response = await client.chat.completions.create(
                model=config.model_name or model_config.model_name,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop_sequences if config.stop_sequences else None,
                functions=config.functions if config.enable_function_calling and config.functions else None,
                function_call=config.function_call if config.enable_function_calling else None,
                timeout=config.timeout_seconds
            )

            return response.choices[0].message.content or ""

        except ImportError:
            raise GenerationException("openai 库未安装")
        except Exception as e:
            self.logger.error(f"OpenAI生成失败: {e}")
            raise GenerationException(f"OpenAI生成失败: {str(e)}")

    async def _generate_openai_stream(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        model_config: ModelConfig
    ) -> AsyncGenerator[StreamingChunk, None]:
        """使用OpenAI模型流式生成。"""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=model_config.api_key,
                base_url=model_config.api_base,
                organization=model_config.organization
            )

            stream = await client.chat.completions.create(
                model=config.model_name or model_config.model_name,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop_sequences if config.stop_sequences else None,
                stream=True,
                timeout=config.timeout_seconds
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield StreamingChunk(
                        content=chunk.choices[0].delta.content,
                        delta=chunk.choices[0].delta.content,
                        finish_reason=chunk.choices[0].finish_reason,
                        usage=chunk.usage.model_dump() if chunk.usage else None
                    )

                if chunk.choices[0].finish_reason:
                    break

        except ImportError:
            raise GenerationException("openai 库未安装")
        except Exception as e:
            self.logger.error(f"OpenAI流式生成失败: {e}")
            raise GenerationException(f"OpenAI流式生成失败: {str(e)}")

    async def _generate_anthropic(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        model_config: ModelConfig
    ) -> str:
        """使用Anthropic模型生成。"""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(
                api_key=model_config.api_key
            )

            # 转换消息格式
            system_message = ""
            user_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)

            response = await client.messages.create(
                model=config.model_name or model_config.model_name,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system=system_message,
                messages=user_messages,
                timeout=config.timeout_seconds
            )

            return response.content[0].text if response.content else ""

        except ImportError:
            raise GenerationException("anthropic 库未安装")
        except Exception as e:
            self.logger.error(f"Anthropic生成失败: {e}")
            raise GenerationException(f"Anthropic生成失败: {str(e)}")

    async def _generate_anthropic_stream(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        model_config: ModelConfig
    ) -> AsyncGenerator[StreamingChunk, None]:
        """使用Anthropic模型流式生成。"""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(
                api_key=model_config.api_key
            )

            # 转换消息格式
            system_message = ""
            user_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)

            async with client.messages.stream(
                model=config.model_name or model_config.model_name,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system=system_message,
                messages=user_messages,
                timeout=config.timeout_seconds
            ) as stream:
                async for chunk in stream:
                    if chunk.type == "content_block_delta":
                        yield StreamingChunk(
                            content=chunk.delta.text,
                            delta=chunk.delta.text,
                            finish_reason=None
                        )

                yield StreamingChunk(
                    content="",
                    delta="",
                    finish_reason="stop"
                )

        except ImportError:
            raise GenerationException("anthropic 库未安装")
        except Exception as e:
            self.logger.error(f"Anthropic流式生成失败: {e}")
            raise GenerationException(f"Anthropic流式生成失败: {str(e)}")

    async def _generate_azure_openai(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        model_config: ModelConfig
    ) -> str:
        """使用Azure OpenAI模型生成。"""
        # 简化实现，实际需要配置Azure OpenAI客户端
        self.logger.warning("Azure OpenAI生成功能待完整实现")
        return await self._generate_openai(messages, config, model_config)

    async def _generate_azure_openai_stream(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        model_config: ModelConfig
    ) -> AsyncGenerator[StreamingChunk, None]:
        """使用Azure OpenAI模型流式生成。"""
        # 简化实现
        async for chunk in self._generate_openai_stream(messages, config, model_config):
            yield chunk

    async def _generate_huggingface(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        model_config: ModelConfig
    ) -> str:
        """使用HuggingFace模型生成。"""
        # 简化实现，实际需要集成Transformers库
        self.logger.warning("HuggingFace生成功能待完整实现")
        return "HuggingFace模型生成功能待实现"

    async def _generate_local(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        model_config: ModelConfig
    ) -> str:
        """使用本地模型生成。"""
        # 简化实现，实际需要集成本地推理框架
        self.logger.warning("本地模型生成功能待完整实现")
        return "本地模型生成功能待实现"

    async def _process_output_format(
        self,
        content: str,
        config: GenerationConfig
    ) -> str:
        """处理输出格式。"""
        if config.output_format == OutputFormat.TEXT:
            return content
        elif config.output_format == OutputFormat.JSON:
            try:
                # 尝试解析JSON
                json.loads(content)
                return content
            except json.JSONDecodeError:
                # 如果不是有效JSON，包装成JSON格式
                return json.dumps({"content": content}, ensure_ascii=False)
        elif config.output_format == OutputFormat.MARKDOWN:
            # 确保是有效的Markdown
            return content
        elif config.output_format == OutputFormat.HTML:
            # 简单转义HTML
            import html
            return f"<div>{html.escape(content)}</div>"

        return content

    async def _estimate_token_usage(
        self,
        messages: List[Dict[str, str]],
        answer: str
    ) -> Dict[str, int]:
        """估算token使用量。"""
        try:
            import tiktoken

            # 使用GPT-3.5的编码器作为默认估算
            encoding = tiktoken.get_encoding("cl100k_base")

            # 计算输入token
            input_text = ""
            for msg in messages:
                input_text += f"{msg['role']}: {msg['content']}\n"

            input_tokens = len(encoding.encode(input_text))
            output_tokens = len(encoding.encode(answer))

            return {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }

        except ImportError:
            # 简化估算：按字符数除以4
            input_text = "".join(f"{msg['role']}: {msg['content']}" for msg in messages)
            input_tokens = len(input_text) // 4
            output_tokens = len(answer) // 4

            return {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }

    def _get_cache_key(
        self,
        messages: List[Dict[str, str]],
        config: GenerationConfig
    ) -> str:
        """生成缓存键。"""
        key_parts = [
            str(msg["role"]) + ":" + str(msg["content"]) for msg in messages
        ]
        key_parts.append(str(config.temperature))
        key_parts.append(str(config.max_tokens))
        return hash(":".join(key_parts))

    async def add_model(self, model_config: Dict[str, Any]) -> bool:
        """
        添加新模型。

        Args:
            model_config: 模型配置

        Returns:
            bool: 添加是否成功
        """
        try:
            model_name = model_config["model_name"]
            if model_name in self.models:
                self.logger.warning(f"模型 {model_name} 已存在")
                return False

            model = ModelConfig(
                provider=ModelProvider(model_config.get("provider", "openai")),
                model_name=model_name,
                **{k: v for k, v in model_config.items() if k != "provider" and k != "model_name"}
            )
            self.models[model_name] = model

            if not self.default_model:
                self.default_model = model_name

            self.logger.info(f"添加模型 {model_name} 成功")
            return True

        except Exception as e:
            self.logger.error(f"添加模型失败: {e}")
            return False

    async def remove_model(self, model_name: str) -> bool:
        """
        移除模型。

        Args:
            model_name: 模型名称

        Returns:
            bool: 移除是否成功
        """
        if model_name not in self.models:
            return False

        if self.default_model == model_name:
            remaining_models = [name for name in self.models.keys() if name != model_name]
            self.default_model = remaining_models[0] if remaining_models else None

        del self.models[model_name]
        self.logger.info(f"移除模型 {model_name} 成功")
        return True

    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        获取模型信息。

        Args:
            model_name: 模型名称

        Returns:
            Optional[Dict[str, Any]]: 模型信息
        """
        model = self.models.get(model_name)
        if not model:
            return None

        return {
            "model_name": model.model_name,
            "provider": model.provider.value,
            "api_base": model.api_base,
            "timeout": model.timeout,
            "max_retries": model.max_retries,
            "usage_count": self.stats["model_usage"].get(model_name, 0)
        }

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        列出所有模型。

        Returns:
            List[Dict[str, Any]]: 模型信息列表
        """
        models_info = []
        for model_name in self.models.keys():
            model_info = await self.get_model_info(model_name)
            if model_info:
                models_info.append(model_info)

        return models_info

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息。"""
        success_rate = 0.0
        if self.stats["total_generations"] > 0:
            success_rate = self.stats["successful_generations"] / self.stats["total_generations"]

        return {
            **self.stats,
            "success_rate": success_rate,
            "available_models": list(self.models.keys()),
            "default_model": self.default_model,
            "cache_size": len(self.response_cache),
            "available_prompts": list(self.prompts.keys())
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查。"""
        health_status = {
            "status": "healthy",
            "models": {},
            "cache_enabled": len(self.response_cache) > 0,
            "errors": []
        }

        # 检查模型状态
        for model_name, model in self.models.items():
            try:
                # 简单的健康检查：尝试生成一个短文本
                test_messages = [
                    {"role": "user", "content": "Hello"}
                ]
                await self._generate_once(test_messages, self.default_generation_config, model)
                health_status["models"][model_name] = {
                    "status": "healthy",
                    "provider": model.provider.value
                }
            except Exception as e:
                health_status["models"][model_name] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["errors"].append(f"Model {model_name}: {str(e)}")

        # 总体状态
        if health_status["errors"]:
            health_status["status"] = "degraded"

        return health_status

    def add_prompt(self, name: str, prompt: str) -> None:
        """添加提示词模板。"""
        self.prompts[name] = prompt
        self.logger.info(f"添加提示词模板: {name}")

    def remove_prompt(self, name: str) -> bool:
        """移除提示词模板。"""
        if name in self.prompts:
            del self.prompts[name]
            self.logger.info(f"移除提示词模板: {name}")
            return True
        return False

    async def clear_cache(self) -> bool:
        """清理缓存。"""
        try:
            self.response_cache.clear()
            self.logger.info("生成缓存已清理")
            return True
        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")
            return False