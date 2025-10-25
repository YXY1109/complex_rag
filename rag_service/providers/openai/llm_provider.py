"""
OpenAI LLM提供商

实现OpenAI兼容的大语言模型服务。
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ...interfaces.llm_interface import (
    LLMInterface,
    LLMConfig,
    LLMProviderCapabilities,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChunk,
    ChatMessage,
    ChatRole,
    ChatCompletionResponseChoice,
    UsageInfo,
    LLMException,
    RateLimitException,
    TokenLimitException,
    ModelUnavailableException
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("rag_service.providers.openai_llm")


class OpenAILLMProvider(LLMInterface):
    """
    OpenAI LLM提供商实现

    提供与OpenAI API完全兼容的LLM服务。
    """

    def __init__(self, config: LLMConfig):
        """
        初始化OpenAI LLM提供商

        Args:
            config: LLM配置
        """
        super().__init__(config)
        self.client: Optional[AsyncOpenAI] = None
        self._capabilities: Optional[LLMProviderCapabilities] = None

    @property
    def capabilities(self) -> LLMProviderCapabilities:
        """获取OpenAI LLM提供商能力信息"""
        if self._capabilities is None:
            # 根据模型类型设置不同的能力
            model_capabilities = self._get_model_capabilities(self.model)
            self._capabilities = LLMProviderCapabilities(
                supported_models=model_capabilities["models"],
                max_tokens=model_capabilities["max_tokens"],
                max_context_length=model_capabilities["max_context_length"],
                supports_streaming=True,
                supports_functions=True,
                supports_tools=True,
                supports_vision=model_capabilities["supports_vision"],
                supports_audio=model_capabilities["supports_audio"],
                supports_system_messages=True,
                supports_seed=True,
                supports_response_format=True,
                pricing_per_1k_tokens=model_capabilities["pricing_per_1k_tokens"],
                currency="USD",
                average_latency_ms=model_capabilities["average_latency_ms"],
            )
        return self._capabilities

    async def initialize(self) -> None:
        """初始化OpenAI客户端"""
        try:
            # 创建OpenAI客户端
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                organization=self.config.organization,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )

            # 测试连接
            await self._test_connection()
            structured_logger.info(
                "OpenAI LLM提供商初始化成功",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.config.base_url or "https://api.openai.com/v1",
                }
            )

        except Exception as e:
            structured_logger.error(
                f"OpenAI LLM提供商初始化失败: {e}",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "error": str(e),
                }
            )
            raise LLMException(
                f"Failed to initialize OpenAI provider: {e}",
                provider=self.provider_name,
                model=self.model
            )

    async def _test_connection(self) -> None:
        """测试OpenAI连接"""
        try:
            # 发送一个简单的测试请求
            test_request = ChatCompletionRequest(
                messages=[
                    ChatMessage(role=ChatRole.USER, content="Hello")
                ],
                model=self.model,
                max_tokens=5
            )
            await self.chat_completion(test_request)
        except Exception as e:
            raise LLMException(f"OpenAI connection test failed: {e}")

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        **kwargs
    ) -> ChatCompletionResponse:
        """
        生成聊天完成响应

        Args:
            request: 聊天完成请求
            **kwargs: 额外参数

        Returns:
            ChatCompletionResponse: 聊天完成响应
        """
        try:
            # 转换消息格式
            openai_messages = self._convert_messages(request.messages)

            # 构建OpenAI请求参数
            openai_params = {
                "model": request.model,
                "messages": openai_messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "n": request.n,
                "stream": False,
                "stop": request.stop,
                "max_tokens": request.max_tokens,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
                "user": request.user,
                **kwargs
            }

            # 添加可选参数
            if request.logit_bias:
                openai_params["logit_bias"] = request.logit_bias
            if request.seed:
                openai_params["seed"] = request.seed
            if request.response_format:
                openai_params["response_format"] = request.response_format

            # 添加工具/功能调用支持
            if request.tools:
                openai_params["tools"] = [
                    {"type": tool.type, "function": tool.function}
                    for tool in request.tools
                ]
            if request.tool_choice:
                openai_params["tool_choice"] = request.tool_choice

            # 调用OpenAI API
            response: ChatCompletion = await self.client.chat.completions.create(**openai_params)

            # 转换响应格式
            return self._convert_response(response)

        except openai.RateLimitError as e:
            raise RateLimitException(
                f"OpenAI rate limit exceeded: {e}",
                provider=self.provider_name,
                model=self.model,
                error_code="rate_limit_exceeded"
            )
        except openai.APIStatusError as e:
            if e.status_code == 429:
                raise RateLimitException(
                    f"OpenAI rate limit exceeded: {e}",
                    provider=self.provider_name,
                    model=self.model,
                    error_code="rate_limit_exceeded"
                )
            elif e.status_code == 404:
                raise ModelUnavailableException(
                    f"Model {request.model} not available: {e}",
                    provider=self.provider_name,
                    model=request.model,
                    error_code="model_not_found"
                )
            else:
                raise LLMException(
                    f"OpenAI API error: {e}",
                    provider=self.provider_name,
                    model=request.model,
                    error_code=f"api_error_{e.status_code}"
                )
        except openai.APITimeoutError as e:
            raise LLMException(
                f"OpenAI API timeout: {e}",
                provider=self.provider_name,
                model=request.model,
                error_code="timeout"
            )
        except Exception as e:
            raise LLMException(
                f"Chat completion failed: {e}",
                provider=self.provider_name,
                model=request.model
            )

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        **kwargs
    ) -> AsyncGenerator[ChatCompletionStreamChunk, None]:
        """
        生成流式聊天完成响应

        Args:
            request: 聊天完成请求
            **kwargs: 额外参数

        Yields:
            ChatCompletionStreamChunk: 流式响应块
        """
        try:
            # 转换消息格式
            openai_messages = self._convert_messages(request.messages)

            # 构建OpenAI请求参数
            openai_params = {
                "model": request.model,
                "messages": openai_messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "n": request.n,
                "stream": True,
                "stop": request.stop,
                "max_tokens": request.max_tokens,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
                "user": request.user,
                **kwargs
            }

            # 添加可选参数
            if request.logit_bias:
                openai_params["logit_bias"] = request.logit_bias
            if request.seed:
                openai_params["seed"] = request.seed

            # 添加工具/功能调用支持
            if request.tools:
                openai_params["tools"] = [
                    {"type": tool.type, "function": tool.function}
                    for tool in request.tools
                ]
            if request.tool_choice:
                openai_params["tool_choice"] = request.tool_choice

            # 调用OpenAI流式API
            stream = await self.client.chat.completions.create(**openai_params)

            # 处理流式响应
            async for chunk in stream:
                yield self._convert_stream_chunk(chunk)

        except openai.RateLimitError as e:
            raise RateLimitException(
                f"OpenAI rate limit exceeded: {e}",
                provider=self.provider_name,
                model=self.model,
                error_code="rate_limit_exceeded"
            )
        except openai.APIStatusError as e:
            if e.status_code == 429:
                raise RateLimitException(
                    f"OpenAI rate limit exceeded: {e}",
                    provider=self.provider_name,
                    model=self.model,
                    error_code="rate_limit_exceeded"
                )
            elif e.status_code == 404:
                raise ModelUnavailableException(
                    f"Model {request.model} not available: {e}",
                    provider=self.provider_name,
                    model=request.model,
                    error_code="model_not_found"
                )
            else:
                raise LLMException(
                    f"OpenAI API error: {e}",
                    provider=self.provider_name,
                    model=request.model,
                    error_code=f"api_error_{e.status_code}"
                )
        except Exception as e:
            raise LLMException(
                f"Stream chat completion failed: {e}",
                provider=self.provider_name,
                model=request.model
            )

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """转换消息格式为OpenAI格式"""
        openai_messages = []
        for msg in messages:
            openai_msg = {
                "role": msg.role.value,
                "content": msg.content,
            }

            # 添加可选字段
            if msg.name:
                openai_msg["name"] = msg.name
            if msg.function_call:
                openai_msg["function_call"] = msg.function_call
            if msg.tool_calls:
                openai_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id

            openai_messages.append(openai_msg)

        return openai_messages

    def _convert_response(self, response: ChatCompletion) -> ChatCompletionResponse:
        """转换OpenAI响应为标准格式"""
        choices = []
        for choice in response.choices:
            message = ChatMessage(
                role=ChatRole(choice.message.role),
                content=choice.message.content or "",
                name=choice.message.name,
                function_call=choice.message.function_call,
                tool_calls=choice.message.tool_calls,
                tool_call_id=choice.message.tool_call_id,
            )

            response_choice = ChatCompletionResponseChoice(
                index=choice.index,
                message=message,
                finish_reason=choice.finish_reason,
                logprobs=choice.logprobs,
            )
            choices.append(response_choice)

        usage = None
        if response.usage:
            usage = UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return ChatCompletionResponse(
            id=response.id,
            object=response.object,
            created=response.created,
            model=response.model,
            choices=choices,
            usage=usage,
            system_fingerprint=response.system_fingerprint,
        )

    def _convert_stream_chunk(self, chunk: ChatCompletionChunk) -> ChatCompletionStreamChunk:
        """转换OpenAI流式响应块为标准格式"""
        choices = []
        for choice in chunk.choices:
            choice_dict = {
                "index": choice.index,
                "delta": choice.delta.content or "",
                "finish_reason": choice.finish_reason,
            }

            if choice.delta.role:
                choice_dict["role"] = choice.delta.role
            if choice.delta.tool_calls:
                choice_dict["tool_calls"] = choice.delta.tool_calls

            choices.append(choice_dict)

        usage = None
        if chunk.usage:
            usage = UsageInfo(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
            )

        return ChatCompletionStreamChunk(
            id=chunk.id,
            object=chunk.object,
            created=chunk.created,
            model=chunk.model,
            choices=choices,
            usage=usage,
        )

    def _get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """获取模型能力信息"""
        # OpenAI模型能力配置
        model_configs = {
            "gpt-4": {
                "models": ["gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613"],
                "max_tokens": 8192,
                "max_context_length": 8192,
                "supports_vision": False,
                "supports_audio": False,
                "pricing_per_1k_tokens": 0.03,
                "average_latency_ms": 2000,
            },
            "gpt-4-turbo": {
                "models": ["gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-preview", "gpt-4-turbo"],
                "max_tokens": 4096,
                "max_context_length": 128000,
                "supports_vision": True,
                "supports_audio": False,
                "pricing_per_1k_tokens": 0.01,
                "average_latency_ms": 1500,
            },
            "gpt-4o": {
                "models": ["gpt-4o", "gpt-4o-2024-05-13"],
                "max_tokens": 4096,
                "max_context_length": 128000,
                "supports_vision": True,
                "supports_audio": True,
                "pricing_per_1k_tokens": 0.005,
                "average_latency_ms": 800,
            },
            "gpt-3.5-turbo": {
                "models": ["gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"],
                "max_tokens": 4096,
                "max_context_length": 16384,
                "supports_vision": False,
                "supports_audio": False,
                "pricing_per_1k_tokens": 0.002,
                "average_latency_ms": 500,
            },
            "gpt-3.5-turbo-instruct": {
                "models": ["gpt-3.5-turbo-instruct"],
                "max_tokens": 4096,
                "max_context_length": 4096,
                "supports_vision": False,
                "supports_audio": False,
                "pricing_per_1k_tokens": 0.002,
                "average_latency_ms": 400,
            },
        }

        # 查找匹配的模型配置
        for key, config in model_configs.items():
            if model in config["models"]:
                return config

        # 默认配置（适用于兼容模型）
        return {
            "models": [model],
            "max_tokens": 4096,
            "max_context_length": 4096,
            "supports_vision": False,
            "supports_audio": False,
            "pricing_per_1k_tokens": 0.002,
            "average_latency_ms": 1000,
        }

    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        计算文本的token数量

        Args:
            text: 要计算的文本
            model: 模型名称

        Returns:
            int: token数量
        """
        try:
            # 使用tiktoken库进行精确计算
            import tiktoken

            model_name = model or self.model
            try:
                # 获取模型的编码器
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # 如果模型不匹配，使用默认编码器
                encoding = tiktoken.get_encoding("cl100k_base")

            # 计算token数量
            token_count = len(encoding.encode(text))
            return token_count

        except ImportError:
            # 如果tiktoken不可用，使用近似计算
            return super().count_tokens(text, model)
        except Exception as e:
            structured_logger.warning(
                f"Token counting failed: {e}, using fallback",
                extra={"model": model or self.model}
            )
            return super().count_tokens(text, model)

    async def cleanup(self) -> None:
        """清理OpenAI客户端资源"""
        if self.client:
            await self.client.close()
            self.client = None
        structured_logger.info("OpenAI LLM提供商资源清理完成")