"""
Ollama LLM提供商

实现Ollama本地模型的大语言模型服务。
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, AsyncGenerator

import aiohttp

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


structured_logger = get_logger("rag_service.providers.ollama_llm")


class OllamaLLMProvider(LLMInterface):
    """
    Ollama LLM提供商实现

    提供Ollama本地模型的LLM服务，兼容OpenAI API格式。
    """

    def __init__(self, config: LLMConfig):
        """
        初始化Ollama LLM提供商

        Args:
            config: LLM配置
        """
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        self.session: Optional[aiohttp.ClientSession] = None
        self._capabilities: Optional[LLMProviderCapabilities] = None
        self._available_models: Optional[List[str]] = None

    @property
    def capabilities(self) -> LLMProviderCapabilities:
        """获取Ollama LLM提供商能力信息"""
        if self._capabilities is None:
            model_capabilities = self._get_model_capabilities(self.model)
            self._capabilities = LLMProviderCapabilities(
                supported_models=model_capabilities["models"],
                max_tokens=model_capabilities["max_tokens"],
                max_context_length=model_capabilities["max_context_length"],
                supports_streaming=True,
                supports_functions=False,  # Ollama暂不支持Function Calling
                supports_tools=False,     # Ollama暂不支持Tool Calling
                supports_vision=model_capabilities["supports_vision"],
                supports_audio=False,
                supports_system_messages=True,
                supports_seed=False,
                supports_response_format=False,
                pricing_per_1k_tokens=None,  # 本地模型无费用
                currency=None,
                average_latency_ms=model_capabilities["average_latency_ms"],
            )
        return self._capabilities

    async def initialize(self) -> None:
        """初始化Ollama客户端"""
        try:
            # 创建HTTP会话
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                connector=aiohttp.TCPConnector(limit=10)
            )

            # 测试连接并获取可用模型
            await self._test_connection()
            await self._refresh_available_models()

            structured_logger.info(
                "Ollama LLM提供商初始化成功",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.base_url,
                }
            )

        except Exception as e:
            structured_logger.error(
                f"Ollama LLM提供商初始化失败: {e}",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "error": str(e),
                }
            )
            raise LLMException(
                f"Failed to initialize Ollama provider: {e}",
                provider=self.provider_name,
                model=self.model
            )

    async def _test_connection(self) -> None:
        """测试Ollama连接"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    raise Exception(f"Ollama API returned status {response.status}")
        except Exception as e:
            raise LLMException(f"Ollama connection test failed: {e}")

    async def _refresh_available_models(self) -> None:
        """刷新可用模型列表"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    self._available_models = [model["name"] for model in data.get("models", [])]
                else:
                    self._available_models = [self.model]  # 默认使用配置的模型
        except Exception as e:
            structured_logger.warning(f"Failed to refresh available models: {e}")
            self._available_models = [self.model]

    @property
    def available_models(self) -> List[str]:
        """获取可用模型列表"""
        return self._available_models or [self.model]

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
            # 转换消息格式为Ollama格式
            ollama_messages = self._convert_messages(request.messages)

            # 构建Ollama请求参数
            ollama_params = {
                "model": request.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens,
                    "stop": request.stop,
                    "frequency_penalty": request.frequency_penalty,
                    "presence_penalty": request.presence_penalty,
                }
            }

            # 添加额外选项
            if kwargs:
                ollama_params["options"].update(kwargs)

            # 调用Ollama API
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=ollama_params
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMException(
                        f"Ollama API error: {error_text}",
                        provider=self.provider_name,
                        model=request.model
                    )

                ollama_response = await response.json()

            # 转换响应格式
            return self._convert_response(ollama_response, request)

        except aiohttp.ClientError as e:
            raise LLMException(
                f"Ollama connection error: {e}",
                provider=self.provider_name,
                model=request.model
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
            # 转换消息格式为Ollama格式
            ollama_messages = self._convert_messages(request.messages)

            # 构建Ollama请求参数
            ollama_params = {
                "model": request.model,
                "messages": ollama_messages,
                "stream": True,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens,
                    "stop": request.stop,
                    "frequency_penalty": request.frequency_penalty,
                    "presence_penalty": request.presence_penalty,
                }
            }

            # 添加额外选项
            if kwargs:
                ollama_params["options"].update(kwargs)

            # 调用Ollama流式API
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=ollama_params
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMException(
                        f"Ollama API error: {error_text}",
                        provider=self.provider_name,
                        model=request.model
                    )

                # 处理流式响应
                async for line in response.content:
                    if line.strip():
                        try:
                            ollama_chunk = json.loads(line.decode('utf-8'))
                            if not ollama_chunk.get("done", False):
                                yield self._convert_stream_chunk(ollama_chunk, request)
                        except json.JSONDecodeError:
                            continue  # 跳过无效的JSON行

        except aiohttp.ClientError as e:
            raise LLMException(
                f"Ollama connection error: {e}",
                provider=self.provider_name,
                model=request.model
            )
        except Exception as e:
            raise LLMException(
                f"Stream chat completion failed: {e}",
                provider=self.provider_name,
                model=request.model
            )

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """转换消息格式为Ollama格式"""
        ollama_messages = []
        for msg in messages:
            ollama_msg = {
                "role": msg.role.value,
                "content": msg.content,
            }

            # 添加可选字段
            if msg.name:
                ollama_msg["name"] = msg.name

            # Ollama支持图片（多模态）
            if hasattr(msg, 'images') and msg.images:
                ollama_msg["images"] = msg.images

            ollama_messages.append(ollama_msg)

        return ollama_messages

    def _convert_response(self, ollama_response: Dict[str, Any], request: ChatCompletionRequest) -> ChatCompletionResponse:
        """转换Ollama响应为标准格式"""
        message = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=ollama_response.get("message", {}).get("content", ""),
        )

        choice = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason="stop",  # Ollama总是返回stop
            logprobs=None,
        )

        # 估算token使用量（Ollama不提供精确计数）
        prompt_tokens = self._estimate_tokens(request.messages)
        completion_tokens = self._estimate_tokens([message])
        total_tokens = prompt_tokens + completion_tokens

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return ChatCompletionResponse(
            id=f"ollama-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[choice],
            usage=usage,
            system_fingerprint=None,
        )

    def _convert_stream_chunk(self, ollama_chunk: Dict[str, Any], request: ChatCompletionRequest) -> ChatCompletionStreamChunk:
        """转换Ollama流式响应块为标准格式"""
        message_content = ollama_chunk.get("message", {}).get("content", "")
        done = ollama_chunk.get("done", False)

        choice = {
            "index": 0,
            "delta": {"content": message_content},
            "finish_reason": "stop" if done else None,
        }

        return ChatCompletionStreamChunk(
            id=f"ollama-{int(time.time())}",
            object="chat.completion.chunk",
            created=int(time.time()),
            model=request.model,
            choices=[choice],
            usage=None,
        )

    def _estimate_tokens(self, messages: List[ChatMessage]) -> int:
        """估算token数量（简单实现）"""
        total_chars = sum(len(msg.content) for msg in messages)
        # 简单估算：约4个字符等于1个token
        return max(1, total_chars // 4)

    def _get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """获取模型能力信息"""
        # 常见Ollama模型能力配置
        model_configs = {
            "llama2": {
                "models": ["llama2", "llama2:7b", "llama2:13b", "llama2:70b"],
                "max_tokens": 4096,
                "max_context_length": 4096,
                "supports_vision": False,
                "average_latency_ms": 2000,
            },
            "llama3": {
                "models": ["llama3", "llama3:8b", "llama3:70b"],
                "max_tokens": 8192,
                "max_context_length": 8192,
                "supports_vision": False,
                "average_latency_ms": 1500,
            },
            "codellama": {
                "models": ["codellama", "codellama:7b", "codellama:13b", "codellama:34b"],
                "max_tokens": 16384,
                "max_context_length": 16384,
                "supports_vision": False,
                "average_latency_ms": 1800,
            },
            "mistral": {
                "models": ["mistral", "mistral:7b"],
                "max_tokens": 8192,
                "max_context_length": 8192,
                "supports_vision": False,
                "average_latency_ms": 1200,
            },
            "mixtral": {
                "models": ["mixtral", "mixtral:8x7b", "mixtral:8x22b"],
                "max_tokens": 32768,
                "max_context_length": 32768,
                "supports_vision": False,
                "average_latency_ms": 3000,
            },
            "qwen": {
                "models": ["qwen", "qwen:7b", "qwen:14b", "qwen:72b"],
                "max_tokens": 8192,
                "max_context_length": 8192,
                "supports_vision": False,
                "average_latency_ms": 1300,
            },
            "gemma": {
                "models": ["gemma", "gemma:2b", "gemma:7b"],
                "max_tokens": 8192,
                "max_context_length": 8192,
                "supports_vision": False,
                "average_latency_ms": 1000,
            },
            "llava": {
                "models": ["llava", "llava:7b", "llava:13b", "llava:34b"],
                "max_tokens": 4096,
                "max_context_length": 4096,
                "supports_vision": True,
                "average_latency_ms": 2500,
            },
        }

        # 查找匹配的模型配置
        for key, config in model_configs.items():
            if any(key in model for model in config["models"]):
                # 检查请求的模型是否在支持列表中
                if model in config["models"]:
                    return config
                # 或者模型名称包含关键词
                if key in model.lower():
                    return config

        # 默认配置（适用于未知模型）
        return {
            "models": [model],
            "max_tokens": 4096,
            "max_context_length": 4096,
            "supports_vision": False,
            "average_latency_ms": 2000,
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
        # Ollama没有提供token计数API，使用近似计算
        return max(1, len(text) // 4)

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict[str, Any]: 健康状态
        """
        try:
            if not self.session:
                return {
                    "status": "unhealthy",
                    "provider": self.provider_name,
                    "model": self.model,
                    "error": "Client not initialized"
                }

            # 测试Ollama API连接
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "provider": self.provider_name,
                        "model": self.model,
                        "base_url": self.base_url,
                        "available_models": self.available_models,
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "provider": self.provider_name,
                        "model": self.model,
                        "error": f"API returned status {response.status}"
                    }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "model": self.model,
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """清理Ollama客户端资源"""
        if self.session:
            await self.session.close()
            self.session = None
        structured_logger.info("Ollama LLM提供商资源清理完成")