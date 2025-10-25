"""
通义千问LLM提供商

实现阿里云通义千问的大语言模型服务。
"""

import asyncio
import hashlib
import hmac
import time
import uuid
from datetime import datetime
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


structured_logger = get_logger("rag_service.providers.qwen_llm")


class QwenLLMProvider(LLMInterface):
    """
    通义千问LLM提供商实现

    提供阿里云通义千问模型的LLM服务，兼容OpenAI API格式。
    """

    def __init__(self, config: LLMConfig):
        """
        初始化通义千问LLM提供商

        Args:
            config: LLM配置
        """
        super().__init__(config)
        self.base_url = config.base_url or "https://dashscope.aliyuncs.com/api/v1"
        self.api_key = config.api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self._capabilities: Optional[LLMProviderCapabilities] = None

    @property
    def capabilities(self) -> LLMProviderCapabilities:
        """获取通义千问LLM提供商能力信息"""
        if self._capabilities is None:
            model_capabilities = self._get_model_capabilities(self.model)
            self._capabilities = LLMProviderCapabilities(
                supported_models=model_capabilities["models"],
                max_tokens=model_capabilities["max_tokens"],
                max_context_length=model_capabilities["max_context_length"],
                supports_streaming=True,
                supports_functions=True,
                supports_tools=True,
                supports_vision=model_capabilities["supports_vision"],
                supports_audio=False,
                supports_system_messages=True,
                supports_seed=False,
                supports_response_format=False,
                pricing_per_1k_tokens=model_capabilities["pricing_per_1k_tokens"],
                currency="CNY",
                average_latency_ms=model_capabilities["average_latency_ms"],
            )
        return self._capabilities

    async def initialize(self) -> None:
        """初始化通义千问客户端"""
        try:
            # 创建HTTP会话
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                connector=aiohttp.TCPConnector(limit=10)
            )

            # 测试连接
            await self._test_connection()
            structured_logger.info(
                "通义千问LLM提供商初始化成功",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.base_url,
                }
            )

        except Exception as e:
            structured_logger.error(
                f"通义千问LLM提供商初始化失败: {e}",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "error": str(e),
                }
            )
            raise LLMException(
                f"Failed to initialize Qwen provider: {e}",
                provider=self.provider_name,
                model=self.model
            )

    async def _test_connection(self) -> None:
        """测试通义千问连接"""
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
            raise LLMException(f"Qwen connection test failed: {e}")

    def _generate_signature(self, method: str, url: str, body: str) -> str:
        """生成阿里云API签名"""
        # 阿里云API签名算法实现
        timestamp = str(int(time.time()))
        nonce = str(uuid.uuid4())

        # 构造签名字符串
        signature_string = f"{method}\n{url}\n{timestamp}\n{nonce}\n{body}"

        # 使用HMAC-SHA256计算签名
        signature = hmac.new(
            self.api_key.encode('utf-8'),
            signature_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature, timestamp, nonce

    def _get_headers(self, method: str, url: str, body: str = "") -> Dict[str, str]:
        """获取请求头"""
        signature, timestamp, nonce = self._generate_signature(method, url, body)

        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Signature": signature,
            "X-DashScope-Timestamp": timestamp,
            "X-DashScope-Nonce": nonce,
        }

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
            # 转换消息格式为通义千问格式
            qwen_messages = self._convert_messages(request.messages)

            # 构建通义千问请求参数
            qwen_params = {
                "model": request.model,
                "input": {
                    "messages": qwen_messages
                },
                "parameters": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "max_tokens": request.max_tokens,
                    "repetition_penalty": 1.0 + request.presence_penalty,
                    "stop": request.stop,
                    "incremental_output": False,
                }
            }

            # 添加工具/功能调用支持
            if request.tools:
                qwen_params["input"]["tools"] = [
                    {
                        "type": tool.type,
                        "function": {
                            "name": tool.function["name"],
                            "description": tool.function.get("description", ""),
                            "parameters": tool.function.get("parameters", {})
                        }
                    }
                    for tool in request.tools
                ]
            if request.tool_choice:
                qwen_params["parameters"]["tool_choice"] = request.tool_choice

            # 添加额外参数
            if kwargs:
                qwen_params["parameters"].update(kwargs)

            # 构造请求URL和Body
            url = f"{self.base_url}/services/aigc/text-generation/generation"
            body = json.dumps(qwen_params, ensure_ascii=False)
            headers = self._get_headers("POST", "/services/aigc/text-generation/generation", body)

            # 调用通义千问API
            async with self.session.post(url, headers=headers, data=body) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._handle_api_error(response.status, error_text, request.model)

                qwen_response = await response.json()

            # 转换响应格式
            return self._convert_response(qwen_response, request)

        except aiohttp.ClientError as e:
            raise LLMException(
                f"Qwen connection error: {e}",
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
            # 转换消息格式为通义千问格式
            qwen_messages = self._convert_messages(request.messages)

            # 构建通义千问请求参数
            qwen_params = {
                "model": request.model,
                "input": {
                    "messages": qwen_messages
                },
                "parameters": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "max_tokens": request.max_tokens,
                    "repetition_penalty": 1.0 + request.presence_penalty,
                    "stop": request.stop,
                    "incremental_output": True,  # 启用流式输出
                }
            }

            # 添加额外参数
            if kwargs:
                qwen_params["parameters"].update(kwargs)

            # 构造请求URL和Body
            url = f"{self.base_url}/services/aigc/text-generation/generation"
            body = json.dumps(qwen_params, ensure_ascii=False)
            headers = self._get_headers("POST", "/services/aigc/text-generation/generation", body)

            # 调用通义千问流式API
            async with self.session.post(url, headers=headers, data=body) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._handle_api_error(response.status, error_text, request.model)

                # 处理流式响应
                async for chunk in self._process_stream_response(response):
                    yield chunk

        except aiohttp.ClientError as e:
            raise LLMException(
                f"Qwen connection error: {e}",
                provider=self.provider_name,
                model=request.model
            )
        except Exception as e:
            raise LLMException(
                f"Stream chat completion failed: {e}",
                provider=self.provider_name,
                model=request.model
            )

    async def _process_stream_response(self, response):
        """处理流式响应"""
        buffer = ""

        async for line in response.content:
            buffer += line.decode('utf-8')

            # 处理SSE格式
            while '\n\n' in buffer:
                chunk, buffer = buffer.split('\n\n', 1)

                if chunk.startswith('data: '):
                    data = chunk[6:]  # 移除 'data: ' 前缀

                    if data == '[DONE]':
                        return

                    try:
                        chunk_data = json.loads(data)
                        if 'output' in chunk_data and 'text' in chunk_data['output']:
                            yield self._convert_stream_chunk(chunk_data)
                    except json.JSONDecodeError:
                        continue

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """转换消息格式为通义千问格式"""
        qwen_messages = []
        for msg in messages:
            qwen_msg = {
                "role": msg.role.value,
                "content": msg.content,
            }

            # 添加可选字段
            if msg.name:
                qwen_msg["name"] = msg.name

            # 通义千问支持图片（多模态）
            if hasattr(msg, 'images') and msg.images:
                qwen_msg["content"] = [
                    {"type": "text", "text": msg.content},
                    *[{"type": "image_url", "image_url": {"url": img}} for img in msg.images]
                ]

            qwen_messages.append(qwen_msg)

        return qwen_messages

    def _convert_response(self, qwen_response: Dict[str, Any], request: ChatCompletionRequest) -> ChatCompletionResponse:
        """转换通义千问响应为标准格式"""
        if "output" not in qwen_response or "text" not in qwen_response["output"]:
            raise LLMException("Invalid response format from Qwen API")

        message = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=qwen_response["output"]["text"],
        )

        choice = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason="stop",
            logprobs=None,
        )

        # 获取使用信息
        usage_info = qwen_response.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_info.get("input_tokens", 0),
            completion_tokens=usage_info.get("output_tokens", 0),
            total_tokens=usage_info.get("total_tokens", 0),
        )

        return ChatCompletionResponse(
            id=f"qwen-{qwen_response.get('request_id', str(uuid.uuid4()))}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[choice],
            usage=usage,
            system_fingerprint=None,
        )

    def _convert_stream_chunk(self, qwen_chunk: Dict[str, Any]) -> ChatCompletionStreamChunk:
        """转换通义千问流式响应块为标准格式"""
        choice = {
            "index": 0,
            "delta": {"content": qwen_chunk["output"]["text"]},
            "finish_reason": None,
        }

        return ChatCompletionStreamChunk(
            id=f"qwen-{qwen_chunk.get('request_id', str(uuid.uuid4()))}",
            object="chat.completion.chunk",
            created=int(time.time()),
            model=qwen_chunk.get("model", "qwen"),
            choices=[choice],
            usage=None,
        )

    def _handle_api_error(self, status_code: int, error_text: str, model: str):
        """处理API错误"""
        try:
            error_data = json.loads(error_text)
            error_code = error_data.get("code", "unknown")
            error_message = error_data.get("message", error_text)
        except json.JSONDecodeError:
            error_code = f"http_{status_code}"
            error_message = error_text

        if status_code == 429:
            raise RateLimitException(
                f"Qwen rate limit exceeded: {error_message}",
                provider=self.provider_name,
                model=model,
                error_code=error_code
            )
        elif status_code == 404:
            raise ModelUnavailableException(
                f"Model {model} not available: {error_message}",
                provider=self.provider_name,
                model=model,
                error_code=error_code
            )
        else:
            raise LLMException(
                f"Qwen API error: {error_message}",
                provider=self.provider_name,
                model=model,
                error_code=error_code
            )

    def _get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """获取模型能力信息"""
        # 通义千问模型能力配置
        model_configs = {
            "qwen-turbo": {
                "models": ["qwen-turbo", "qwen-turbo-chat"],
                "max_tokens": 8000,
                "max_context_length": 8000,
                "supports_vision": False,
                "pricing_per_1k_tokens": 0.002,  # CNY
                "average_latency_ms": 800,
            },
            "qwen-plus": {
                "models": ["qwen-plus", "qwen-plus-chat"],
                "max_tokens": 30000,
                "max_context_length": 30000,
                "supports_vision": False,
                "pricing_per_1k_tokens": 0.004,
                "average_latency_ms": 1200,
            },
            "qwen-max": {
                "models": ["qwen-max", "qwen-max-chat"],
                "max_tokens": 6000,
                "max_context_length": 6000,
                "supports_vision": False,
                "pricing_per_1k_tokens": 0.02,
                "average_latency_ms": 2000,
            },
            "qwen-vl-plus": {
                "models": ["qwen-vl-plus", "qwen-vl-plus-chat"],
                "max_tokens": 2000,
                "max_context_length": 2000,
                "supports_vision": True,
                "pricing_per_1k_tokens": 0.01,
                "average_latency_ms": 3000,
            },
            "qwen-vl-max": {
                "models": ["qwen-vl-max", "qwen-vl-max-chat"],
                "max_tokens": 2000,
                "max_context_length": 2000,
                "supports_vision": True,
                "pricing_per_1k_tokens": 0.03,
                "average_latency_ms": 4000,
            },
        }

        # 查找匹配的模型配置
        for key, config in model_configs.items():
            if model in config["models"]:
                return config
            # 或者模型名称包含关键词
            if key in model.lower():
                return config

        # 默认配置（适用于未知模型）
        return {
            "models": [model],
            "max_tokens": 8000,
            "max_context_length": 8000,
            "supports_vision": False,
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
        # 通义千问没有提供token计数API，使用近似计算
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

            # 简单的健康检查（可以调用模型列表API）
            url = f"{self.base_url}/services/aigc/text-generation/models"
            headers = self._get_headers("GET", "/services/aigc/text-generation/models")

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "provider": self.provider_name,
                        "model": self.model,
                        "base_url": self.base_url,
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
        """清理通义千问客户端资源"""
        if self.session:
            await self.session.close()
            self.session = None
        structured_logger.info("通义千问LLM提供商资源清理完成")