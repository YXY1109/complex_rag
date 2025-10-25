"""
百度文心LLM提供商

实现百度文心的大语言模型服务。
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
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


structured_logger = get_logger("rag_service.providers.bce_llm")


class BCELLMProvider(LLMInterface):
    """
    百度文心LLM提供商实现

    提供百度文心模型的LLM服务，兼容OpenAI API格式。
    """

    def __init__(self, config: LLMConfig):
        """
        初始化百度文心LLM提供商

        Args:
            config: LLM配置
        """
        super().__init__(config)
        self.base_url = config.base_url or "https://aip.baidubce.com/rpc/2.0/ai_custom/v1"
        self.api_key = config.api_key
        self.secret_key = config.organization  # 使用organization字段存储secret_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[float] = None
        self._capabilities: Optional[LLMProviderCapabilities] = None

    @property
    def capabilities(self) -> LLMProviderCapabilities:
        """获取百度文心LLM提供商能力信息"""
        if self._capabilities is None:
            model_capabilities = self._get_model_capabilities(self.model)
            self._capabilities = LLMProviderCapabilities(
                supported_models=model_capabilities["models"],
                max_tokens=model_capabilities["max_tokens"],
                max_context_length=model_capabilities["max_context_length"],
                supports_streaming=True,
                supports_functions=True,
                supports_tools=False,
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
        """初始化百度文心客户端"""
        try:
            # 创建HTTP会话
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                connector=aiohttp.TCPConnector(limit=10)
            )

            # 获取访问令牌
            await self._refresh_access_token()

            # 测试连接
            await self._test_connection()
            structured_logger.info(
                "百度文心LLM提供商初始化成功",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.base_url,
                }
            )

        except Exception as e:
            structured_logger.error(
                f"百度文心LLM提供商初始化失败: {e}",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "error": str(e),
                }
            )
            raise LLMException(
                f"Failed to initialize BCE provider: {e}",
                provider=self.provider_name,
                model=self.model
            )

    async def _refresh_access_token(self) -> None:
        """刷新访问令牌"""
        try:
            # 构造获取token的请求
            token_url = f"https://aip.baidubce.com/oauth/2.0/token"
            params = {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            }

            async with self.session.post(token_url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMException(f"Failed to get access token: {error_text}")

                token_data = await response.json()
                self.access_token = token_data["access_token"]
                self.token_expires_at = time.time() + token_data["expires_in"] - 300  # 提前5分钟刷新

        except Exception as e:
            raise LLMException(f"Failed to refresh access token: {e}")

    async def _get_valid_access_token(self) -> str:
        """获取有效的访问令牌"""
        if not self.access_token or not self.token_expires_at or time.time() >= self.token_expires_at:
            await self._refresh_access_token()
        return self.access_token

    async def _test_connection(self) -> None:
        """测试百度文心连接"""
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
            raise LLMException(f"BCE connection test failed: {e}")

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
            # 获取有效的访问令牌
            access_token = await self._get_valid_access_token()

            # 转换消息格式为百度文心格式
            bce_messages = self._convert_messages(request.messages)

            # 构建百度文心请求参数
            bce_params = {
                "messages": bce_messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "penalty_score": 1.0 - request.presence_penalty,  # 百度文心的惩罚分数
                "stream": False,
                "user_id": request.user or "default",
            }

            # 添加可选参数
            if request.max_tokens:
                bce_params["max_output_tokens"] = request.max_tokens
            if request.stop:
                bce_params["stop"] = request.stop

            # 构造请求URL
            endpoint = self._get_endpoint(request.model)
            url = f"{self.base_url}/{endpoint}?access_token={access_token}"

            # 调用百度文心API
            async with self.session.post(url, json=bce_params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._handle_api_error(response.status, error_text, request.model)

                bce_response = await response.json()

            # 转换响应格式
            return self._convert_response(bce_response, request)

        except aiohttp.ClientError as e:
            raise LLMException(
                f"BCE connection error: {e}",
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
            # 获取有效的访问令牌
            access_token = await self._get_valid_access_token()

            # 转换消息格式为百度文心格式
            bce_messages = self._convert_messages(request.messages)

            # 构建百度文心请求参数
            bce_params = {
                "messages": bce_messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "penalty_score": 1.0 - request.presence_penalty,
                "stream": True,
                "user_id": request.user or "default",
            }

            # 添加可选参数
            if request.max_tokens:
                bce_params["max_output_tokens"] = request.max_tokens
            if request.stop:
                bce_params["stop"] = request.stop

            # 构造请求URL
            endpoint = self._get_endpoint(request.model)
            url = f"{self.base_url}/{endpoint}?access_token={access_token}"

            # 调用百度文心流式API
            async with self.session.post(url, json=bce_params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._handle_api_error(response.status, error_text, request.model)

                # 处理流式响应
                async for line in response.content:
                    if line.strip():
                        try:
                            bce_chunk = json.loads(line.decode('utf-8'))
                            if bce_chunk.get("is_end", False):
                                continue
                            yield self._convert_stream_chunk(bce_chunk, request)
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            raise LLMException(
                f"BCE connection error: {e}",
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
        """转换消息格式为百度文心格式"""
        bce_messages = []
        for msg in messages:
            bce_msg = {
                "role": msg.role.value,
                "content": msg.content,
            }

            # 添加可选字段
            if msg.name:
                bce_msg["name"] = msg.name

            bce_messages.append(bce_msg)

        return bce_messages

    def _convert_response(self, bce_response: Dict[str, Any], request: ChatCompletionRequest) -> ChatCompletionResponse:
        """转换百度文心响应为标准格式"""
        if "result" not in bce_response:
            raise LLMException("Invalid response format from BCE API")

        message = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=bce_response["result"],
        )

        choice = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason=bce_response.get("finish_reason", "stop"),
            logprobs=None,
        )

        # 获取使用信息
        usage_info = bce_response.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_info.get("prompt_tokens", 0),
            completion_tokens=usage_info.get("completion_tokens", 0),
            total_tokens=usage_info.get("total_tokens", 0),
        )

        return ChatCompletionResponse(
            id=f"bce-{bce_response.get('id', str(int(time.time())))}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[choice],
            usage=usage,
            system_fingerprint=None,
        )

    def _convert_stream_chunk(self, bce_chunk: Dict[str, Any], request: ChatCompletionRequest) -> ChatCompletionStreamChunk:
        """转换百度文心流式响应块为标准格式"""
        choice = {
            "index": 0,
            "delta": {"content": bce_chunk.get("result", "")},
            "finish_reason": "stop" if bce_chunk.get("is_end", False) else None,
        }

        return ChatCompletionStreamChunk(
            id=f"bce-{bce_chunk.get('id', str(int(time.time())))}",
            object="chat.completion.chunk",
            created=int(time.time()),
            model=request.model,
            choices=[choice],
            usage=None,
        )

    def _get_endpoint(self, model: str) -> str:
        """根据模型名称获取API端点"""
        # 百度文心模型端点映射
        endpoint_mapping = {
            "ernie-bot-4": "wenxinworkshop/chat/ernie-4.0-8k",
            "ernie-bot": "wenxinworkshop/chat/eb-instant",
            "ernie-bot-turbo": "wenxinworkshop/chat/ernie-speed-8k",
            "ernie-bot-8k": "wenxinworkshop/chat/ernie-8k",
            "ernie-bot-turbo-8k": "wenxinworkshop/chat/ernie-speed-8k",
            "ernie-bot-pro": "wenxinworkshop/chat/completions_pro",
            "ernie-bot-8k-pro": "wenxinworkshop/chat/completions",
            "ernie-vilg": "wenxinworkshop/text2image",
        }

        # 查找匹配的端点
        for model_pattern, endpoint in endpoint_mapping.items():
            if model_pattern in model:
                return endpoint

        # 默认端点
        return "wenxinworkshop/chat/eb-instant"

    def _handle_api_error(self, status_code: int, error_text: str, model: str):
        """处理API错误"""
        try:
            error_data = json.loads(error_text)
            error_code = error_data.get("error_code", "unknown")
            error_message = error_data.get("error_msg", error_text)
        except json.JSONDecodeError:
            error_code = f"http_{status_code}"
            error_message = error_text

        if status_code == 429:
            raise RateLimitException(
                f"BCE rate limit exceeded: {error_message}",
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
                f"BCE API error: {error_message}",
                provider=self.provider_name,
                model=model,
                error_code=error_code
            )

    def _get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """获取模型能力信息"""
        # 百度文心模型能力配置
        model_configs = {
            "ernie-bot-4": {
                "models": ["ernie-bot-4", "ernie-4.0-8k"],
                "max_tokens": 8192,
                "max_context_length": 8192,
                "supports_vision": False,
                "pricing_per_1k_tokens": 0.12,  # CNY
                "average_latency_ms": 2000,
            },
            "ernie-bot": {
                "models": ["ernie-bot", "ernie-bot-turbo", "eb-instant"],
                "max_tokens": 8192,
                "max_context_length": 8192,
                "supports_vision": False,
                "pricing_per_1k_tokens": 0.008,
                "average_latency_ms": 1000,
            },
            "ernie-bot-pro": {
                "models": ["ernie-bot-pro", "completions_pro"],
                "max_tokens": 8192,
                "max_context_length": 8192,
                "supports_vision": False,
                "pricing_per_1k_tokens": 0.12,
                "average_latency_ms": 1500,
            },
            "ernie-bot-8k": {
                "models": ["ernie-bot-8k", "ernie-speed-8k", "completions"],
                "max_tokens": 2048,
                "max_context_length": 8192,
                "supports_vision": False,
                "pricing_per_1k_tokens": 0.004,
                "average_latency_ms": 800,
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
            "max_tokens": 8192,
            "max_context_length": 8192,
            "supports_vision": False,
            "pricing_per_1k_tokens": 0.008,
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
        # 百度文心没有提供token计数API，使用近似计算
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

            # 检查访问令牌是否有效
            try:
                access_token = await self._get_valid_access_token()
                return {
                    "status": "healthy",
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "has_access_token": bool(access_token),
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "provider": self.provider_name,
                    "model": self.model,
                    "error": f"Token error: {str(e)}"
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "model": self.model,
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """清理百度文心客户端资源"""
        if self.session:
            await self.session.close()
            self.session = None
        self.access_token = None
        self.token_expires_at = None
        structured_logger.info("百度文心LLM提供商资源清理完成")