"""
OpenAI Embedding提供商

实现OpenAI兼容的文本向量化服务。
"""

import asyncio
from typing import List, Dict, Any, Optional, Union

import openai
from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse

from ...interfaces.embedding_interface import (
    EmbeddingInterface,
    EmbeddingConfig,
    EmbeddingProviderCapabilities,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
    EmbeddingException,
    RateLimitException,
    TokenLimitException,
    ModelUnavailableException
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("rag_service.providers.openai_embedding")


class OpenAIEmbeddingProvider(EmbeddingInterface):
    """
    OpenAI Embedding提供商实现

    提供与OpenAI API完全兼容的文本向量化服务。
    """

    def __init__(self, config: EmbeddingConfig):
        """
        初始化OpenAI Embedding提供商

        Args:
            config: Embedding配置
        """
        super().__init__(config)
        self.client: Optional[AsyncOpenAI] = None
        self._capabilities: Optional[EmbeddingProviderCapabilities] = None

    @property
    def capabilities(self) -> EmbeddingProviderCapabilities:
        """获取OpenAI Embedding提供商能力信息"""
        if self._capabilities is None:
            model_capabilities = self._get_model_capabilities(self.model)
            self._capabilities = EmbeddingProviderCapabilities(
                supported_models=model_capabilities["models"],
                max_input_length=model_capabilities["max_input_length"],
                max_batch_size=model_capabilities["max_batch_size"],
                embedding_dimensions=model_capabilities["embedding_dimensions"],
                supports_batch=True,
                supports_different_input_types=False,
                supports_custom_dimensions=False,
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
                "OpenAI Embedding提供商初始化成功",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.config.base_url or "https://api.openai.com/v1",
                }
            )

        except Exception as e:
            structured_logger.error(
                f"OpenAI Embedding提供商初始化失败: {e}",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "error": str(e),
                }
            )
            raise EmbeddingException(
                f"Failed to initialize OpenAI embedding provider: {e}",
                provider=self.provider_name,
                model=self.model
            )

    async def _test_connection(self) -> None:
        """测试OpenAI连接"""
        try:
            # 发送一个简单的测试请求
            test_request = EmbeddingRequest(
                input="Hello world",
                model=self.model
            )
            await self.create_embedding(test_request)
        except Exception as e:
            raise EmbeddingException(f"OpenAI connection test failed: {e}")

    async def create_embedding(
        self,
        request: EmbeddingRequest,
        **kwargs
    ) -> EmbeddingResponse:
        """
        创建文本向量

        Args:
            request: 向量化请求
            **kwargs: 额外参数

        Returns:
            EmbeddingResponse: 向量化响应
        """
        try:
            # 构建OpenAI请求参数
            openai_params = {
                "model": request.model,
                "input": request.input,
                "encoding_format": request.encoding_format,
                "user": request.user,
                **kwargs
            }

            # 添加自定义维度支持（仅支持的模型）
            if request.dimensions and self.supports_feature("custom_dimensions"):
                openai_params["dimensions"] = request.dimensions

            # 调用OpenAI API
            response: CreateEmbeddingResponse = await self.client.embeddings.create(**openai_params)

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
                raise EmbeddingException(
                    f"OpenAI API error: {e}",
                    provider=self.provider_name,
                    model=request.model,
                    error_code=f"api_error_{e.status_code}"
                )
        except openai.APITimeoutError as e:
            raise EmbeddingException(
                f"OpenAI API timeout: {e}",
                provider=self.provider_name,
                model=self.model,
                error_code="timeout"
            )
        except Exception as e:
            raise EmbeddingException(
                f"Embedding creation failed: {e}",
                provider=self.provider_name,
                model=self.model
            )

    def _convert_response(self, response: CreateEmbeddingResponse) -> EmbeddingResponse:
        """转换OpenAI响应为标准格式"""
        data = []
        for item in response.data:
            embedding_data = EmbeddingData(
                object=item.object,
                embedding=item.embedding,
                index=item.index,
            )
            data.append(embedding_data)

        usage = EmbeddingUsage(
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return EmbeddingResponse(
            object=response.object,
            data=data,
            model=response.model,
            usage=usage,
        )

    def _get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """获取模型能力信息"""
        # OpenAI嵌入模型能力配置
        model_configs = {
            "text-embedding-ada-002": {
                "models": ["text-embedding-ada-002"],
                "max_input_length": 8191,
                "max_batch_size": 2048,
                "embedding_dimensions": {"text-embedding-ada-002": 1536},
                "pricing_per_1k_tokens": 0.0001,
                "average_latency_ms": 200,
            },
            "text-embedding-3-small": {
                "models": ["text-embedding-3-small"],
                "max_input_length": 8191,
                "max_batch_size": 2048,
                "embedding_dimensions": {"text-embedding-3-small": 1536},
                "supports_custom_dimensions": True,
                "pricing_per_1k_tokens": 0.00002,
                "average_latency_ms": 150,
            },
            "text-embedding-3-large": {
                "models": ["text-embedding-3-large"],
                "max_input_length": 8191,
                "max_batch_size": 2048,
                "embedding_dimensions": {
                    "text-embedding-3-large": 3072,
                },
                "supports_custom_dimensions": True,
                "pricing_per_1k_tokens": 0.00013,
                "average_latency_ms": 300,
            },
        }

        # 查找匹配的模型配置
        for key, config in model_configs.items():
            if model in config["models"]:
                return config

        # 默认配置（适用于兼容模型）
        return {
            "models": [model],
            "max_input_length": 8191,
            "max_batch_size": 2048,
            "embedding_dimensions": {model: 1536},
            "supports_custom_dimensions": False,
            "pricing_per_1k_tokens": 0.0001,
            "average_latency_ms": 200,
        }

    async def count_tokens(self, text: Union[str, List[str]], model: Optional[str] = None) -> Union[int, List[int]]:
        """
        计算文本的token数量

        Args:
            text: 要计算的文本或文本列表
            model: 模型名称

        Returns:
            Union[int, List[int]]: token数量或token数量列表
        """
        try:
            import tiktoken

            model_name = model or self.model
            try:
                # 获取模型的编码器
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # 如果模型不匹配，使用默认编码器
                encoding = tiktoken.get_encoding("cl100k_base")

            if isinstance(text, str):
                # 单个文本
                token_count = len(encoding.encode(text))
                return token_count
            else:
                # 文本列表
                token_counts = []
                for t in text:
                    token_count = len(encoding.encode(t))
                    token_counts.append(token_count)
                return token_counts

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
        structured_logger.info("OpenAI Embedding提供商资源清理完成")