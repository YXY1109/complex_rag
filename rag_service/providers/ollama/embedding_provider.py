"""
Ollama Embedding提供商

实现Ollama本地模型的文本向量化服务。
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Union

import aiohttp

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


structured_logger = get_logger("rag_service.providers.ollama_embedding")


class OllamaEmbeddingProvider(EmbeddingInterface):
    """
    Ollama Embedding提供商实现

    提供Ollama本地模型的文本向量化服务。
    """

    def __init__(self, config: EmbeddingConfig):
        """
        初始化Ollama Embedding提供商

        Args:
            config: Embedding配置
        """
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        self.session: Optional[aiohttp.ClientSession] = None
        self._capabilities: Optional[EmbeddingProviderCapabilities] = None
        self._available_models: Optional[List[str]] = None

    @property
    def capabilities(self) -> EmbeddingProviderCapabilities:
        """获取Ollama Embedding提供商能力信息"""
        if self._capabilities is None:
            model_capabilities = self._get_model_capabilities(self.model)
            self._capabilities = EmbeddingProviderCapabilities(
                supported_models=model_capabilities["models"],
                max_input_length=model_capabilities["max_input_length"],
                max_batch_size=1,  # Ollama一次只能处理一个文本
                embedding_dimensions=model_capabilities["embedding_dimensions"],
                supports_batch=False,
                supports_different_input_types=False,
                supports_custom_dimensions=False,
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
                "Ollama Embedding提供商初始化成功",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.base_url,
                }
            )

        except Exception as e:
            structured_logger.error(
                f"Ollama Embedding提供商初始化失败: {e}",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "error": str(e),
                }
            )
            raise EmbeddingException(
                f"Failed to initialize Ollama embedding provider: {e}",
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
            raise EmbeddingException(f"Ollama connection test failed: {e}")

    async def _refresh_available_models(self) -> None:
        """刷新可用模型列表"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    # 过滤出支持embedding的模型
                    all_models = [model["name"] for model in data.get("models", [])]
                    embedding_models = [
                        model for model in all_models
                        if any(keyword in model.lower() for keyword in ["embed", "embedding"])
                    ]
                    self._available_models = embedding_models if embedding_models else [self.model]
                else:
                    self._available_models = [self.model]
        except Exception as e:
            structured_logger.warning(f"Failed to refresh available models: {e}")
            self._available_models = [self.model]

    @property
    def available_models(self) -> List[str]:
        """获取可用模型列表"""
        return self._available_models or [self.model]

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
            # 处理输入
            input_text = request.input
            if isinstance(input_text, list):
                if len(input_text) == 1:
                    input_text = input_text[0]
                else:
                    # Ollama一次只能处理一个文本，需要分别处理
                    return await self._create_embeddings_batch(input_text, request, **kwargs)

            # 构建Ollama请求参数
            ollama_params = {
                "model": request.model,
                "prompt": input_text,
                "options": {}
            }

            # 添加额外选项
            if kwargs:
                ollama_params["options"].update(kwargs)

            # 调用Ollama API
            async with self.session.post(
                f"{self.base_url}/api/embeddings",
                json=ollama_params
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise EmbeddingException(
                        f"Ollama API error: {error_text}",
                        provider=self.provider_name,
                        model=request.model
                    )

                ollama_response = await response.json()

            # 转换响应格式
            return self._convert_response(ollama_response, request)

        except aiohttp.ClientError as e:
            raise EmbeddingException(
                f"Ollama connection error: {e}",
                provider=self.provider_name,
                model=request.model
            )
        except Exception as e:
            raise EmbeddingException(
                f"Embedding creation failed: {e}",
                provider=self.provider_name,
                model=request.model
            )

    async def _create_embeddings_batch(
        self,
        texts: List[str],
        request: EmbeddingRequest,
        **kwargs
    ) -> EmbeddingResponse:
        """
        批量创建文本向量（Ollama需要逐个处理）

        Args:
            texts: 文本列表
            request: 原始请求
            **kwargs: 额外参数

        Returns:
            EmbeddingResponse: 向量化响应
        """
        embeddings = []
        total_tokens = 0

        for i, text in enumerate(texts):
            # 构建单个文本的请求
            ollama_params = {
                "model": request.model,
                "prompt": text,
                "options": {}
            }

            if kwargs:
                ollama_params["options"].update(kwargs)

            # 调用Ollama API
            async with self.session.post(
                f"{self.base_url}/api/embeddings",
                json=ollama_params
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise EmbeddingException(
                        f"Ollama API error at index {i}: {error_text}",
                        provider=self.provider_name,
                        model=request.model
                    )

                ollama_response = await response.json()
                embeddings.append(ollama_response["embedding"])
                total_tokens += self._estimate_tokens(text)

        # 构造响应数据
        data = []
        for i, embedding in enumerate(embeddings):
            embedding_data = EmbeddingData(
                object="embedding",
                embedding=embedding,
                index=i,
            )
            data.append(embedding_data)

        usage = EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
        )

        return EmbeddingResponse(
            object="list",
            data=data,
            model=request.model,
            usage=usage,
        )

    def _convert_response(self, ollama_response: Dict[str, Any], request: EmbeddingRequest) -> EmbeddingResponse:
        """转换Ollama响应为标准格式"""
        embedding = ollama_response.get("embedding", [])

        data = [
            EmbeddingData(
                object="embedding",
                embedding=embedding,
                index=0,
            )
        ]

        # 估算token使用量
        input_text = request.input
        if isinstance(input_text, str):
            prompt_tokens = self._estimate_tokens(input_text)
        else:
            prompt_tokens = sum(self._estimate_tokens(text) for text in input_text)

        usage = EmbeddingUsage(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
        )

        return EmbeddingResponse(
            object="list",
            data=data,
            model=request.model,
            usage=usage,
        )

    def _estimate_tokens(self, text: str) -> int:
        """估算token数量（简单实现）"""
        # 简单估算：约4个字符等于1个token
        return max(1, len(text) // 4)

    def _get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """获取模型能力信息"""
        # 常见Ollama嵌入模型能力配置
        model_configs = {
            "all-minilm": {
                "models": ["all-minilm", "all-minilm:l6-v2", "all-minilm:l12-v2"],
                "max_input_length": 512,
                "embedding_dimensions": {"all-minilm": 384},
                "average_latency_ms": 100,
            },
            "mxbai-embed-large": {
                "models": ["mxbai-embed-large"],
                "max_input_length": 512,
                "embedding_dimensions": {"mxbai-embed-large": 1024},
                "average_latency_ms": 200,
            },
            "nomic-embed-text": {
                "models": ["nomic-embed-text", "nomic-embed-text:v1.5"],
                "max_input_length": 8192,
                "embedding_dimensions": {"nomic-embed-text": 768},
                "average_latency_ms": 150,
            },
            "e5-mistral": {
                "models": ["e5-mistral"],
                "max_input_length": 512,
                "embedding_dimensions": {"e5-mistral": 1024},
                "average_latency_ms": 180,
            },
            "bge-large": {
                "models": ["bge-large", "bge-large:zh", "bge-large:en"],
                "max_input_length": 512,
                "embedding_dimensions": {"bge-large": 1024},
                "average_latency_ms": 220,
            },
            "bge-small": {
                "models": ["bge-small", "bge-small:zh", "bge-small:en"],
                "max_input_length": 512,
                "embedding_dimensions": {"bge-small": 512},
                "average_latency_ms": 120,
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
            "max_input_length": 512,
            "embedding_dimensions": {model: 768},  # 默认768维
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
        # Ollama没有提供token计数API，使用近似计算
        if isinstance(text, str):
            return max(1, len(text) // 4)
        else:
            return [max(1, len(t) // 4) for t in text]

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
                    # 测试嵌入功能
                    test_params = {
                        "model": self.model,
                        "prompt": "test"
                    }

                    async with self.session.post(
                        f"{self.base_url}/api/embeddings",
                        json=test_params
                    ) as embed_response:
                        if embed_response.status == 200:
                            return {
                                "status": "healthy",
                                "provider": self.provider_name,
                                "model": self.model,
                                "base_url": self.base_url,
                                "available_models": self.available_models,
                            }
                        else:
                            return {
                                "status": "degraded",
                                "provider": self.provider_name,
                                "model": self.model,
                                "error": f"Embedding test failed: {embed_response.status}"
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
        structured_logger.info("Ollama Embedding提供商资源清理完成")