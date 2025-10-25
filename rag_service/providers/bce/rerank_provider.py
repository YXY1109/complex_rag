"""
百度文心Rerank提供商

实现百度文心的文档重排序服务。
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from typing import List, Dict, Any, Optional, Tuple

import aiohttp

from ...interfaces.rerank_interface import (
    RerankInterface,
    RerankConfig,
    RerankProviderCapabilities,
    RerankRequest,
    RerankResponse,
    RerankDocument
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("rag_service.providers.bce_rerank")


class BCERerankProvider(RerankInterface):
    """
    百度文心Rerank提供商实现

    提供百度文心的文档重排序服务。
    """

    def __init__(self, config: RerankConfig):
        """
        初始化百度文心Rerank提供商

        Args:
            config: Rerank配置
        """
        super().__init__(config)
        self.base_url = config.base_url or "https://aip.baidubce.com/rpc/2.0/ai_custom/v1"
        self.api_key = config.api_key
        self.secret_key = config.organization  # 使用organization字段存储secret_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[float] = None
        self._capabilities: Optional[RerankProviderCapabilities] = None

    @property
    def capabilities(self) -> RerankProviderCapabilities:
        """获取百度文心Rerank提供商能力信息"""
        if self._capabilities is None:
            model_capabilities = self._get_model_capabilities(self.model)
            self._capabilities = RerankProviderCapabilities(
                supported_models=model_capabilities["models"],
                max_query_length=model_capabilities["max_query_length"],
                max_document_length=model_capabilities["max_document_length"],
                max_documents_per_request=model_capabilities["max_documents_per_request"],
                max_top_k=model_capabilities["max_top_k"],
                supports_chunking=False,
                supports_overlap=False,
                supports_custom_top_k=True,
                supports_return_documents=True,
                supports_scoring_only=True,
                pricing_per_1k_tokens=model_capabilities["pricing_per_1k_tokens"],
                currency="CNY",
                average_latency_ms=model_capabilities["average_latency_ms"],
            )
        return self._capabilities

    async def initialize(self) -> None:
        """初始化百度文心Rerank客户端"""
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
                "百度文心Rerank提供商初始化成功",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.base_url,
                }
            )

        except Exception as e:
            structured_logger.error(
                f"百度文心Rerank提供商初始化失败: {e}",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "base_url": self.base_url,
                    "error": str(e),
                }
            )
            raise Exception(f"Failed to initialize BCE rerank provider: {e}")

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
                    raise Exception(f"Failed to get access token: {error_text}")

                token_data = await response.json()
                self.access_token = token_data["access_token"]
                self.token_expires_at = time.time() + token_data["expires_in"] - 300  # 提前5分钟刷新

        except Exception as e:
            raise Exception(f"Failed to refresh access token: {e}")

    async def _get_valid_access_token(self) -> str:
        """获取有效的访问令牌"""
        if not self.access_token or not self.token_expires_at or time.time() >= self.token_expires_at:
            await self._refresh_access_token()
        return self.access_token

    async def _test_connection(self) -> None:
        """测试百度文心Rerank连接"""
        try:
            # 发送一个简单的测试请求
            test_request = RerankRequest(
                model=self.model,
                query="What is artificial intelligence?",
                documents=[
                    "Artificial intelligence is a branch of computer science.",
                    "Machine learning is a subset of AI.",
                ],
                top_k=2
            )
            await self.rerank(test_request)
        except Exception as e:
            raise Exception(f"BCE rerank connection test failed: {e}")

    async def rerank(
        self,
        request: RerankRequest,
        **kwargs
    ) -> RerankResponse:
        """
        文档重排序

        Args:
            request: 重排序请求
            **kwargs: 额外参数

        Returns:
            RerankResponse: 重排序响应
        """
        try:
            # 获取有效的访问令牌
            access_token = await self._get_valid_access_token()

            # 验证文档数量
            if len(request.documents) > self.capabilities.max_documents_per_request:
                raise ValueError(f"Too many documents: {len(request.documents)} > {self.capabilities.max_documents_per_request}")

            # 构建百度文心请求参数
            bce_params = {
                "query": request.query,
                "passages": request.documents,
                "top_n": request.top_k or self.default_top_k,
                "user_id": request.user or "default",
            }

            # 添加额外参数
            if kwargs:
                bce_params.update(kwargs)

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
            raise Exception(f"BCE rerank connection error: {e}")
        except Exception as e:
            raise Exception(f"Rerank failed: {e}")

    def _convert_response(self, bce_response: Dict[str, Any], request: RerankRequest) -> RerankResponse:
        """转换百度文心响应为标准格式"""
        if "results" not in bce_response:
            raise Exception("Invalid response format from BCE rerank API")

        results = []
        for i, result in enumerate(bce_response["results"]):
            rerank_doc = RerankDocument(
                index=i,
                relevance_score=result.get("score", 0.0),
                document=result.get("text", "") if request.return_documents else None,
                text=result.get("text", "") if request.return_text else None,
            )
            results.append(rerank_doc)

        return RerankResponse(
            object="rerank",
            model=request.model,
            results=results,
            id=f"bce-rerank-{int(time.time())}",
            created=int(time.time()),
            usage=bce_response.get("usage"),
        )

    def _get_endpoint(self, model: str) -> str:
        """根据模型名称获取API端点"""
        # 百度文心重排序模型端点映射
        endpoint_mapping = {
            "bce-reranker-base": "wenxinworkshop/reranker/bce-reranker-base",
            "bce-reranker-large": "wenxinworkshop/reranker/bce-reranker-large",
        }

        # 查找匹配的端点
        for model_pattern, endpoint in endpoint_mapping.items():
            if model_pattern in model:
                return endpoint

        # 默认端点
        return "wenxinworkshop/reranker/bce-reranker-base"

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
            raise Exception(f"BCE rerank rate limit exceeded: {error_message}")
        elif status_code == 404:
            raise Exception(f"Model {model} not available: {error_message}")
        else:
            raise Exception(f"BCE rerank API error: {error_message}")

    def _get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """获取模型能力信息"""
        # 百度文心重排序模型能力配置
        model_configs = {
            "bce-reranker-base": {
                "models": ["bce-reranker-base"],
                "max_query_length": 512,
                "max_document_length": 8000,
                "max_documents_per_request": 100,
                "max_top_k": 100,
                "pricing_per_1k_tokens": 0.001,  # CNY
                "average_latency_ms": 300,
            },
            "bce-reranker-large": {
                "models": ["bce-reranker-large"],
                "max_query_length": 512,
                "max_document_length": 8000,
                "max_documents_per_request": 50,
                "max_top_k": 50,
                "pricing_per_1k_tokens": 0.002,  # CNY
                "average_latency_ms": 500,
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
            "max_query_length": 512,
            "max_document_length": 8000,
            "max_documents_per_request": 100,
            "max_top_k": 100,
            "pricing_per_1k_tokens": 0.001,
            "average_latency_ms": 300,
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
        """清理百度文心Rerank客户端资源"""
        if self.session:
            await self.session.close()
            self.session = None
        self.access_token = None
        self.token_expires_at = None
        structured_logger.info("百度文心Rerank提供商资源清理完成")