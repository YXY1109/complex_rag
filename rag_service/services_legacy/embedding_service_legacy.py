"""
嵌入服务

基于RAGFlow架构的高性能文本嵌入服务，
支持多种嵌入模型、批处理、缓存等特性。
"""

import asyncio
import hashlib
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import logging
import time
from dataclasses import dataclass, field

from ..interfaces.rag_interface import EmbeddingInterface, EmbeddingException
from ...infrastructure.cache.implementations.redis_client import RedisCacheClient
from ...infrastructure.cache.implementations.memory_client import MemoryCacheClient


@dataclass
class EmbeddingModel:
    """嵌入模型配置。"""

    model_name: str
    model_type: str  # openai, huggingface, local
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    dimension: int = 1536
    max_tokens: int = 8191
    batch_size: int = 100
    timeout: int = 30
    retry_count: int = 3
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingCache:
    """嵌入缓存配置。"""

    enable_cache: bool = True
    cache_ttl: int = 3600  # 秒
    max_cache_size: int = 100000
    cache_backend: str = "redis"  # redis, memory
    redis_config: Dict[str, Any] = field(default_factory=dict)
    memory_config: Dict[str, Any] = field(default_factory=dict)


class EmbeddingService(EmbeddingInterface):
    """嵌入服务。"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化嵌入服务。

        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 模型配置
        self.models: Dict[str, EmbeddingModel] = {}
        self.default_model: Optional[str] = None

        # 缓存配置
        self.cache_config = EmbeddingCache(**config.get("cache", {}))
        self.cache_client: Optional[Union[RedisCacheClient, MemoryCacheClient]] = None

        # 性能配置
        self.batch_size = config.get("batch_size", 100)
        self.max_concurrent_requests = config.get("max_concurrent_requests", 10)
        self.request_timeout = config.get("request_timeout", 30)

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens": 0,
            "total_embed_time": 0.0,
            "average_embed_time": 0.0,
            "model_usage": {}
        }

        # 初始化模型
        self._init_models()

    def _init_models(self) -> None:
        """初始化嵌入模型。"""
        models_config = self.config.get("models", {})

        for model_name, model_config in models_config.items():
            model = EmbeddingModel(
                model_name=model_name,
                **model_config
            )
            self.models[model_name] = model

            # 设置默认模型
            if not self.default_model or model_config.get("default", False):
                self.default_model = model_name

        if not self.default_model and self.models:
            self.default_model = list(self.models.keys())[0]

        self.logger.info(f"初始化了 {len(self.models)} 个嵌入模型，默认模型: {self.default_model}")

    async def initialize(self) -> bool:
        """初始化嵌入服务。"""
        try:
            # 初始化缓存
            if self.cache_config.enable_cache:
                await self._init_cache()

            # 预热模型
            if self.default_model:
                await self._warmup_model(self.default_model)

            self.logger.info("嵌入服务初始化成功")
            return True

        except Exception as e:
            self.logger.error(f"嵌入服务初始化失败: {e}")
            return False

    async def cleanup(self) -> None:
        """清理嵌入服务资源。"""
        try:
            if self.cache_client:
                await self.cache_client.disconnect()
                self.cache_client = None

            self.logger.info("嵌入服务资源清理完成")

        except Exception as e:
            self.logger.error(f"嵌入服务清理失败: {e}")

    async def _init_cache(self) -> None:
        """初始化缓存客户端。"""
        try:
            if self.cache_config.cache_backend == "redis":
                redis_config = self.cache_config.redis_config or {}
                self.cache_client = RedisCacheClient(redis_config)
                await self.cache_client.connect()
                self.logger.info("Redis缓存客户端初始化成功")

            elif self.cache_config.cache_backend == "memory":
                memory_config = self.cache_config.memory_config or {}
                self.cache_client = MemoryCacheClient(memory_config)
                await self.cache_client.connect()
                self.logger.info("内存缓存客户端初始化成功")

            else:
                self.logger.warning(f"不支持的缓存后端: {self.cache_config.cache_backend}")

        except Exception as e:
            self.logger.error(f"缓存初始化失败: {e}")
            self.cache_config.enable_cache = False

    async def _warmup_model(self, model_name: str) -> None:
        """预热模型。"""
        try:
            model = self.models.get(model_name)
            if not model:
                return

            # 使用简单文本预热
            test_text = "This is a test text for model warmup."
            await self._embed_with_model([test_text], model)
            self.logger.info(f"模型 {model_name} 预热完成")

        except Exception as e:
            self.logger.warning(f"模型 {model_name} 预热失败: {e}")

    async def embed(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        """
        生成文本嵌入。

        Args:
            texts: 文本列表
            model_name: 模型名称

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []

        model_name = model_name or self.default_model
        if not model_name:
            raise EmbeddingException("未指定嵌入模型")

        model = self.models.get(model_name)
        if not model:
            raise EmbeddingException(f"模型 {model_name} 不存在")

        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            # 检查缓存
            embeddings = await self._get_cached_embeddings(texts, model_name)
            cache_hit_count = len(embeddings)

            if cache_hit_count == len(texts):
                self.stats["cache_hits"] += len(texts)
                return embeddings

            # 计算未缓存的文本
            cached_indices = set()
            if cache_hit_count > 0:
                # 重新获取缓存结果以确定哪些文本已缓存
                cached_results = {}
                for i, text in enumerate(texts):
                    cache_key = self._get_cache_key(text, model_name)
                    cached_result = await self.cache_client.get(cache_key) if self.cache_client else None
                    if cached_result:
                        cached_results[i] = cached_result
                        cached_indices.add(i)

                # 重建缓存结果
                embeddings = [None] * len(texts)
                for i, cached_result in cached_results.items():
                    embeddings[i] = cached_result

            # 识别需要计算的文本
            uncached_texts = []
            uncached_indices = []
            for i, text in enumerate(texts):
                if i not in cached_indices:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # 批量计算未缓存的嵌入
            if uncached_texts:
                uncached_embeddings = await self._embed_with_model(uncached_texts, model)

                # 更新缓存和结果
                for i, embedding in enumerate(uncached_embeddings):
                    original_index = uncached_indices[i]
                    embeddings[original_index] = embedding

                    # 缓存结果
                    if self.cache_client:
                        cache_key = self._get_cache_key(uncached_texts[i], model_name)
                        await self.cache_client.set(
                            cache_key,
                            embedding,
                            ttl=self.cache_config.cache_ttl
                        )

            # 更新统计
            embed_time = time.time() - start_time
            self.stats["total_embed_time"] += embed_time
            self.stats["total_embed_time"] = embed_time
            self.stats["total_tokens"] += sum(len(text.split()) for text in texts)
            self.stats["cache_misses"] += len(uncached_texts)

            # 更新模型使用统计
            if model_name not in self.stats["model_usage"]:
                self.stats["model_usage"][model_name] = 0
            self.stats["model_usage"][model_name] += len(texts)

            # 更新平均时间
            total_completed = self.stats["cache_hits"] + self.stats["cache_misses"]
            if total_completed > 0:
                self.stats["average_embed_time"] = self.stats["total_embed_time"] / total_completed

            return embeddings

        except Exception as e:
            self.logger.error(f"生成嵌入失败: {e}")
            raise EmbeddingException(f"生成嵌入失败: {str(e)}")

    async def embed_single(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """
        生成单个文本的嵌入。

        Args:
            text: 文本字符串
            model_name: 模型名称

        Returns:
            List[float]: 嵌入向量
        """
        embeddings = await self.embed([text], model_name)
        return embeddings[0] if embeddings else []

    async def _embed_with_model(self, texts: List[str], model: EmbeddingModel) -> List[List[float]]:
        """使用指定模型生成嵌入。"""
        if model.model_type == "openai":
            return await self._embed_with_openai(texts, model)
        elif model.model_type == "huggingface":
            return await self._embed_with_huggingface(texts, model)
        elif model.model_type == "local":
            return await self._embed_with_local(texts, model)
        else:
            raise EmbeddingException(f"不支持的模型类型: {model.model_type}")

    async def _embed_with_openai(self, texts: List[str], model: EmbeddingModel) -> List[List[float]]:
        """使用OpenAI模型生成嵌入。"""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=model.api_key,
                base_url=model.api_base
            )

            # 分批处理
            embeddings = []
            for i in range(0, len(texts), model.batch_size):
                batch_texts = texts[i:i + model.batch_size]

                response = await client.embeddings.create(
                    model=model.model_name,
                    input=batch_texts
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            return embeddings

        except ImportError:
            raise EmbeddingException("openai 库未安装")
        except Exception as e:
            self.logger.error(f"OpenAI嵌入失败: {e}")
            raise EmbeddingException(f"OpenAI嵌入失败: {str(e)}")

    async def _embed_with_huggingface(self, texts: List[str], model: EmbeddingModel) -> List[List[float]]:
        """使用HuggingFace模型生成嵌入。"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # 检查CUDA可用性
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 加载模型
            transformer = SentenceTransformer(model.model_name, device=device)

            # 分批处理
            embeddings = []
            for i in range(0, len(texts), model.batch_size):
                batch_texts = texts[i:i + model.batch_size]

                with torch.no_grad():
                    batch_embeddings = transformer.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        convert_to_numpy=True
                    )

                embeddings.extend(batch_embeddings.tolist())

            return embeddings

        except ImportError:
            raise EmbeddingException("sentence_transformers 库未安装")
        except Exception as e:
            self.logger.error(f"HuggingFace嵌入失败: {e}")
            raise EmbeddingException(f"HuggingFace嵌入失败: {str(e)}")

    async def _embed_with_local(self, texts: List[str], model: EmbeddingModel) -> List[List[float]]:
        """使用本地模型生成嵌入。"""
        try:
            # 这里可以实现本地模型的嵌入逻辑
            # 例如使用ONNX、TensorFlow等模型
            self.logger.warning("本地嵌入模型功能待实现")

            # 简化实现：返回随机向量
            return [
                [np.random.normal(0, 1) for _ in range(model.dimension)]
                for _ in texts
            ]

        except Exception as e:
            self.logger.error(f"本地嵌入失败: {e}")
            raise EmbeddingException(f"本地嵌入失败: {str(e)}")

    async def _get_cached_embeddings(self, texts: List[str], model_name: str) -> List[List[float]]:
        """获取缓存的嵌入。"""
        if not self.cache_client or not self.cache_config.enable_cache:
            return []

        embeddings = []
        for text in texts:
            cache_key = self._get_cache_key(text, model_name)
            cached_embedding = await self.cache_client.get(cache_key)
            if cached_embedding:
                embeddings.append(cached_embedding)

        return embeddings

    def _get_cache_key(self, text: str, model_name: str) -> str:
        """生成缓存键。"""
        # 使用文本和模型名称的哈希作为缓存键
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    async def compute_similarity(
        self,
        text1: Union[str, List[float]],
        text2: Union[str, List[float]],
        model_name: Optional[str] = None
    ) -> float:
        """
        计算文本相似度。

        Args:
            text1: 文本1或嵌入向量1
            text2: 文本2或嵌入向量2
            model_name: 模型名称

        Returns:
            float: 相似度分数（0-1）
        """
        try:
            # 获取嵌入向量
            if isinstance(text1, str):
                embedding1 = await self.embed_single(text1, model_name)
            else:
                embedding1 = text1

            if isinstance(text2, str):
                embedding2 = await self.embed_single(text2, model_name)
            else:
                embedding2 = text2

            # 计算余弦相似度
            embedding1_np = np.array(embedding1)
            embedding2_np = np.array(embedding2)

            dot_product = np.dot(embedding1_np, embedding2_np)
            norm1 = np.linalg.norm(embedding1_np)
            norm2 = np.linalg.norm(embedding2_np)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            self.logger.error(f"计算相似度失败: {e}")
            return 0.0

    async def batch_compute_similarity(
        self,
        query_text: str,
        candidate_texts: List[str],
        model_name: Optional[str] = None,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        批量计算文本相似度。

        Args:
            query_text: 查询文本
            candidate_texts: 候选文本列表
            model_name: 模型名称
            top_k: 返回top_k结果

        Returns:
            List[Tuple[str, float]]: (文本, 相似度分数) 列表
        """
        try:
            # 获取查询文本的嵌入
            query_embedding = await self.embed_single(query_text, model_name)

            # 获取候选文本的嵌入
            candidate_embeddings = await self.embed(candidate_texts, model_name)

            # 计算相似度
            similarities = []
            query_np = np.array(query_embedding)

            for i, candidate_embedding in enumerate(candidate_embeddings):
                candidate_np = np.array(candidate_embedding)
                similarity = np.dot(query_np, candidate_np) / (
                    np.linalg.norm(query_np) * np.linalg.norm(candidate_np)
                )
                similarities.append((candidate_texts[i], float(similarity)))

            # 排序并返回top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            self.logger.error(f"批量计算相似度失败: {e}")
            return []

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

            model = EmbeddingModel(**model_config)
            self.models[model_name] = model

            # 如果是第一个模型，设为默认模型
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

        # 如果是默认模型，需要重新选择
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
            "model_type": model.model_type,
            "dimension": model.dimension,
            "max_tokens": model.max_tokens,
            "batch_size": model.batch_size,
            "timeout": model.timeout,
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

    async def clear_cache(self, model_name: Optional[str] = None) -> bool:
        """
        清理缓存。

        Args:
            model_name: 模型名称（可选）

        Returns:
            bool: 清理是否成功
        """
        try:
            if not self.cache_client:
                return False

            if model_name:
                # 清理特定模型的缓存
                # 这里需要实现模式匹配删除
                self.logger.info(f"清理模型 {model_name} 缓存功能待实现")
            else:
                # 清理所有缓存
                await self.cache_client.clear()

            self.logger.info("缓存清理完成")
            return True

        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息。"""
        cache_hit_rate = 0.0
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_requests > 0:
            cache_hit_rate = self.stats["cache_hits"] / total_requests

        return {
            **self.stats,
            "cache_hit_rate": cache_hit_rate,
            "available_models": list(self.models.keys()),
            "default_model": self.default_model,
            "cache_enabled": self.cache_config.enable_cache
        }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查。"""
        health_status = {
            "status": "healthy",
            "models": {},
            "cache": False,
            "errors": []
        }

        # 检查模型状态
        for model_name, model in self.models.items():
            try:
                # 简单测试模型
                test_embedding = await self._embed_with_model(["test"], model)
                health_status["models"][model_name] = {
                    "status": "healthy",
                    "dimension": len(test_embedding[0]) if test_embedding else model.dimension
                }
            except Exception as e:
                health_status["models"][model_name] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["errors"].append(f"Model {model_name}: {str(e)}")

        # 检查缓存状态
        try:
            if self.cache_client:
                # 简单测试缓存
                await self.cache_client.set("health_check", "test", ttl=1)
                cached_value = await self.cache_client.get("health_check")
                health_status["cache"] = cached_value == "test"
        except Exception as e:
            health_status["errors"].append(f"Cache: {str(e)}")

        # 总体状态
        if health_status["errors"]:
            health_status["status"] = "degraded"

        return health_status