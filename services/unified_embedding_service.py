"""
统一嵌入服务
整合BCE、Qwen3和通用嵌入服务的功能到一个统一的架构中
"""

import asyncio
import hashlib
import json
import numpy as np
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModel

from config.loguru_config import get_logger

logger = get_logger("services.unified_embedding")


@dataclass
class EmbeddingProvider:
    """嵌入提供商配置"""

    name: str
    model_type: str  # sentence_transformers, transformers, custom
    model_path: str
    device: str = "cpu"
    dimension: int = 768
    max_length: int = 512
    batch_size: int = 32
    normalize: bool = True
    cache_enabled: bool = True
    timeout: int = 30

    # BCE特有配置
    bce_type: Optional[str] = None  # embedding, rerank

    # Qwen3特有配置
    max_length_qwen: int = 8192
    prefix_tokens: Optional[List[int]] = None
    suffix_tokens: Optional[List[int]] = None
    token_false_id: Optional[int] = None
    token_true_id: Optional[int] = None


@dataclass
class EmbeddingCache:
    """嵌入缓存配置"""

    enabled: bool = True
    ttl: int = 3600  # 秒
    backend: str = "memory"  # memory, redis
    max_size: int = 10000


class UnifiedEmbeddingService:
    """统一嵌入服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger

        # 服务配置
        self.default_provider = config.get("default_provider", "qwen3-embedding")
        self.cache_config = EmbeddingCache(**config.get("cache", {}))

        # 模型管理
        self.providers: Dict[str, EmbeddingProvider] = {}
        self.models: Dict[str, Any] = {}

        # 缓存
        self._cache: Dict[str, Any] = {}

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens": 0,
            "provider_usage": {},
            "total_time": 0.0,
            "avg_time": 0.0
        }

        # 初始化
        self._init_providers()

    def _init_providers(self):
        """初始化嵌入提供商"""
        providers_config = self.config.get("providers", {})

        # 添加BCE提供商
        if "bce" in providers_config:
            self._add_bce_providers(providers_config["bce"])

        # 添加Qwen3提供商
        if "qwen3" in providers_config:
            self._add_qwen3_providers(providers_config["qwen3"])

        # 添加通用提供商
        if "general" in providers_config:
            self._add_general_providers(providers_config["general"])

        logger.info(f"初始化了 {len(self.providers)} 个嵌入提供商")

    def _add_bce_providers(self, bce_config: Dict[str, Any]):
        """添加BCE模型提供商"""
        root_dir = Path(self.config.get("models_root", "models"))
        bce_dir = root_dir / "bce"

        # BCE嵌入模型
        bce_embedding_path = bce_dir / "bce-embedding-base_v1"
        if bce_embedding_path.exists():
            self.providers["bce-embedding"] = EmbeddingProvider(
                name="bce-embedding",
                model_type="sentence_transformers",
                model_path=str(bce_embedding_path),
                dimension=768,
                bce_type="embedding",
                normalize=True
            )

        # BCE重排序模型
        bce_rerank_path = bce_dir / "bce-reranker-base_v1"
        if bce_rerank_path.exists():
            self.providers["bce-rerank"] = EmbeddingProvider(
                name="bce-rerank",
                model_type="sentence_transformers",
                model_path=str(bce_rerank_path),
                dimension=768,
                bce_type="rerank",
                normalize=False
            )

    def _add_qwen3_providers(self, qwen3_config: Dict[str, Any]):
        """添加Qwen3模型提供商"""
        root_dir = Path(self.config.get("models_root", "models"))
        qwen_dir = root_dir / "Qwen"

        # Qwen3嵌入模型
        qwen3_embedding_path = qwen_dir / "Qwen3-Embedding-0.6B"
        if qwen3_embedding_path.exists():
            self.providers["qwen3-embedding"] = EmbeddingProvider(
                name="qwen3-embedding",
                model_type="transformers",
                model_path=str(qwen3_embedding_path),
                dimension=768,
                max_length=8192,
                device="cuda" if torch.cuda.is_available() else "cpu",
                cache_enabled=True
            )

        # Qwen3重排序模型
        qwen3_rerank_path = qwen_dir / "Qwen3-Reranker-0.6B"
        if qwen3_rerank_path.exists():
            self.providers["qwen3-rerank"] = EmbeddingProvider(
                name="qwen3-rerank",
                model_type="transformers",
                model_path=str(qwen3_rerank_path),
                dimension=768,
                max_length=8192,
                device="cuda" if torch.cuda.is_available() else "cpu",
                bce_type="rerank",
                cache_enabled=True
            )

    def _add_general_providers(self, general_config: Dict[str, Any]):
        """添加通用模型提供商"""
        # OpenAI提供商
        if "openai" in general_config:
            self.providers["openai-ada-002"] = EmbeddingProvider(
                name="openai-ada-002",
                model_type="openai",
                model_path="text-embedding-ada-002",
                dimension=1536,
                api_key=general_config["openai"].get("api_key"),
                api_base=general_config["openai"].get("api_base"),
                cache_enabled=True
            )

        # 本地SentenceTransformers模型
        if "sentence_transformers" in general_config:
            for model_config in general_config["sentence_transformers"]:
                model_name = model_config.get("name")
                if model_name:
                    self.providers[model_name] = EmbeddingProvider(
                        name=model_name,
                        model_type="sentence_transformers",
                        **model_config
                    )

    async def initialize(self) -> bool:
        """初始化服务"""
        try:
            # 预加载默认模型
            if self.default_provider in self.providers:
                await self._load_model(self.default_provider)

            logger.info("统一嵌入服务初始化成功")
            return True
        except Exception as e:
            logger.error(f"统一嵌入服务初始化失败: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        self.models.clear()
        self._cache.clear()
        logger.info("统一嵌入服务资源清理完成")

    async def _load_model(self, provider_name: str):
        """加载指定提供商的模型"""
        if provider_name in self.models:
            return self.models[provider_name]

        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"提供商 {provider_name} 不存在")

        try:
            if provider.model_type == "sentence_transformers":
                if provider.bce_type == "rerank":
                    # BCE重排序模型
                    model = CrossEncoder(
                        provider.model_path,
                        max_length=provider.max_length,
                        device=provider.device
                    )
                else:
                    # BCE嵌入模型
                    model = SentenceTransformer(provider.model_path)
                    if provider.device != "cpu":
                        model = model.to(provider.device)

                self.models[provider_name] = model
                logger.info(f"加载 {provider_name} 模型成功")

            elif provider.model_type == "transformers":
                # Qwen3模型
                tokenizer = AutoTokenizer.from_pretrained(
                    provider.model_path,
                    trust_remote_code=True,
                    padding_side='left'
                )
                model = AutoModel.from_pretrained(
                    provider.model_path,
                    trust_remote_code=True
                )

                if provider.device != "cpu":
                    model = model.to(provider.device)
                model.eval()

                # 预处理配置
                if provider.name == "qwen3-rerank":
                    token_false_id = tokenizer.convert_tokens_to_ids("no")
                    token_true_id = tokenizer.convert_tokens_to_ids("yes")

                    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
                    suffix = "<|im_end|>\n<|im_start|>assistant\n"

                    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
                    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

                    # 更新提供商配置
                    provider.prefix_tokens = prefix_tokens
                    provider.suffix_tokens = suffix_tokens
                    provider.token_false_id = token_false_id
                    provider.token_true_id = token_true_id

                self.models[provider_name] = {
                    "model": model,
                    "tokenizer": tokenizer
                }
                logger.info(f"加载 {provider_name} 模型成功")

            elif provider.model_type == "openai":
                # OpenAI模型（延迟加载）
                self.models[provider_name] = "openai_client"
                logger.info(f"OpenAI模型 {provider_name} 将按需加载")

            else:
                logger.warning(f"不支持的模型类型: {provider.model_type}")

        except Exception as e:
            logger.error(f"加载模型 {provider_name} 失败: {e}")
            raise

    async def embed(
        self,
        texts: List[str],
        provider_name: Optional[str] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        生成文本嵌入

        Args:
            texts: 文本列表
            provider_name: 提供商名称
            **kwargs: 额外参数

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []

        provider_name = provider_name or self.default_provider
        start_time = time.time()

        # 更新统计
        self.stats["total_requests"] += 1

        try:
            # 检查缓存
            cache_key = self._get_cache_key(texts, provider_name)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.stats["cache_hits"] += 1
                return cached_result

            # 获取模型
            model = self.models.get(provider_name)
            if not model:
                await self._load_model(provider_name)
                model = self.models[provider_name]

            provider = self.providers[provider_name]

            # 生成嵌入
            if provider.model_type == "sentence_transformers":
                embeddings = await self._embed_with_sentence_transformers(
                    texts, model, provider
                )
            elif provider.model_type == "transformers":
                embeddings = await self._embed_with_transformers(
                    texts, model, provider
                )
            elif provider.model_type == "openai":
                embeddings = await self._embed_with_openai(
                    texts, model, provider
                )
            else:
                raise ValueError(f"不支持的模型类型: {provider.model_type}")

            # 缓存结果
            if self.cache_config.enabled:
                self._set_cache(cache_key, embeddings)

            # 更新统计
            self.stats["cache_misses"] += 1
            self.stats["total_time"] += time.time() - start_time
            self.stats["total_tokens"] += sum(len(text.split()) for text in texts)

            if provider_name not in self.stats["provider_usage"]:
                self.stats["provider_usage"][provider_name] = 0
            self.stats["provider_usage"][provider_name] += len(texts)

            if self.stats["cache_misses"] > 0:
                self.stats["avg_time"] = self.stats["total_time"] / self.stats["cache_misses"]

            return embeddings

        except Exception as e:
            logger.error(f"生成嵌入失败: {e}")
            raise

    async def embed_single(
        self,
        text: str,
        provider_name: Optional[str] = None,
        **kwargs
    ) -> List[float]:
        """生成单个文本的嵌入"""
        embeddings = await self.embed([text], provider_name, **kwargs)
        return embeddings[0] if embeddings else []

    async def _embed_with_sentence_transformers(
        self,
        texts: List[str],
        model,
        provider: EmbeddingProvider
    ) -> List[List[float]]:
        """使用SentenceTransformers生成嵌入"""
        # 分批处理
        all_embeddings = []
        for i in range(0, len(texts), provider.batch_size):
            batch_texts = texts[i:i + provider.batch_size]

            with torch.no_grad():
                if provider.bce_type == "rerank":
                    # BCE重排序模型
                    batch_embeddings = model.predict(
                        [text for text in batch_texts],
                        batch_size=len(batch_texts)
                    )
                else:
                    # BCE嵌入模型
                    batch_embeddings = model.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        normalize_embeddings=provider.normalize
                    )

                if isinstance(batch_embeddings, np.ndarray):
                    all_embeddings.extend(batch_embeddings.tolist())
                else:
                    all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_with_transformers(
        self,
        texts: List[str],
        model_dict,
        provider: EmbeddingProvider
    ) -> List[List[float]]:
        """使用Transformers生成嵌入"""
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]

        # 处理文本
        max_length = provider.max_length_qwen if provider.name.startswith("qwen3") else provider.max_length

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        if provider.device != "cpu":
            inputs = {k: v.to(provider.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            if provider.bce_type == "rerank":
                # Qwen3重排序模型
                # 这里需要实现Qwen3重排序的逻辑
                embeddings = outputs.last_hidden_state[:, -1, :].cpu().numpy()
            else:
                # Qwen3嵌入模型 - 使用最后一层隐藏状态的平均值
                attention_mask = inputs["attention_mask"]
                last_hidden_state = outputs.last_hidden_state

                # 选择最后一个非paddingtoken
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                embeddings = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device),
                    sequence_lengths
                ].cpu().numpy()

            # 标准化嵌入向量
            if provider.normalize:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            return embeddings.tolist()

    async def _embed_with_openai(
        self,
        texts: List[str],
        model,
        provider: EmbeddingProvider
    ) -> List[List[float]]:
        """使用OpenAI生成嵌入"""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=provider.api_key,
                base_url=provider.api_base
            )

            # 分批处理
            all_embeddings = []
            for i in range(0, len(texts), provider.batch_size):
                batch_texts = texts[i:i + provider.batch_size]

                response = await client.embeddings.create(
                    model=provider.model_path,
                    input=batch_texts
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except ImportError:
            raise ImportError("openai 库未安装")
        except Exception as e:
            raise RuntimeError(f"OpenAI嵌入失败: {str(e)}")

    async def rerank(
        self,
        query: str,
        documents: List[str],
        provider_name: Optional[str] = None,
        top_k: int = 10,
        **kwargs
    ) -> List[Tuple[str, float, int]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            provider_name: 提供商名称
            top_k: 返回结果数量

        Returns:
            List[Tuple[str, float, int]]: (文档, 分数, 索引) 元组列表
        """
        if not query or not documents:
            return []

        provider_name = provider_name or "bce-rerank"
        if provider_name not in self.providers:
            raise ValueError(f"重排序提供商 {provider_name} 不存在")

        provider = self.providers[provider_name]

        try:
            # 获取模型
            model = self.models.get(provider_name)
            if not model:
                await self._load_model(provider_name)
                model = self.models[provider_name]

            # 根据不同提供商执行重排序
            if provider.model_type == "sentence_transformers" and provider.bce_type == "rerank":
                # BCE重排序
                scores = model.predict([query] + [doc for doc in documents])

            elif provider.model_type == "transformers" and provider.bce_type == "rerank":
                # Qwen3重排序
                scores = await self._rerank_with_qwen3(query, documents, model, provider)

            else:
                raise ValueError(f"提供商 {provider_name} 不支持重排序功能")

            # 整理结果
            results = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                results.append((doc, float(score), i))

            # 按分数排序
            results.sort(key=lambda x: x[1], reverse=True)

            return results[:top_k]

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return []

    async def _rerank_with_qwen3(
        self,
        query: str,
        documents: List[str],
        model_dict,
        provider: EmbeddingProvider
    ) -> List[float]:
        """使用Qwen3执行重排序"""
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]

        # 格式化输入
        pairs = []
        for doc in documents:
            formatted_input = f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}\nDocument: {doc}"
            pairs.append(formatted_input)

        # 处理输入
        max_length = provider.max_length_qwen - len(provider.prefix_tokens) - len(provider.suffix_tokens)
        inputs = tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=max_length
        )

        # 添加前后缀
        for i in range(len(inputs['input_ids'])):
            inputs['input_ids'][i] = provider.prefix_tokens + inputs['input_ids'][i] + provider.suffix_tokens

        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        if provider.device != "cpu":
            inputs = {k: v.to(provider.device) for k, v in inputs.items()}

        # 计算分数
        with torch.no_grad():
            outputs = model(**inputs).logits[:, -1, :]
            batch_scores = outputs[:, provider.token_true_id]
            false_vector = outputs[:, provider.token_false_id]
            batch_scores = torch.stack([false_vector, batch_scores], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()

        return scores

    def _get_cache_key(self, texts: List[str], provider_name: str) -> str:
        """生成缓存键"""
        content = f"{provider_name}:{json.dumps(texts, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[List[List[float]]]:
        """从缓存获取结果"""
        if not self.cache_config.enabled:
            return None

        cache_item = self._cache.get(cache_key)
        if cache_item and cache_item.get("expires_at", 0) > time.time():
            return cache_item["data"]

        return None

    def _set_cache(self, cache_key: str, data: Any):
        """设置缓存"""
        if not self.cache_config.enabled:
            return

        # 检查缓存大小限制
        if len(self._cache) >= self.cache_config.max_size:
            # 简单的LRU：删除最旧的缓存项
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["created_at"])
            del self._cache[oldest_key]

        self._cache[cache_key] = {
            "data": data,
            "created_at": time.time(),
            "expires_at": time.time() + self.cache_config.ttl
        }

    async def compute_similarity(
        self,
        text1: Union[str, List[float]],
        text2: Union[str, List[float]],
        provider_name: Optional[str] = None
    ) -> float:
        """计算文本相似度"""
        try:
            # 获取嵌入向量
            if isinstance(text1, str):
                embedding1 = await self.embed_single(text1, provider_name)
            else:
                embedding1 = text1

            if isinstance(text2, str):
                embedding2 = await self.embed_single(text2, provider_name)
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
            logger.error(f"计算相似度失败: {e}")
            return 0.0

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        cache_hit_rate = 0.0
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_requests > 0:
            cache_hit_rate = self.stats["cache_hits"] / total_requests

        return {
            **self.stats,
            "cache_hit_rate": cache_hit_rate,
            "available_providers": list(self.providers.keys()),
            "default_provider": self.default_provider,
            "cache_enabled": self.cache_config.enabled,
            "cache_size": len(self._cache)
        }

    async def list_providers(self) -> List[Dict[str, Any]]:
        """列出所有提供商"""
        providers_info = []
        for name, provider in self.providers.items():
            providers_info.append({
                "name": name,
                "model_type": provider.model_type,
                "model_path": provider.model_path,
                "device": provider.device,
                "dimension": provider.dimension,
                "bce_type": provider.bce_type,
                "cache_enabled": provider.cache_enabled,
                "usage_count": self.stats["provider_usage"].get(name, 0)
            })

        return providers_info

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "status": "healthy",
            "providers": {},
            "cache": False,
            "errors": []
        }

        # 检查提供商状态
        for provider_name, provider in self.providers.items():
            try:
                if provider_name in self.models:
                    # 简单测试
                    test_result = await self.embed_single("health check test", provider_name)
                    health_status["providers"][provider_name] = {
                        "status": "healthy",
                        "dimension": len(test_result),
                        "loaded": True
                    }
                else:
                    health_status["providers"][provider_name] = {
                        "status": "not_loaded",
                        "dimension": provider.dimension
                    }
            except Exception as e:
                health_status["providers"][provider_name] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["errors"].append(f"Provider {provider_name}: {str(e)}")

        # 检查缓存状态
        try:
            if self.cache_config.enabled and self._cache:
                # 简单测试
                test_key = "health_test"
                self._set_cache(test_key, ["test"])
                cached_result = self._get_from_cache(test_key)
                health_status["cache"] = cached_result is not None
        except Exception as e:
            health_status["errors"].append(f"Cache: {str(e)}")

        # 总体状态
        if health_status["errors"]:
            health_status["status"] = "degraded"

        return health_status


# 工厂函数
async def create_unified_embedding_service(config: Dict[str, Any]) -> UnifiedEmbeddingService:
    """创建统一嵌入服务实例"""
    service = UnifiedEmbeddingService(config)
    success = await service.initialize()

    if not success:
        raise RuntimeError("统一嵌入服务初始化失败")

    return service