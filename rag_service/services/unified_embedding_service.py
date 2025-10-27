"""
统一嵌入服务

整合BCE、Qwen3、Generic三个嵌入服务的统一架构，
支持可插拔的模型后端、智能缓存、生命周期管理等功能。
"""

import asyncio
import hashlib
import json
import logging
import time
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModel

from ..interfaces.rag_interface import EmbeddingInterface, EmbeddingException


class ModelType(Enum):
    """模型类型枚举"""
    BCE = "bce"
    QWEN3 = "qwen3"
    OPENAI = "openai"
    GENERIC = "generic"
    SENTENCE_TRANSFORMER = "sentence_transformer"


class DeviceType(Enum):
    """设备类型枚举"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


@dataclass
class EmbeddingModelConfig:
    """嵌入模型配置"""
    model_name: str
    model_type: ModelType
    model_path: Optional[str] = None
    device: DeviceType = DeviceType.CPU
    use_gpu: bool = False
    dimension: int = 1536
    max_length: int = 512
    batch_size: int = 32
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_params: Dict[str, Any] = field(default_factory=dict)
    cache_enabled: bool = True
    priority: int = 1  # 优先级，数字越小优先级越高


@dataclass
class EmbeddingRequest:
    """嵌入请求"""
    texts: Union[str, List[str]]
    model_name: Optional[str] = None
    normalize: bool = True
    batch_size: Optional[int] = None
    use_cache: bool = True


@dataclass
class EmbeddingResponse:
    """嵌入响应"""
    embeddings: List[List[float]]
    model_name: str
    dimension: int
    usage: Dict[str, int] = field(default_factory=dict)
    cached_count: int = 0
    processing_time: float = 0.0


@dataclass
class SimilarityRequest:
    """相似度计算请求"""
    text1: str
    text2: str
    model_name: Optional[str] = None


@dataclass
class SimilarityResponse:
    """相似度计算响应"""
    similarity_score: float
    model_name: str
    processing_time: float = 0.0


class EmbeddingBackend(ABC):
    """嵌入后端抽象基类"""

    def __init__(self, config: EmbeddingModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_loaded = False

    @abstractmethod
    async def load_model(self) -> None:
        """加载模型"""
        pass

    @abstractmethod
    async def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """生成嵌入向量"""
        pass

    @abstractmethod
    async def compute_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """卸载模型释放内存"""
        pass


class BCEEmbeddingBackend(EmbeddingBackend):
    """BCE嵌入后端"""

    async def load_model(self) -> None:
        """加载BCE模型"""
        try:
            if not self.config.model_path:
                raise EmbeddingException("BCE模型路径未配置")

            device = self._get_device()
            self.model = SentenceTransformer(self.config.model_path)
            self.model = self.model.to(device)
            self.device = device
            self.is_loaded = True

            logging.info(f"BCE嵌入模型加载成功: {self.config.model_path}")

        except Exception as e:
            logging.error(f"BCE模型加载失败: {str(e)}")
            raise EmbeddingException(f"BCE模型加载失败: {str(e)}")

    async def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """生成BCE嵌入向量"""
        if not self.is_loaded:
            await self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        # 批量处理
        embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                normalize_embeddings=self.config.model_params.get('normalize', True),
                batch_size= len(batch_texts),
                convert_to_numpy=True
            )
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings) if len(embeddings) > 1 else embeddings[0]

    async def compute_similarity(self, text1: str, text2: str) -> float:
        """计算BCE文本相似度"""
        embeddings = await self.embed([text1, text2])

        # 计算余弦相似度
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        return float(similarity)

    def unload_model(self) -> None:
        """卸载BCE模型"""
        if self.model:
            del self.model
            self.model = None
        self.is_loaded = False

    def _get_device(self) -> str:
        """获取设备"""
        if self.config.use_gpu and torch.cuda.is_available():
            return "cuda"
        return "cpu"


class Qwen3EmbeddingBackend(EmbeddingBackend):
    """Qwen3嵌入后端"""

    async def load_model(self) -> None:
        """加载Qwen3模型"""
        try:
            if not self.config.model_path:
                raise EmbeddingException("Qwen3模型路径未配置")

            device = self._get_device()

            # 加载分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                padding_side='left'
            )
            self.model = AutoModel.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            self.model = self.model.to(device)
            self.model.eval()

            self.device = device
            self.is_loaded = True

            logging.info(f"Qwen3嵌入模型加载成功: {self.config.model_path}")

        except Exception as e:
            logging.error(f"Qwen3模型加载失败: {str(e)}")
            raise EmbeddingException(f"Qwen3模型加载失败: {str(e)}")

    async def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """生成Qwen3嵌入向量"""
        if not self.is_loaded:
            await self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        for text in texts:
            # 处理单个文本
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用最后一层的平均池化
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                if self.config.model_params.get('normalize', True):
                    embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)

            embeddings.append(embedding.flatten())

        return np.array(embeddings)

    async def compute_similarity(self, text1: str, text2: str) -> float:
        """计算Qwen3文本相似度"""
        embeddings = await self.embed([text1, text2])

        # 计算余弦相似度
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        return float(similarity)

    def unload_model(self) -> None:
        """卸载Qwen3模型"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        self.is_loaded = False

    def _get_device(self) -> str:
        """获取设备"""
        if self.config.use_gpu and torch.cuda.is_available():
            return "cuda"
        return "cpu"


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI嵌入后端"""

    def __init__(self, config: EmbeddingModelConfig):
        super().__init__(config)
        self._async_client = None

    async def load_model(self) -> None:
        """初始化OpenAI客户端"""
        try:
            import openai

            if not self.config.api_key:
                raise EmbeddingException("OpenAI API密钥未配置")

            self._async_client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base
            )
            self.is_loaded = True

            logging.info("OpenAI嵌入客户端初始化成功")

        except ImportError:
            raise EmbeddingException("未安装openai包")
        except Exception as e:
            logging.error(f"OpenAI客户端初始化失败: {str(e)}")
            raise EmbeddingException(f"OpenAI客户端初始化失败: {str(e)}")

    async def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """生成OpenAI嵌入向量"""
        if not self.is_loaded:
            await self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        try:
            response = await self._async_client.embeddings.create(
                model=self.config.model_name,
                input=texts
            )

            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings)

        except Exception as e:
            logging.error(f"OpenAI嵌入生成失败: {str(e)}")
            raise EmbeddingException(f"OpenAI嵌入生成失败: {str(e)}")

    async def compute_similarity(self, text1: str, text2: str) -> float:
        """计算OpenAI文本相似度"""
        embeddings = await self.embed([text1, text2])

        # 计算余弦相似度
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        return float(similarity)

    def unload_model(self) -> None:
        """清理OpenAI客户端"""
        self._async_client = None
        self.is_loaded = False


class UnifiedEmbeddingService:
    """统一嵌入服务"""

    def __init__(self, config: Dict[str, Any]):
        """初始化统一嵌入服务"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 模型管理
        self.backends: Dict[str, EmbeddingBackend] = {}
        self.model_configs: Dict[str, EmbeddingModelConfig] = {}
        self.default_model: Optional[str] = None

        # 缓存管理
        self.cache_enabled = config.get("cache", {}).get("enabled", True)
        self.cache_ttl = config.get("cache", {}).get("ttl", 3600)
        self.cache: Dict[str, Tuple[np.ndarray, float]] = {}

        # 性能配置
        self.max_concurrent_requests = config.get("max_concurrent_requests", 10)
        self.default_batch_size = config.get("default_batch_size", 32)

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_texts": 0,
            "total_processing_time": 0.0,
            "model_loads": 0,
            "model_unloads": 0
        }

        # 初始化模型配置
        self._initialize_model_configs()

    def _initialize_model_configs(self) -> None:
        """初始化模型配置"""
        model_configs = self.config.get("models", {})

        for name, config_data in model_configs.items():
            # 处理设备配置
            device = DeviceType.CPU
            use_gpu = False

            if "device" in config_data:
                device = DeviceType(config_data["device"])
                use_gpu = device == DeviceType.CUDA and torch.cuda.is_available()

            # 获取模型路径
            model_path = config_data.get("model_path")
            if not model_path:
                # 尝试从默认路径获取
                model_type = ModelType(config_data["model_type"])
                model_path = self._get_default_model_path(model_type)

            self.model_configs[name] = EmbeddingModelConfig(
                model_name=name,
                model_type=ModelType(config_data["model_type"]),
                model_path=model_path,
                device=device,
                use_gpu=use_gpu,
                dimension=config_data.get("dimension", 1536),
                max_length=config_data.get("max_length", 512),
                batch_size=config_data.get("batch_size", 32),
                api_key=config_data.get("api_key"),
                api_base=config_data.get("api_base"),
                model_params=config_data.get("model_params", {}),
                cache_enabled=config_data.get("cache_enabled", True),
                priority=config_data.get("priority", 1)
            )

        # 设置默认模型
        default_model_name = self.config.get("default_model")
        if default_model_name and default_model_name in self.model_configs:
            self.default_model = default_model_name
        elif self.model_configs:
            # 选择优先级最高的模型作为默认模型
            self.default_model = min(
                self.model_configs.keys(),
                key=lambda x: self.model_configs[x].priority
            )

        self.logger.info(f"默认嵌入模型: {self.default_model}")

    def _get_default_model_path(self, model_type: ModelType) -> Optional[str]:
        """获取默认模型路径"""
        base_dir = os.path.join(os.path.dirname(__file__), "..", "models")

        model_paths = {
            ModelType.BCE: os.path.join(base_dir, "bce-embedding-base_v1"),
            ModelType.QWEN3: os.path.join(base_dir, "Qwen", "Qwen3-Embedding-0.6B"),
            ModelType.GENERIC: os.path.join(base_dir, "generic-embedding"),
        }

        path = model_paths.get(model_type)
        return path if os.path.exists(path) else None

    async def initialize(self) -> None:
        """初始化服务"""
        self.logger.info("正在初始化统一嵌入服务...")

        # 预加载默认模型
        if self.default_model:
            try:
                await self._ensure_model_loaded(self.default_model)
                self.logger.info(f"默认模型 {self.default_model} 预加载完成")
            except Exception as e:
                self.logger.warning(f"默认模型预加载失败: {str(e)}")

        self.logger.info("统一嵌入服务初始化完成")

    async def _ensure_model_loaded(self, model_name: str) -> EmbeddingBackend:
        """确保模型已加载"""
        if model_name not in self.backends:
            if model_name not in self.model_configs:
                raise EmbeddingException(f"未知的模型: {model_name}")

            config = self.model_configs[model_name]

            # 创建后端实例
            if config.model_type == ModelType.BCE:
                backend = BCEEmbeddingBackend(config)
            elif config.model_type == ModelType.QWEN3:
                backend = Qwen3EmbeddingBackend(config)
            elif config.model_type == ModelType.OPENAI:
                backend = OpenAIEmbeddingBackend(config)
            else:
                raise EmbeddingException(f"不支持的模型类型: {config.model_type}")

            # 加载模型
            await backend.load_model()
            self.backends[model_name] = backend
            self.stats["model_loads"] += 1

            self.logger.info(f"模型 {model_name} 加载完成")

        return self.backends[model_name]

    def _get_cache_key(self, text: str, model_name: str) -> str:
        """生成缓存键"""
        content = f"{text}:{model_name}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """生成嵌入向量"""
        start_time = time.time()

        # 处理请求参数
        model_name = request.model_name or self.default_model
        if not model_name:
            raise EmbeddingException("未指定模型且未配置默认模型")

        texts = request.texts if isinstance(request.texts, list) else [request.texts]

        try:
            # 确保模型已加载
            backend = await self._ensure_model_loaded(model_name)

            # 缓存检查
            cache_results = []
            uncached_texts = []
            uncached_indices = []

            if request.use_cache and self.cache_enabled:
                for i, text in enumerate(texts):
                    cache_key = self._get_cache_key(text, model_name)
                    cached_data = self.cache.get(cache_key)

                    if cached_data:
                        embedding, timestamp = cached_data
                        if time.time() - timestamp < self.cache_ttl:
                            cache_results.append((i, embedding))
                            self.stats["cache_hits"] += 1
                            continue

                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self.stats["cache_misses"] += 1
            else:
                uncached_texts = texts
                uncached_indices = list(range(len(texts)))

            # 生成未缓存的嵌入
            uncached_embeddings = []
            if uncached_texts:
                batch_size = request.batch_size or self.default_batch_size

                for batch_start in range(0, len(uncached_texts), batch_size):
                    batch_texts = uncached_texts[batch_start:batch_start + batch_size]
                    batch_embeddings = await backend.embed(batch_texts)

                    if len(batch_embeddings.shape) == 1:
                        batch_embeddings = batch_embeddings.reshape(1, -1)

                    uncached_embeddings.extend(batch_embeddings.tolist())

                    # 更新缓存
                    if request.use_cache and self.cache_enabled:
                        for text, embedding in zip(batch_texts, batch_embeddings):
                            cache_key = self._get_cache_key(text, model_name)
                            self.cache[cache_key] = (embedding, time.time())

            # 合并结果
            embeddings = [None] * len(texts)

            # 填入缓存结果
            for i, embedding in cache_results:
                embeddings[i] = embedding.tolist()

            # 填入新生成的结果
            for idx, embedding in zip(uncached_indices, uncached_embeddings):
                embeddings[idx] = embedding

            # 创建响应
            processing_time = time.time() - start_time
            config = self.model_configs[model_name]

            response = EmbeddingResponse(
                embeddings=embeddings,
                model_name=model_name,
                dimension=config.dimension,
                usage={
                    "prompt_tokens": sum(len(text.split()) for text in texts),
                    "total_tokens": sum(len(text.split()) for text in texts)
                },
                cached_count=len(cache_results),
                processing_time=processing_time
            )

            # 更新统计
            self.stats["total_requests"] += 1
            self.stats["total_texts"] += len(texts)
            self.stats["total_processing_time"] += processing_time

            return response

        except Exception as e:
            self.logger.error(f"嵌入向量生成失败: {str(e)}")
            raise EmbeddingException(f"嵌入向量生成失败: {str(e)}")

    async def compute_similarity(self, request: SimilarityRequest) -> SimilarityResponse:
        """计算文本相似度"""
        start_time = time.time()

        # 处理请求参数
        model_name = request.model_name or self.default_model
        if not model_name:
            raise EmbeddingException("未指定模型且未配置默认模型")

        try:
            # 确保模型已加载
            backend = await self._ensure_model_loaded(model_name)

            # 计算相似度
            similarity = await backend.compute_similarity(request.text1, request.text2)

            processing_time = time.time() - start_time

            response = SimilarityResponse(
                similarity_score=similarity,
                model_name=model_name,
                processing_time=processing_time
            )

            return response

        except Exception as e:
            self.logger.error(f"文本相似度计算失败: {str(e)}")
            raise EmbeddingException(f"文本相似度计算失败: {str(e)}")

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "status": "healthy",
            "loaded_models": list(self.backends.keys()),
            "available_models": list(self.model_configs.keys()),
            "default_model": self.default_model,
            "cache_size": len(self.cache),
            "stats": self.stats.copy(),
            "timestamp": time.time()
        }

        # 检查默认模型是否可用
        if self.default_model and self.default_model not in self.backends:
            health_status["status"] = "degraded"
            health_status["warning"] = "默认模型未加载"

        return health_status

    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用模型"""
        models = []
        for name, config in self.model_configs.items():
            models.append({
                "name": name,
                "type": config.model_type.value,
                "dimension": config.dimension,
                "max_length": config.max_length,
                "loaded": name in self.backends,
                "is_default": name == self.default_model,
                "priority": config.priority
            })

        return sorted(models, key=lambda x: (x["priority"], x["name"]))

    async def unload_model(self, model_name: str) -> None:
        """卸载指定模型"""
        if model_name in self.backends:
            backend = self.backends[model_name]
            backend.unload_model()
            del self.backends[model_name]
            self.stats["model_unloads"] += 1
            self.logger.info(f"模型 {model_name} 已卸载")

    async def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.logger.info("嵌入向量缓存已清空")

    async def shutdown(self) -> None:
        """关闭服务"""
        self.logger.info("正在关闭统一嵌入服务...")

        # 卸载所有模型
        for model_name in list(self.backends.keys()):
            await self.unload_model(model_name)

        # 清空缓存
        await self.clear_cache()

        self.logger.info("统一嵌入服务关闭完成")