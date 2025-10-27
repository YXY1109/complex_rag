"""
统一检索器服务

基于统一向量存储服务的高效文档检索实现，
支持多种检索策略和混合搜索功能。
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from .unified_vector_store import (
    UnifiedVectorStore, SearchQuery, SearchResult,
    DistanceMetric, IndexType, VectorDBType,
    VectorData
)


class RetrievalStrategy(Enum):
    """检索策略枚举"""
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    RERANKED = "reranked"
    MULTI_VECTOR = "multi_vector"


class RetrievalMode(Enum):
    """检索模式枚举"""
    EXACT = "exact"
    APPROXIMATE = "approximate"
    RECURSIVE = "recursive"
    EXPANSIVE = "expansive"


class FilterOperator(Enum):
    """过滤操作符枚举"""
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    REGEX = "regex"


@dataclass
class RetrievalConfig:
    """检索配置"""
    strategy: RetrievalStrategy = RetrievalStrategy.VECTOR_ONLY
    mode: RetrievalMode = RetrievalMode.APPROXIMATE
    top_k: int = 10
    min_score: float = 0.0
    max_results: int = 100
    rerank_top_k: int = 5
    enable_caching: bool = True
    cache_ttl: int = 3600
    parallel_search: bool = True
    search_timeout: int = 30
    filter_threshold: float = 0.5


@dataclass
class QueryFilter:
    """查询过滤器"""
    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = False


@dataclass
class RetrievalQuery:
    """检索查询"""
    query: str
    query_vector: Optional[List[float]] = None
    collection_name: Optional[str] = None
    filters: List[QueryFilter] = field(default_factory=list)
    config: Optional[RetrievalConfig] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """文档块"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    vector: Optional[List[float]] = None
    collection_name: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    chunks: List[DocumentChunk]
    total_found: int
    search_time: float
    strategy_used: str
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """检索器抽象基类"""

    def __init__(self, config: RetrievalConfig, vector_store: Optional[UnifiedVectorStore] = None):
        self.config = config
        self.vector_store = vector_store
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        # 缓存管理
        self._query_cache = {} if config.enable_caching else None
        self._cache_timestamps = {}

        # 统计信息
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_search_time": 0.0,
            "average_search_time": 0.0,
            "total_chunks_retrieved": 0,
            "last_query_time": None
        }

        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """初始化检索器"""
        pass

    @abstractmethod
    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """执行文档检索"""
        pass

    @abstractmethod
    async def add_documents(self, documents: List[DocumentChunk], collection_name: str) -> List[str]:
        """添加文档到检索器"""
        pass

    @abstractmethod
    async def delete_documents(self, document_ids: List[str], collection_name: str) -> int:
        """删除文档"""
        pass

    def _generate_cache_key(self, query: RetrievalQuery) -> str:
        """生成缓存键"""
        import hashlib
        content = f"{query.query}:{str(query.filters)}:{query.collection_name}:{self.config.top_k}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if not self._query_cache or cache_key not in self._cache_timestamps:
            return False

        timestamp = self._cache_timestamps[cache_key]
        return time.time() - timestamp < self.config.cache_ttl

    def _get_from_cache(self, cache_key: str) -> Optional[RetrievalResult]:
        """从缓存获取结果"""
        if not self._query_cache or not self._is_cache_valid(cache_key):
            return None

        return self._query_cache[cache_key]

    def _store_in_cache(self, cache_key: str, result: RetrievalResult) -> None:
        """存储结果到缓存"""
        if self._query_cache:
            self._query_cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()

    def _update_stats(self, search_time: float, chunk_count: int, cache_hit: bool = False) -> None:
        """更新统计信息"""
        self.stats["total_queries"] += 1
        self.stats["total_search_time"] += search_time
        self.stats["total_chunks_retrieved"] += chunk_count
        self.stats["last_query_time"] = datetime.now()

        if cache_hit:
            self.stats["cache_hits"] += 1
        else:
            self.stats["cache_misses"] += 1

        # 计算平均搜索时间
        if self.stats["total_queries"] > 0:
            self.stats["average_search_time"] = (
                self.stats["total_search_time"] / self.stats["total_queries"]
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        stats = self.stats.copy()
        if stats["total_queries"] > 0:
            stats["cache_hit_rate"] = (
                stats["cache_hits"] / stats["total_queries"] * 100
            )
        else:
            stats["cache_hit_rate"] = 0.0

        return stats


class VectorRetriever(BaseRetriever):
    """向量检索器"""

    def __init__(self, config: RetrievalConfig, vector_store: UnifiedVectorStore):
        """初始化向量检索器"""
        super().__init__(config, vector_store)
        self.embedding_service = None

    async def initialize(self) -> bool:
        """初始化向量检索器"""
        try:
            self.logger.info("初始化向量检索器")

            # 检查向量存储连接
            if not self.vector_store:
                raise ValueError("向量存储服务未提供")

            # 初始化向量化服务（如果需要）
            if not self.embedding_service and self.config.strategy == RetrievalStrategy.VECTOR_ONLY:
                self.logger.warning("未提供向量化服务，将使用外部查询向量")

            self._initialized = True
            self.logger.info("向量检索器初始化成功")
            return True

        except Exception as e:
            self.logger.error(f"向量检索器初始化失败: {str(e)}")
            return False

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """执行向量检索"""
        start_time = time.time()

        try:
            if not self._initialized:
                raise RuntimeError("检索器未初始化")

            # 检查缓存
            cache_key = self._generate_cache_key(query)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                cached_result.cache_hit = True
                self._update_stats(0, len(cached_result.chunks), cache_hit=True)
                return cached_result

            # 执行向量检索
            if not query.query_vector and self.embedding_service:
                # 需要生成查询向量
                query_vector = await self._generate_query_vector(query.query)
            else:
                query_vector = query.query_vector

            if not query_vector:
                raise ValueError("查询向量为空")

            # 构建搜索查询
            search_query = SearchQuery(
                vector=query_vector,
                top_k=self.config.top_k,
                filters=self._build_search_filters(query.filters),
                threshold=self.config.min_score
            )

            # 执行搜索
            collection_name = query.collection_name or "default"
            search_results = await self.vector_store.search_vectors(
                search_query=search_query,
                collection_name=collection_name
            )

            # 转换为文档块
            chunks = []
            for result in search_results:
                chunk = DocumentChunk(
                    id=result.id,
                    content=result.metadata.get("content", ""),
                    metadata=result.metadata,
                    score=result.score,
                    vector=query_vector,
                    collection_name=collection_name
                )
                chunks.append(chunk)

            search_time = time.time() - start_time

            # 创建检索结果
            retrieval_result = RetrievalResult(
                query=query.query,
                chunks=chunks,
                total_found=len(chunks),
                search_time=search_time,
                strategy_used="vector_search",
                cache_hit=False,
                metadata={
                    "collection_name": collection_name,
                    "vector_dimension": len(query_vector),
                    "search_params": search_query.parameters
                },
                stats={
                    "backend": self.vector_store.get_backend("default").backend_type.value,
                    "index_type": self.vector_store.get_backend("default").capabilities.get("supported_index_types", ["hnsw"])[0]
                }
            )

            # 缓存结果
            self._store_in_cache(cache_key, retrieval_result)
            self._update_stats(search_time, len(chunks), cache_hit=False)

            return retrieval_result

        except Exception as e:
            self.logger.error(f"向量检索失败: {str(e)}")
            raise

    async def add_documents(self, documents: List[DocumentChunk], collection_name: str) -> List[str]:
        """添加文档到向量检索器"""
        if not self._initialized:
            raise RuntimeError("检索器未初始化")

        try:
            # 转换为向量数据格式
            vector_data_list = []
            for doc in documents:
                vector_data = VectorData(
                    id=doc.id,
                    vector=doc.vector or [],
                    metadata={
                        "content": doc.content,
                        **doc.metadata,
                        "created_at": doc.created_at.isoformat() if doc.created_at else None,
                        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None
                    },
                    collection_name=collection_name
                )
                vector_data_list.append(vector_data)

            # 批量插入
            document_ids = await self.vector_store.upsert_vectors(
                vectors=vector_data_list,
                collection_name=collection_name
            )

            self.logger.info(f"添加了 {len(documents)} 个文档到向量存储")
            return document_ids

        except Exception as e:
            self.logger.error(f"添加文档失败: {str(e)}")
            return []

    async def delete_documents(self, document_ids: List[str], collection_name: str) -> int:
        """删除文档"""
        if not self._initialized:
            raise RuntimeError("检索器未初始化")

        try:
            deleted_count = await self.vector_store.delete_vectors(
                vector_ids=document_ids,
                collection_name=collection_name
            )

            self.logger.info(f"删除了 {deleted_count} 个文档")
            return deleted_count

        except Exception as e:
            self.logger.error(f"删除文档失败: {str(e)}")
            return 0

    async def _generate_query_vector(self, query_text: str) -> List[float]:
        """生成查询向量"""
        if not self.embedding_service:
            raise ValueError("未配置向量化服务")

        # 这里需要调用统一嵌入服务
        from rag_service.services.unified_embedding_service import EmbeddingRequest
        from config.unified_embedding_config import get_unified_embedding_config

        # 创建临时嵌入服务实例
        embedding_service = UnifiedEmbeddingService(get_unified_embedding_config())
        await embedding_service.initialize()

        # 生成查询向量
        request = EmbeddingRequest(
            texts=[query_text],
            use_cache=True
        )

        response = await embedding_service.embed(request)
        return response.embeddings[0] if response.embeddings else []

    def _build_search_filters(self, filters: List[QueryFilter]) -> Optional[Dict[str, Any]]:
        """构建搜索过滤器"""
        if not filters:
            return None

        filter_dict = {}
        for filter_item in filters:
            key = f"metadata.{filter_item.field}"

            if filter_item.operator == FilterOperator.EQ:
                filter_dict[key] = filter_item.value
            elif filter_item.operator == FilterOperator.IN:
                filter_dict[key] = {"$in": filter_item.value}
            elif filter_item.operator == FilterOperator.CONTAINS:
                filter_dict[key] = {"$regex": filter_item.value}
            # 可以添加更多操作符支持

        return filter_dict if filter_dict else None


class HybridRetriever(BaseRetriever):
    """混合检索器"""

    def __init__(self, config: RetrievalConfig, vector_retriever: VectorRetriever):
        """初始化混合检索器"""
        super().__init__(config)
        self.vector_retriever = vector_retriever

    async def initialize(self) -> bool:
        """初始化混合检索器"""
        try:
            # 初始化向量检索器
            success = await self.vector_retriever.initialize()
            if not success:
                return False

            self._initialized = True
            self.logger.info("混合检索器初始化成功")
            return True

        except Exception as e:
            self.logger.error(f"混合检索器初始化失败: {str(e)}")
            return False

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """执行混合检索"""
        start_time = time.time()

        try:
            # 检查缓存
            cache_key = self._generate_cache_key(query)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                cached_result.cache_hit = True
                self._update_stats(0, len(cached_result.chunks), cache_hit=True)
                return cached_result

            # 执行向量检索
            vector_result = await self.vector_retriever.retrieve(query)

            # 这里可以添加关键词搜索、重排序等逻辑
            # 目前暂时返回向量检索结果
            hybrid_result = RetrievalResult(
                query=query.query,
                chunks=vector_result.chunks,
                total_found=vector_result.total_found,
                search_time=time.time() - start_time,
                strategy_used="hybrid_search",
                cache_hit=False,
                metadata={
                    "vector_search_time": vector_result.search_time,
                    "components": ["vector"]
                },
                stats=vector_result.stats
            )

            # 缓存结果
            self._store_in_cache(cache_key, hybrid_result)
            self._update_stats(hybrid_result.search_time, len(hybrid_result.chunks))

            return hybrid_result

        except Exception as e:
            self.logger.error(f"混合检索失败: {str(e)}")
            raise

    async def add_documents(self, documents: List[DocumentChunk], collection_name: str) -> List[str]:
        """添加文档"""
        return await self.vector_retriever.add_documents(documents, collection_name)

    async def delete_documents(self, document_ids: List[str], collection_name: str) -> int:
        """删除文档"""
        return await self.vector_retriever.delete_documents(document_ids, collection_name)


class RetrievalFactory:
    """检索器工厂"""

    @staticmethod
    def create_retriever(
        strategy: RetrievalStrategy,
        config: RetrievalConfig,
        vector_store: UnifiedVectorStore,
        **kwargs
    ) -> BaseRetriever:
        """创建检索器实例"""
        if strategy == RetrievalStrategy.VECTOR_ONLY:
            return VectorRetriever(config, vector_store)
        elif strategy == RetrievalStrategy.HYBRID:
            vector_retriever = VectorRetriever(config, vector_store)
            return HybridRetriever(config, vector_retriever)
        else:
            raise ValueError(f"不支持的检索策略: {strategy}")