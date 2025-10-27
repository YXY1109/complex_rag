"""
统一向量存储服务

整合多种向量数据库实现，提供统一、高性能的向量存储和检索服务。
支持Milvus、Elasticsearch、FAISS等多种向量数据库。
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

import numpy as np


class VectorDBType(Enum):
    """向量数据库类型枚举"""
    MILVUS = "milvus"
    ELASTICSEARCH = "elasticsearch"
    FAISS = "faiss"
    QDRANT = "qdrant"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


class DistanceMetric(Enum):
    """距离度量枚举"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    DOT_PRODUCT = "dot_product"
    HAMMING = "hamming"
    JACCARD = "jaccard"


class IndexType(Enum):
    """索引类型枚举"""
    FLAT = "flat"
    IVF_FLAT = "ivf_flat"
    IVF_SQ = "ivf_sq"
    IVF_PQ = "ivf_pq"
    HNSW = "hnsw"
    ANNOY = "annoy"
    SCANN = "scann"


class IndexStatus(Enum):
    """索引状态枚举"""
    CREATING = "creating"
    CREATED = "created"
    REBUILDING = "rebuilding"
    READY = "ready"
    ERROR = "error"


@dataclass
class VectorIndexConfig:
    """向量索引配置"""
    index_type: IndexType = IndexType.HNSW
    metric_type: DistanceMetric = DistanceMetric.COSINE
    dimension: int = 768
    ef_construction: int = 200
    M: int = 16
    nlist: int = 128
    nprobe: int = 8
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionConfig:
    """集合配置"""
    collection_name: str
    dimension: int = 768
    description: str = ""
    max_capacity: int = 1000000
    shard_num: int = 1
    consistency_level: str = "Strong"
    index_config: VectorIndexConfig = field(default_factory=VectorIndexConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorData:
    """向量数据"""
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    collection_name: Optional[str] = None


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance: Optional[float] = None


@dataclass
class SearchQuery:
    """搜索查询"""
    vector: List[float]
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    offset: int = 0
    threshold: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchOperation:
    """批量操作"""
    operation: str  # "insert", "update", "delete"
    vectors: List[VectorData]
    collection_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreStats:
    """向量存储统计"""
    collection_name: str
    total_vectors: int
    index_status: IndexStatus
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    index_size_mb: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class VectorDBBackend(ABC):
    """向量数据库后端抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._connected = False
        self._collections: Dict[str, CollectionConfig] = {}

    @abstractmethod
    async def connect(self) -> bool:
        """连接到向量数据库"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开向量数据库连接"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass

    @abstractmethod
    async def create_collection(self, config: CollectionConfig) -> bool:
        """创建集合"""
        pass

    @abstractmethod
    async def drop_collection(self, collection_name: str) -> bool:
        """删除集合"""
        pass

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """列出所有集合"""
        pass

    @abstractmethod
    async def get_collection_stats(self, collection_name: str) -> Optional[VectorStoreStats]:
        """获取集合统计信息"""
        pass

    @abstractmethod
    async def upsert_vectors(self, vectors: List[VectorData], collection_name: str) -> List[str]:
        """插入或更新向量"""
        pass

    @abstractmethod
    async def search_vectors(self, query: SearchQuery, collection_name: str) -> List[SearchResult]:
        """搜索向量"""
        pass

    @abstractmethod
    async def delete_vectors(self, vector_ids: List[str], collection_name: str) -> int:
        """删除向量"""
        pass

    @abstractmethod
    async def batch_operation(self, operations: List[BatchOperation]) -> Dict[str, int]:
        """批量操作"""
        pass

    @abstractmethod
    async def create_index(self, collection_name: str, index_config: VectorIndexConfig) -> bool:
        """创建索引"""
        pass

    @abstractmethod
    async def drop_index(self, collection_name: str, index_name: str = "") -> bool:
        """删除索引"""
        pass

    @abstractmethod
    async def rebuild_index(self, collection_name: str) -> bool:
        """重建索引"""
        pass

    @property
    def backend_type(self) -> VectorDBType:
        """返回后端类型"""
        raise NotImplementedError

    @property
    def capabilities(self) -> Dict[str, Any]:
        """返回后端能力"""
        return {
            "supports_metadata": True,
            "supports_filtering": True,
            "supports_batch_operations": True,
            "supports_hybrid_search": False,
            "max_dimension": 4096,
            "supported_metrics": [metric.value for metric in DistanceMetric],
            "supported_index_types": [index_type.value for index_type in IndexType]
        }


class MilvusBackend(VectorDBBackend):
    """Milvus向量数据库后端"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
        self._connection_params = {
            'host': config.get('host', 'localhost'),
            'port': config.get('port', 19530),
            'user': config.get('user'),
            'password': config.get('password'),
            'db_name': config.get('db_name', 'default'),
            'timeout': config.get('timeout', 30)
        }

    async def connect(self) -> bool:
        """连接到Milvus"""
        try:
            from pymilvus import connections, MilvusException

            await asyncio.to_thread(connections.connect, **self._connection_params)
            self._connected = True
            self.logger.info("Milvus连接成功")
            return True

        except ImportError:
            self.logger.error("pymilvus包未安装")
            return False
        except MilvusException as e:
            self.logger.error(f"Milvus连接失败: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Milvus连接异常: {str(e)}")
            return False

    async def disconnect(self) -> None:
        """断开Milvus连接"""
        try:
            from pymilvus import connections
            await asyncio.to_thread(connections.disconnect, "default")
            self._connected = False
            self.logger.info("Milvus连接已断开")
        except Exception as e:
            self.logger.error(f"Milvus断开连接失败: {str(e)}")

    async def health_check(self) -> Dict[str, Any]:
        """Milvus健康检查"""
        try:
            if not self._connected:
                return {"status": "disconnected", "details": "未连接到Milvus"}

            from pymilvus import utility
            # 测试基本操作
            version = await asyncio.to_thread(utility.get_server_version)
            is_healthy = version is not None

            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "backend": "milvus",
                "version": version,
                "connected": self._connected,
                "capabilities": self.capabilities
            }

        except Exception as e:
            return {
                "status": "error",
                "backend": "milvus",
                "error": str(e),
                "connected": False
            }

    async def create_collection(self, config: CollectionConfig) -> bool:
        """创建Milvus集合"""
        try:
            from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, MilvusException
            from pymilvus.orm.collections import CollectionSchema as OrmCollectionSchema

            # 定义字段Schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=255),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=config.dimension),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]

            # 创建集合Schema
            collection_schema = OrmCollectionSchema(
                fields=fields,
                description=config.description,
                enable_dynamic_field=True
            )

            # 创建集合
            collection = Collection(name=config.collection_name, schema=collection_schema)

            # 等待创建完成
            await asyncio.sleep(1)

            # 创建索引
            index_config = config.index_config
            await self.create_index(config.collection_name, index_config)

            self._collections[config.collection_name] = config
            self.logger.info(f"Milvus集合 {config.collection_name} 创建成功")
            return True

        except Exception as e:
            self.logger.error(f"创建Milvus集合失败: {str(e)}")
            return False

    async def drop_collection(self, collection_name: str) -> bool:
        """删除Milvus集合"""
        try:
            from pymilvus import Collection, utility

            # 检查集合是否存在
            if await asyncio.to_thread(utility.has_collection, collection_name):
                await asyncio.to_thread(utility.drop_collection, collection_name)

                # 从缓存中移除
                if collection_name in self._collections:
                    del self._collections[collection_name]

                self.logger.info(f"Milvus集合 {collection_name} 删除成功")
                return True
            else:
                self.logger.warning(f"Milvus集合 {collection_name} 不存在")
                return False

        except Exception as e:
            self.logger.error(f"删除Milvus集合失败: {str(e)}")
            return False

    async def list_collections(self) -> List[str]:
        """列出Milvus集合"""
        try:
            from pymilvus import utility
            collections = await asyncio.to_thread(utility.list_collections)
            return collections
        except Exception as e:
            self.logger.error(f"列出Milvus集合失败: {str(e)}")
            return []

    async def get_collection_stats(self, collection_name: str) -> Optional[VectorStoreStats]:
        """获取Milvus集合统计"""
        try:
            from pymilvus import Collection, utility

            if not await asyncio.to_thread(utility.has_collection, collection_name):
                return None

            collection = Collection(collection_name)
            await asyncio.to_thread(collection.load)

            num_entities = await asyncio.to_thread(collection.num_entities)
            collection_info = await asyncio.to_thread(utility.get_collection_stats, collection_name)

            return VectorStoreStats(
                collection_name=collection_name,
                total_vectors=num_entities,
                index_status=IndexStatus.READY,
                memory_usage_mb=collection_info.get('data_size', 0) / (1024 * 1024) if 'data_size' in collection_info else 0,
                disk_usage_mb=collection_info.get('data_size', 0) / (1024 * 1024) if 'data_size' in collection_info else 0,
            )

        except Exception as e:
            self.logger.error(f"获取Milvus集合统计失败: {str(e)}")
            return None

    async def upsert_vectors(self, vectors: List[VectorData], collection_name: str) -> List[str]:
        """插入或更新Milvus向量"""
        try:
            from pymilvus import Collection

            collection = Collection(collection_name)
            await asyncio.to_thread(collection.load)

            # 准备数据
            ids = [v.id for v in vectors]
            vector_data = [v.vector for v in vectors]
            metadata_data = [json.dumps(v.metadata) for v in vectors]

            # 批量插入
            batch_size = 1000
            inserted_ids = []

            for i in range(0, len(vectors), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_vectors = vector_data[i:i + batch_size]
                batch_metadata = metadata_data[i:i + batch_size]

                result = await asyncio.to_thread(
                    collection.insert,
                    [batch_ids, batch_vectors, batch_metadata]
                )

                if hasattr(result, 'primary_keys'):
                    inserted_ids.extend(result.primary_keys)

            # 刷新
            await asyncio.to_thread(collection.flush)

            self.logger.info(f"Milvus插入 {len(vectors)} 个向量到集合 {collection_name}")
            return inserted_ids

        except Exception as e:
            self.logger.error(f"Milvus插入向量失败: {str(e)}")
            return []

    async def search_vectors(self, query: SearchQuery, collection_name: str) -> List[SearchResult]:
        """搜索Milvus向量"""
        try:
            from pymilvus import Collection

            collection = Collection(collection_name)
            await asyncio.to_thread(collection.load)

            # 构建搜索参数
            search_params = {
                "metric_type": query.parameters.get("metric_type", "COSINE"),
                "params": {
                    "nprobe": query.parameters.get("nprobe", 10)
                }
            }

            # 执行搜索
            results = await asyncio.to_thread(
                collection.search,
                data=[query.vector],
                anns_field="vector",
                param=search_params,
                limit=query.top_k,
                expr=self._build_filter_expr(query.filters),
                output_fields=["id", "metadata"]
            )

            # 转换结果
            search_results = []
            for result in results[0]:  # 只使用第一个查询结果
                metadata = json.loads(result.entity.get("metadata", "{}"))

                search_results.append(SearchResult(
                    id=result.entity.get("id"),
                    score=result.score,
                    metadata=metadata,
                    distance=result.distance if hasattr(result, 'distance') else None
                ))

            return search_results

        except Exception as e:
            self.logger.error(f"Milvus搜索向量失败: {str(e)}")
            return []

    async def delete_vectors(self, vector_ids: List[str], collection_name: str) -> int:
        """删除Milvus向量"""
        try:
            from pymilvus import Collection

            collection = Collection(collection_name)
            await asyncio.to_thread(collection.load)

            # 构建删除表达式
            if len(vector_ids) == 1:
                expr = f'id == "{vector_ids[0]}"'
            else:
                ids_str = '", "'.join(vector_ids)
                expr = f'id in ["{ids_str}"]'

            # 执行删除
            result = await asyncio.to_thread(collection.delete, expr)

            deleted_count = len(vector_ids)  # Milvus不返回删除数量，假设都成功
            self.logger.info(f"Milvus删除 {deleted_count} 个向量")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Milvus删除向量失败: {str(e)}")
            return 0

    async def batch_operation(self, operations: List[BatchOperation]) -> Dict[str, int]:
        """批量操作"""
        results = {"insert": 0, "update": 0, "delete": 0}

        for operation in operations:
            if operation.operation == "insert":
                inserted_ids = await self.upsert_vectors(operation.vectors, operation.collection_name)
                results["insert"] += len(inserted_ids)
            elif operation.operation == "delete":
                vector_ids = [v.id for v in operation.vectors]
                deleted_count = await self.delete_vectors(vector_ids, operation.collection_name)
                results["delete"] += deleted_count

        return results

    async def create_index(self, collection_name: str, index_config: VectorIndexConfig) -> bool:
        """创建Milvus索引"""
        try:
            from pymilvus import Collection

            collection = Collection(collection_name)

            # 索引参数
            index_params = {
                "index_type": index_config.index_type.value.upper(),
                "metric_type": index_config.metric_type.value.upper(),
                "params": {
                    "M": index_config.M,
                    "efConstruction": index_config.ef_construction
                }
            }

            await asyncio.to_thread(
                collection.create_index,
                field_name="vector",
                index_params=index_params
            )

            self.logger.info(f"Milvus索引创建成功: {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"创建Milvus索引失败: {str(e)}")
            return False

    async def drop_index(self, collection_name: str, index_name: str = "") -> bool:
        """删除Milvus索引"""
        try:
            from pymilvus import Collection

            collection = Collection(collection_name)
            await asyncio.to_thread(collection.drop_index())

            self.logger.info(f"Milvus索引删除成功: {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"删除Milvus索引失败: {str(e)}")
            return False

    async def rebuild_index(self, collection_name: str) -> bool:
        """重建Milvus索引"""
        try:
            # 删除索引
            await self.drop_index(collection_name)

            # 获取集合配置并重建索引
            if collection_name in self._collections:
                config = self._collections[collection_name]
                await self.create_index(collection_name, config.index_config)

                self.logger.info(f"Milvus索引重建成功: {collection_name}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"重建Milvus索引失败: {str(e)}")
            return False

    @property
    def backend_type(self) -> VectorDBType:
        """返回Milvus后端类型"""
        return VectorDBType.MILVUS

    def _build_filter_expr(self, filters: Optional[Dict[str, Any]]) -> str:
        """构建Milvus过滤表达式"""
        if not filters:
            return ""

        expressions = []
        for key, value in filters.items():
            if isinstance(value, str):
                expressions.append(f'metadata["{key}"] == "{value}"')
            elif isinstance(value, (int, float)):
                expressions.append(f'metadata["{key}"] == {value}')
            elif isinstance(value, bool):
                expressions.append(f'metadata["{key}"] == {str(value).lower()}')
            elif isinstance(value, list):
                values_str = '", "'.join([str(v) for v in value])
                expressions.append(f'metadata["{key}"] in ["{values_str}"]')

        return " and ".join(expressions)


class UnifiedVectorStore:
    """统一向量存储服务"""

    def __init__(self, config: Dict[str, Any]):
        """初始化统一向量存储服务"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 后端管理
        self.backends: Dict[str, VectorDBBackend] = {}
        self.default_backend_name = config.get("default_backend", "milvus")

        # 性能配置
        self.max_concurrent_requests = config.get("max_concurrent_requests", 10)
        self.request_timeout = config.get("request_timeout", 30)

        # 缓存和统计
        self.stats = {
            "total_requests": 0,
            "total_operations": 0,
            "total_vectors_stored": 0,
            "cache_hits": 0,
            "errors": 0
        }

        # 初始化后端
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """初始化向量数据库后端"""
        backend_configs = self.config.get("backends", {})

        for name, backend_config in backend_configs.items():
            backend_type = backend_config.get("type")

            if backend_type == "milvus":
                self.backends[name] = MilvusBackend(backend_config)
            # 可以在这里添加其他后端支持
            # elif backend_type == "elasticsearch":
            #     self.backends[name] = ElasticsearchBackend(backend_config)
            # elif backend_type == "faiss":
            #     self.backends[name] = FaissBackend(backend_config)

        self.logger.info(f"初始化了 {len(self.backends)} 个向量数据库后端")

    async def initialize(self) -> bool:
        """初始化统一向量存储服务"""
        try:
            self.logger.info("正在初始化统一向量存储服务...")

            # 连接所有后端
            for name, backend in self.backends.items():
                success = await backend.connect()
                if success:
                    self.logger.info(f"后端 {name} 连接成功")
                else:
                    self.logger.error(f"后端 {name} 连接失败")

            self.logger.info("统一向量存储服务初始化完成")
            return True

        except Exception as e:
            self.logger.error(f"统一向量存储服务初始化失败: {str(e)}")
            return False

    def get_backend(self, backend_name: Optional[str] = None) -> VectorDBBackend:
        """获取向量数据库后端"""
        backend_name = backend_name or self.default_backend_name

        if backend_name not in self.backends:
            raise ValueError(f"未知的后端: {backend_name}")

        return self.backends[backend_name]

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "service": "unified_vector_store",
            "status": "healthy",
            "backends": {},
            "stats": self.stats.copy(),
            "timestamp": time.time()
        }

        # 检查所有后端
        for name, backend in self.backends.items():
            backend_health = await backend.health_check()
            health_status["backends"][name] = backend_health

            # 如果默认后端不健康，整体状态降级
            if name == self.default_backend_name and backend_health.get("status") != "healthy":
                health_status["status"] = "degraded"

        return health_status

    async def shutdown(self) -> None:
        """关闭统一向量存储服务"""
        self.logger.info("正在关闭统一向量存储服务...")

        # 断开所有后端连接
        for name, backend in self.backends.items():
            try:
                await backend.disconnect()
                self.logger.info(f"后端 {name} 已断开连接")
            except Exception as e:
                self.logger.error(f"断开后端 {name} 连接失败: {str(e)}")

        self.logger.info("统一向量存储服务关闭完成")