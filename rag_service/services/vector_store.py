"""
向量存储服务

基于RAGFlow架构实现的高性能向量存储和检索服务，
支持多种向量数据库和高效的相似度搜索。
"""

import asyncio
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import uuid
from dataclasses import dataclass, field

from ..interfaces.rag_interface import (
    VectorStoreInterface, DocumentChunk, VectorStoreException
)
from ...infrastructure.database.implementations.vector.milvus_client import MilvusClient
from ...infrastructure.database.implementations.search.elasticsearch_client import ElasticsearchClient


@dataclass
class VectorIndexConfig:
    """向量索引配置。"""

    index_type: str = "HNSW"          # 索引类型：HNSW, IVF_FLAT, IVF_PQ, FLAT
    metric_type: str = "COSINE"       # 距离类型：COSINE, L2, IP
    ef_construction: int = 200        # HNSW构建参数
    M: int = 16                       # HNSW连接数
    nlist: int = 128                  # IVF聚类数
    nprobe: int = 8                   # IVF搜索探测数


@dataclass
class CollectionConfig:
    """集合配置。"""

    collection_name: str
    dimension: int
    description: str = ""
    max_capacity: int = 1000000
    shard_num: int = 1
    consistency_level: str = "Strong"
    index_config: VectorIndexConfig = field(default_factory=VectorIndexConfig)


class VectorStore(VectorStoreInterface):
    """向量存储服务。"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化向量存储。

        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 向量数据库客户端
        self.milvus_client: Optional[MilvusClient] = None
        self.elasticsearch_client: Optional[ElasticsearchClient] = None

        # 集合配置缓存
        self.collection_configs: Dict[str, CollectionConfig] = {}

        # 连接池和缓存
        self.connection_pool_size = config.get("connection_pool_size", 10)
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_ttl = config.get("cache_ttl", 3600)

        # 性能配置
        self.batch_size = config.get("batch_size", 1000)
        self.parallel_search = config.get("parallel_search", True)
        self.search_timeout = config.get("search_timeout", 30)

    async def initialize(self) -> bool:
        """初始化向量存储。"""
        try:
            # 初始化Milvus客户端
            milvus_config = self.config.get("milvus", {})
            if milvus_config:
                self.milvus_client = MilvusClient(milvus_config)
                if await self.milvus_client.connect():
                    self.logger.info("Milvus客户端初始化成功")
                else:
                    self.logger.error("Milvus客户端初始化失败")
                    return False

            # 初始化Elasticsearch客户端（用于混合检索）
            es_config = self.config.get("elasticsearch", {})
            if es_config:
                self.elasticsearch_client = ElasticsearchClient(es_config)
                if await self.elasticsearch_client.connect():
                    self.logger.info("Elasticsearch客户端初始化成功")
                else:
                    self.logger.warning("Elasticsearch客户端初始化失败，将只使用向量检索")

            return True

        except Exception as e:
            self.logger.error(f"向量存储初始化失败: {e}")
            return False

    async def cleanup(self) -> None:
        """清理向量存储资源。"""
        try:
            if self.milvus_client:
                await self.milvus_client.disconnect()
                self.milvus_client = None

            if self.elasticsearch_client:
                await self.elasticsearch_client.disconnect()
                self.elasticsearch_client = None

            self.logger.info("向量存储资源清理完成")

        except Exception as e:
            self.logger.error(f"向量存储清理失败: {e}")

    async def create_collection(self, config: CollectionConfig) -> bool:
        """
        创建向量集合。

        Args:
            config: 集合配置

        Returns:
            bool: 创建是否成功
        """
        try:
            if not self.milvus_client:
                raise VectorStoreException("Milvus客户端未初始化")

            # 检查集合是否已存在
            if await self.milvus_client.has_collection(config.collection_name):
                self.logger.info(f"集合 {config.collection_name} 已存在")
                return True

            # 定义字段schema
            fields = [
                {
                    "field_name": "id",
                    "data_type": "VARCHAR",
                    "is_primary": True,
                    "max_length": 255
                },
                {
                    "field_name": "vector",
                    "data_type": "FLOAT_VECTOR",
                    "dim": config.dimension
                },
                {
                    "field_name": "content",
                    "data_type": "VARCHAR",
                    "max_length": 65535
                },
                {
                    "field_name": "metadata",
                    "data_type": "JSON"
                },
                {
                    "field_name": "document_id",
                    "data_type": "VARCHAR",
                    "max_length": 255
                },
                {
                    "field_name": "chunk_index",
                    "data_type": "INT64"
                },
                {
                    "field_name": "created_at",
                    "data_type": "INT64"
                }
            ]

            # 创建集合
            await self.milvus_client.create_collection(
                collection_name=config.collection_name,
                fields=fields,
                shard_num=config.shard_num,
                consistency_level=config.consistency_level
            )

            # 创建索引
            index_params = {
                "index_type": config.index_config.index_type,
                "metric_type": config.index_config.metric_type,
                "params": {
                    "M": config.index_config.M,
                    "efConstruction": config.index_config.ef_construction
                } if config.index_config.index_type == "HNSW" else {
                    "nlist": config.index_config.nlist
                }
            }

            await self.milvus_client.create_index(
                collection_name=config.collection_name,
                field_name="vector",
                index_params=index_params
            )

            # 缓存配置
            self.collection_configs[config.collection_name] = config

            self.logger.info(f"向量集合 {config.collection_name} 创建成功")
            return True

        except Exception as e:
            self.logger.error(f"创建向量集合失败: {e}")
            raise VectorStoreException(f"创建集合失败: {str(e)}")

    async def add_vectors(
        self,
        vectors: List[List[float]],
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        添加向量。

        Args:
            vectors: 向量列表
            documents: 文档列表

        Returns:
            List[str]: 文档ID列表
        """
        try:
            if not self.milvus_client:
                raise VectorStoreException("Milvus客户端未初始化")

            if len(vectors) != len(documents):
                raise VectorStoreException("向量和文档数量不匹配")

            # 生成文档ID
            document_ids = [str(uuid.uuid4()) for _ in documents]

            # 准备插入数据
            current_time = int(datetime.now().timestamp() * 1000)

            insert_data = {
                "id": document_ids,
                "vector": vectors,
                "content": [doc.get("content", "") for doc in documents],
                "metadata": [json.dumps(doc.get("metadata", {})) for doc in documents],
                "document_id": [doc.get("document_id", "") for doc in documents],
                "chunk_index": [doc.get("chunk_index", 0) for doc in documents],
                "created_at": [current_time] * len(documents)
            }

            # 分批插入
            collection_name = documents[0].get("collection_name", "default")
            if collection_name not in self.collection_configs:
                # 使用默认配置创建集合
                default_config = CollectionConfig(
                    collection_name=collection_name,
                    dimension=len(vectors[0])
                )
                await self.create_collection(default_config)

            await self.milvus_client.insert(collection_name, insert_data)

            # 刷新集合使数据可见
            await self.milvus_client.flush([collection_name])

            self.logger.info(f"成功添加 {len(vectors)} 个向量到集合 {collection_name}")

            # 同时添加到Elasticsearch（如果可用）
            if self.elasticsearch_client:
                await self._add_to_elasticsearch(document_ids, documents, vectors)

            return document_ids

        except Exception as e:
            self.logger.error(f"添加向量失败: {e}")
            raise VectorStoreException(f"添加向量失败: {str(e)}")

    async def _add_to_elasticsearch(
        self,
        document_ids: List[str],
        documents: List[Dict[str, Any]],
        vectors: List[List[float]]
    ) -> None:
        """添加文档到Elasticsearch。"""
        try:
            es_docs = []
            for doc_id, doc, vector in zip(document_ids, documents, vectors):
                es_doc = {
                    "_id": doc_id,
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "document_id": doc.get("document_id", ""),
                    "chunk_index": doc.get("chunk_index", 0),
                    "created_at": datetime.now().isoformat(),
                    "vector": vector  # 存储向量用于某些ES的向量搜索功能
                }
                es_docs.append(es_doc)

            # 批量索引
            await self.elasticsearch_client.bulk_index("vectors", es_docs)

        except Exception as e:
            self.logger.warning(f"添加文档到Elasticsearch失败: {e}")

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: str = "default",
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        搜索相似向量。

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            filters: 过滤条件
            collection_name: 集合名称
            search_params: 搜索参数

        Returns:
            List[Tuple[str, float, Dict[str, Any]]]: (文档ID, 相似度分数, 元数据) 列表
        """
        try:
            if not self.milvus_client:
                raise VectorStoreException("Milvus客户端未初始化")

            # 构建搜索参数
            config = self.collection_configs.get(collection_name)
            if not config:
                raise VectorStoreException(f"集合 {collection_name} 不存在")

            default_search_params = {
                "metric_type": config.index_config.metric_type,
                "params": {
                    "ef": 64,
                    "nprobe": config.index_config.nprobe
                }
            }

            search_params = search_params or default_search_params

            # 构建输出字段
            output_fields = ["content", "metadata", "document_id", "chunk_index", "created_at"]

            # 执行向量搜索
            results = await self.milvus_client.search(
                collection_name=collection_name,
                vectors=[query_vector],
                limit=top_k,
                output_fields=output_fields,
                search_params=search_params,
                expr=self._build_filter_expression(filters) if filters else None
            )

            # 格式化结果
            formatted_results = []
            if results and len(results) > 0 and len(results[0]) > 0:
                for hit in results[0]:
                    doc_id = hit.get("id")
                    score = hit.get("distance", 0.0)

                    # 如果是COSINE距离，转换为相似度分数
                    if config.index_config.metric_type == "COSINE":
                        score = 1 - score

                    metadata = {
                        "content": hit.get("content", ""),
                        "document_id": hit.get("document_id", ""),
                        "chunk_index": hit.get("chunk_index", 0),
                        "created_at": hit.get("created_at"),
                        "raw_metadata": hit.get("metadata", "{}")
                    }

                    # 解析JSON元数据
                    try:
                        if metadata["raw_metadata"]:
                            metadata.update(json.loads(metadata["raw_metadata"]))
                    except (json.JSONDecodeError, TypeError):
                        pass

                    formatted_results.append((doc_id, score, metadata))

            self.logger.info(f"向量搜索完成，返回 {len(formatted_results)} 个结果")

            return formatted_results

        except Exception as e:
            self.logger.error(f"向量搜索失败: {e}")
            raise VectorStoreException(f"向量搜索失败: {str(e)}")

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: str = "default",
        text_weight: float = 0.5,
        vector_weight: float = 0.5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        混合搜索（文本+向量）。

        Args:
            query_text: 查询文本
            query_vector: 查询向量
            top_k: 返回结果数量
            filters: 过滤条件
            collection_name: 集合名称
            text_weight: 文本搜索权重
            vector_weight: 向量搜索权重

        Returns:
            List[Tuple[str, float, Dict[str, Any]]]: (文档ID, 融合分数, 元数据) 列表
        """
        try:
            # 并行执行文本搜索和向量搜索
            tasks = []

            # 向量搜索
            vector_task = asyncio.create_task(
                self.search(query_vector, top_k * 2, filters, collection_name)
            )
            tasks.append(vector_task)

            # 文本搜索（如果Elasticsearch可用）
            if self.elasticsearch_client:
                text_task = asyncio.create_task(
                    self._text_search(query_text, top_k * 2, filters)
                )
                tasks.append(text_task)

            # 等待搜索结果
            search_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理搜索结果
            vector_results = []
            text_results = []

            if isinstance(search_results[0], list):
                vector_results = search_results[0]

            if len(search_results) > 1 and isinstance(search_results[1], list):
                text_results = search_results[1]

            # 融合搜索结果
            fused_results = self._fuse_search_results(
                vector_results,
                text_results,
                vector_weight,
                text_weight
            )

            # 返回top_k结果
            return fused_results[:top_k]

        except Exception as e:
            self.logger.error(f"混合搜索失败: {e}")
            # 降级到向量搜索
            return await self.search(query_vector, top_k, filters, collection_name)

    async def _text_search(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """文本搜索。"""
        try:
            if not self.elasticsearch_client:
                return []

            # 构建搜索查询
            search_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": ["content", "metadata.title"],
                                    "type": "best_fields"
                                }
                            }
                        ]
                    }
                },
                "size": top_k,
                "_source": ["content", "metadata", "document_id", "chunk_index", "created_at"]
            }

            # 添加过滤条件
            if filters:
                filter_clauses = []
                for field, value in filters.items():
                    if isinstance(value, str):
                        filter_clauses.append({"term": {f"metadata.{field}": value}})
                    elif isinstance(value, list):
                        filter_clauses.append({"terms": {f"metadata.{field}": value}})

                if filter_clauses:
                    search_query["query"]["bool"]["filter"] = filter_clauses

            # 执行搜索
            es_results = await self.elasticsearch_client.search("vectors", search_query)

            # 格式化结果
            formatted_results = []
            for hit in es_results.get("hits", {}).get("hits", []):
                doc_id = hit["_id"]
                score = hit["_score"] / 10.0  # 标准化分数
                source = hit["_source"]

                metadata = {
                    "content": source.get("content", ""),
                    "document_id": source.get("document_id", ""),
                    "chunk_index": source.get("chunk_index", 0),
                    "created_at": source.get("created_at"),
                    **source.get("metadata", {})
                }

                formatted_results.append((doc_id, score, metadata))

            return formatted_results

        except Exception as e:
            self.logger.warning(f"文本搜索失败: {e}")
            return []

    def _fuse_search_results(
        self,
        vector_results: List[Tuple[str, float, Dict[str, Any]]],
        text_results: List[Tuple[str, float, Dict[str, Any]]],
        vector_weight: float,
        text_weight: float
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """融合搜索结果。"""
        # 创建文档ID到结果的映射
        result_map = {}

        # 处理向量搜索结果
        for doc_id, score, metadata in vector_results:
            result_map[doc_id] = {
                "vector_score": score,
                "text_score": 0.0,
                "metadata": metadata
            }

        # 处理文本搜索结果
        for doc_id, score, metadata in text_results:
            if doc_id in result_map:
                result_map[doc_id]["text_score"] = score
            else:
                result_map[doc_id] = {
                    "vector_score": 0.0,
                    "text_score": score,
                    "metadata": metadata
                }

        # 计算融合分数
        fused_results = []
        for doc_id, scores in result_map.items():
            fused_score = (
                scores["vector_score"] * vector_weight +
                scores["text_score"] * text_weight
            )
            fused_results.append((doc_id, fused_score, scores["metadata"]))

        # 按分数排序
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results

    async def delete_vectors(self, document_ids: List[str], collection_name: str = "default") -> bool:
        """
        删除向量。

        Args:
            document_ids: 文档ID列表
            collection_name: 集合名称

        Returns:
            bool: 删除是否成功
        """
        try:
            if not self.milvus_client:
                raise VectorStoreException("Milvus客户端未初始化")

            # 从Milvus删除
            await self.milvus_client.delete(
                collection_name=collection_name,
                filter_expr=f"id in {json.dumps(document_ids)}"
            )

            # 从Elasticsearch删除
            if self.elasticsearch_client:
                await self.elasticsearch_client.delete_documents("vectors", document_ids)

            self.logger.info(f"成功删除 {len(document_ids)} 个向量")
            return True

        except Exception as e:
            self.logger.error(f"删除向量失败: {e}")
            return False

    async def update_vectors(
        self,
        document_id: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: str = "default"
    ) -> bool:
        """
        更新向量。

        Args:
            document_id: 文档ID
            vector: 新的向量（可选）
            metadata: 新的元数据（可选）
            collection_name: 集合名称

        Returns:
            bool: 更新是否成功
        """
        try:
            if not self.milvus_client:
                raise VectorStoreException("Milvus客户端未初始化")

            # 构建更新数据
            update_data = {}
            if vector is not None:
                update_data["vector"] = [vector]
            if metadata is not None:
                update_data["metadata"] = [json.dumps(metadata)]

            if not update_data:
                return True

            # 执行更新（Milvus 2.3+支持更新操作）
            await self.milvus_client.upsert(
                collection_name=collection_name,
                data=update_data,
                filter_expr=f"id == '{document_id}'"
            )

            # 更新Elasticsearch
            if self.elasticsearch_client and metadata is not None:
                await self.elasticsearch_client.update_document(
                    "vectors",
                    document_id,
                    {"metadata": metadata}
                )

            self.logger.info(f"成功更新向量 {document_id}")
            return True

        except Exception as e:
            self.logger.error(f"更新向量失败: {e}")
            return False

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        获取集合统计信息。

        Args:
            collection_name: 集合名称

        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            if not self.milvus_client:
                raise VectorStoreException("Milvus客户端未初始化")

            stats = await self.milvus_client.get_collection_stats(collection_name)

            return {
                "collection_name": collection_name,
                "row_count": stats.get("row_count", 0),
                "index_info": stats.get("index_descriptions", []),
                "config": self.collection_configs.get(collection_name).__dict__ if collection_name in self.collection_configs else None
            }

        except Exception as e:
            self.logger.error(f"获取集合统计失败: {e}")
            return {}

    def _build_filter_expression(self, filters: Dict[str, Any]) -> str:
        """构建过滤表达式。"""
        if not filters:
            return ""

        expressions = []

        for field, value in filters.items():
            if isinstance(value, str):
                expressions.append(f'metadata["{field}"] == "{value}"')
            elif isinstance(value, (int, float)):
                expressions.append(f'metadata["{field}"] == {value}')
            elif isinstance(value, list):
                if value and isinstance(value[0], str):
                    json_list = json.dumps(value)
                    expressions.append(f'metadata["{field}"] in {json_list}')
                elif value and isinstance(value[0], (int, float)):
                    expressions.append(f'metadata["{field}"] in [{", ".join(map(str, value))}]')

        return " and ".join(expressions)

    async def health_check(self) -> Dict[str, Any]:
        """健康检查。"""
        health_status = {
            "milvus": False,
            "elasticsearch": False,
            "collections": []
        }

        try:
            # 检查Milvus连接
            if self.milvus_client:
                health_status["milvus"] = await self.milvus_client.ping()
        except Exception as e:
            self.logger.error(f"Milvus健康检查失败: {e}")

        try:
            # 检查Elasticsearch连接
            if self.elasticsearch_client:
                es_info = await self.elasticsearch_client.info()
                health_status["elasticsearch"] = True
        except Exception as e:
            self.logger.error(f"Elasticsearch健康检查失败: {e}")

        # 检查集合状态
        for collection_name in self.collection_configs.keys():
            try:
                stats = await self.get_collection_stats(collection_name)
                health_status["collections"].append({
                    "name": collection_name,
                    "status": "healthy",
                    "document_count": stats.get("row_count", 0)
                })
            except Exception as e:
                health_status["collections"].append({
                    "name": collection_name,
                    "status": "error",
                    "error": str(e)
                })

        return health_status

    async def optimize_collection(self, collection_name: str) -> bool:
        """
        优化集合。

        Args:
            collection_name: 集合名称

        Returns:
            bool: 优化是否成功
        """
        try:
            if not self.milvus_client:
                raise VectorStoreException("Milvus客户端未初始化")

            # 执行集合优化
            await self.milvus_client.compact(collection_name)
            self.logger.info(f"集合 {collection_name} 优化完成")
            return True

        except Exception as e:
            self.logger.error(f"优化集合失败: {e}")
            return False

    async def backup_collection(self, collection_name: str, backup_path: str) -> bool:
        """
        备份集合。

        Args:
            collection_name: 集合名称
            backup_path: 备份路径

        Returns:
            bool: 备份是否成功
        """
        try:
            # 这里可以实现集合备份逻辑
            # 例如导出数据到文件
            self.logger.info(f"集合备份功能待实现: {collection_name} -> {backup_path}")
            return False

        except Exception as e:
            self.logger.error(f"备份集合失败: {e}")
            return False