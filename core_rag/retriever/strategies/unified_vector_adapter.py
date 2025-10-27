"""
统一向量存储适配器

将core_rag的检索器接口适配到新的统一向量存储服务。
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

from ..interfaces.retriever_interface import (
    RetrieverInterface,
    RetrievalQuery,
    RetrievalResult,
    DocumentChunk,
    RetrieverConfig
)
from ...rag_service.services.unified_vector_store import (
    UnifiedVectorStore, VectorData, SearchQuery, SearchResult
)
from ...rag_service.services.unified_retriever import (
    UnifiedRetriever, RetrievalConfig, RetrievalStrategy,
    DocumentChunk as UnifiedDocumentChunk, RetrievalQuery as UnifiedRetrievalQuery
)


class UnifiedVectorStoreAdapter(UnifiedRetriever, RetrieverInterface):
    """统一向量存储适配器"""

    def __init__(self, config: RetrieverConfig, vector_store: UnifiedVectorStore):
        # 转换配置
        retrieval_config = RetrievalConfig(
            strategy=RetrievalStrategy.VECTOR_ONLY,
            top_k=config.top_k or 10,
            min_score=config.min_score or 0.0,
            enable_caching=config.enable_caching or True,
            cache_ttl=config.cache_ttl or 3600,
            parallel_search=config.parallel_search or True,
            search_timeout=config.search_timeout or 30
        )

        # 初始化统一检索器
        super().__init__(retrieval_config, vector_store)

        # 保存原始配置
        self.original_config = config
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> bool:
        """初始化适配器"""
        try:
            self.logger.info("初始化统一向量存储适配器")

            # 初始化统一检索器
            success = await super().initialize()

            if success:
                self.logger.info("统一向量存储适配器初始化成功")

            return success

        except Exception as e:
            self.logger.error(f"统一向量存储适配器初始化失败: {str(e)}")
            return False

    async def query(self, query: RetrievalQuery) -> RetrievalResult:
        """执行检索查询（core_rag接口）"""
        start_time = time.time()

        try:
            # 转换查询格式
            unified_query = self._convert_query(query)

            # 执行统一检索
            unified_result = await self.retrieve(unified_query)

            # 转换结果格式
            core_rag_result = self._convert_result(unified_result)

            # 记录查询时间
            search_time = time.time() - start_time
            core_rag_result.search_time = search_time

            self.logger.info(f"检索完成，返回 {len(core_rag_result.chunks)} 个结果，耗时 {search_time:.3f}s")

            return core_rag_result

        except Exception as e:
            self.logger.error(f"检索查询失败: {str(e)}")
            raise

    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """添加文档（core_rag接口）"""
        try:
            # 转换文档格式
            unified_docs = []

            for doc in documents:
                # 生成文档ID
                doc_id = doc.get('id') or f"doc_{int(time.time() * 1000)}"

                # 转换为统一文档格式
                unified_doc = UnifiedDocumentChunk(
                    id=doc_id,
                    content=doc.get('content', ''),
                    metadata={
                        'title': doc.get('title', ''),
                        'source': doc.get('source', ''),
                        'author': doc.get('author', ''),
                        'created_at': doc.get('created_at'),
                        **doc.get('metadata', {})
                    },
                    vector=doc.get('embedding'),  # 如果有预计算的向量
                    created_at=doc.get('created_at'),
                    updated_at=doc.get('updated_at')
                )
                unified_docs.append(unified_doc)

            # 使用默认集合名称
            collection_name = self.original_config.collection_name or 'default'

            # 添加到统一向量存储
            document_ids = await self.add_documents(unified_docs, collection_name)

            self.logger.info(f"添加了 {len(documents)} 个文档到集合 {collection_name}")
            return document_ids

        except Exception as e:
            self.logger.error(f"添加文档失败: {str(e)}")
            return []

    async def update_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """更新文档（core_rag接口）"""
        # 对于统一向量存储，更新和添加是相同的操作（基于ID）
        return await self.add_documents(documents)

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """删除文档（core_rag接口）"""
        try:
            # 使用默认集合名称
            collection_name = self.original_config.collection_name or 'default'

            # 从统一向量存储删除
            deleted_count = await self.delete_documents(document_ids, collection_name)

            success = deleted_count == len(document_ids)

            if success:
                self.logger.info(f"删除了 {len(document_ids)} 个文档")
            else:
                self.logger.warning(f"只删除了 {deleted_count}/{len(document_ids)} 个文档")

            return success

        except Exception as e:
            self.logger.error(f"删除文档失败: {str(e)}")
            return False

    async def clear_cache(self) -> None:
        """清空缓存"""
        try:
            self._query_cache.clear()
            self._cache_timestamps.clear()
            self.logger.info("缓存已清空")
        except Exception as e:
            self.logger.error(f"清空缓存失败: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计"""
        try:
            # 获取统一检索器统计
            unified_stats = super().get_stats()

            # 转换为core_rag格式
            core_rag_stats = {
                'total_queries': unified_stats.get('total_queries', 0),
                'cache_hits': unified_stats.get('cache_hits', 0),
                'cache_misses': unified_stats.get('cache_misses', 0),
                'average_search_time': unified_stats.get('average_search_time', 0.0),
                'total_chunks_retrieved': unified_stats.get('total_chunks_retrieved', 0),
                'cache_hit_rate': unified_stats.get('cache_hit_rate', 0.0),
                'last_query_time': unified_stats.get('last_query_time'),
                'strategy_used': 'unified_vector_store',
                'backend_type': self.vector_store.get_backend('default').backend_type.value,
                'vector_dimension': self.config.top_k  # 这里简化处理
            }

            return core_rag_stats

        except Exception as e:
            self.logger.error(f"获取统计信息失败: {str(e)}")
            return {}

    def _convert_query(self, core_rag_query: RetrievalQuery) -> UnifiedRetrievalQuery:
        """将core_rag查询转换为统一检索查询"""
        # 构建过滤器
        filters = []

        if core_rag_query.filters:
            for key, value in core_rag_query.filters.items():
                # 这里简化处理，实际可以支持更复杂的过滤逻辑
                from ...rag_service.services.unified_retriever import QueryFilter, FilterOperator
                filter_obj = QueryFilter(
                    field=key,
                    operator=FilterOperator.EQ,
                    value=value
                )
                filters.append(filter_obj)

        # 转换查询
        unified_query = UnifiedRetrievalQuery(
            query=core_rag_query.query,
            query_vector=core_rag_query.query_vector,
            collection_name=core_rag_query.collection_name,
            filters=filters,
            user_id=core_rag_query.user_id,
            tenant_id=core_rag_query.tenant_id,
            metadata=core_rag_query.metadata
        )

        return unified_query

    def _convert_result(self, unified_result) -> RetrievalResult:
        """将统一检索结果转换为core_rag格式"""
        # 转换文档块
        core_rag_chunks = []

        for chunk in unified_result.chunks:
            core_rag_chunk = DocumentChunk(
                id=chunk.id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=chunk.score,
                vector=chunk.vector,
                source=chunk.metadata.get('source', 'unknown'),
                title=chunk.metadata.get('title', ''),
                chunk_index=chunk.metadata.get('chunk_index', 0)
            )
            core_rag_chunks.append(core_rag_chunk)

        # 构建core_rag结果
        core_rag_result = RetrievalResult(
            query=unified_result.query,
            chunks=core_rag_chunks,
            total_found=len(core_rag_chunks),
            search_time=unified_result.search_time,
            strategy_used=unified_result.strategy_used,
            cache_hit=unified_result.cache_hit,
            metadata={
                'collection_name': unified_result.metadata.get('collection_name'),
                'backend_type': unified_result.stats.get('backend'),
                'vector_dimension': unified_result.metadata.get('vector_dimension')
            }
        )

        return core_rag_result


def create_unified_adapter(
    config: RetrieverConfig,
    vector_store: UnifiedVectorStore
) -> UnifiedVectorStoreAdapter:
    """创建统一向量存储适配器"""
    return UnifiedVectorStoreAdapter(config, vector_store)