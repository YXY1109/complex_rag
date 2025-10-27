"""
向量检索器

基于向量相似度的文档检索实现。
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from ..interfaces.retriever_interface import (
    RetrieverInterface,
    RetrievalQuery,
    RetrievalResult,
    DocumentChunk,
    RetrievalStrategy,
    RetrieverConfig,
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.retriever.strategies.vector_retriever")


@dataclass
class VectorIndexConfig:
    """向量索引配置"""
    dimension: int = 768
    metric: str = "cosine"  # cosine, euclidean, manhattan
    index_type: str = "flat"  # flat, ivf, hnsw
    ef_construction: int = 200
    ef_search: int = 50
    nlist: int = 100
    nprobe: int = 10


class VectorRetriever(RetrieverInterface):
    """
    向量检索器

    基于向量相似度进行文档检索，支持多种距离度量和索引类型。
    """

    def __init__(self, config: RetrieverConfig, embedding_service=None):
        """
        初始化向量检索器

        Args:
            config: 检索器配置
            embedding_service: 向量化服务
        """
        self.config = config
        self.embedding_service = embedding_service
        self.vector_index_config = VectorIndexConfig()

        # 向量存储
        self.documents: Dict[str, DocumentChunk] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.document_ids: List[str] = []
        self.embedding_matrix: Optional[np.ndarray] = None

        # 缓存
        self._query_cache = {} if config.enable_caching else None
        self._cache_timestamps = {}

        self._initialized = False

    async def initialize(self) -> bool:
        """
        初始化向量检索器

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "初始化向量检索器",
                extra={
                    "dimension": self.vector_index_config.dimension,
                    "metric": self.vector_index_config.metric,
                    "index_type": self.vector_index_config.index_type,
                }
            )

            # 初始化向量化服务
            if not self.embedding_service:
                structured_logger.warning("未提供向量化服务，将在需要时创建默认实现")

            # 构建初始索引
            await self._rebuild_index()

            self._initialized = True
            structured_logger.info("向量检索器初始化成功")
            return True

        except Exception as e:
            structured_logger.error(f"向量检索器初始化失败: {e}")
            return False

    async def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        添加文档到向量检索器

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档ID列表
        """
        if not self._initialized:
            raise RuntimeError("向量检索器未初始化")

        document_ids = []
        batch_size = 32  # 批量处理大小

        try:
            structured_logger.info(
                f"开始添加 {len(documents)} 个文档到向量检索器"
            )

            # 分批处理文档
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_ids = await self._process_document_batch(batch)
                document_ids.extend(batch_ids)

                # 定期重建索引
                if i % (batch_size * 10) == 0:
                    await self._rebuild_index()

            # 最终重建索引
            await self._rebuild_index()

            structured_logger.info(
                f"成功添加 {len(document_ids)} 个文档到向量检索器"
            )

            return document_ids

        except Exception as e:
            structured_logger.error(f"添加文档失败: {e}")
            raise Exception(f"Failed to add documents: {e}")

    async def _process_document_batch(self, documents: List[Dict[str, Any]]) -> List[str]:
        """处理文档批次"""
        document_ids = []
        texts = []

        # 准备文本和元数据
        for doc in documents:
            doc_id = doc.get("id") or f"doc_{len(self.documents)}_{int(time.time())}"
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # 创建文档片段
            chunk = DocumentChunk(
                id=doc_id,
                content=content,
                metadata=metadata,
                source=doc.get("source"),
                chunk_index=doc.get("chunk_index", 0),
                start_pos=doc.get("start_pos", 0),
                end_pos=doc.get("end_pos", len(content)),
                created_at=datetime.utcnow().isoformat(),
            )

            self.documents[doc_id] = chunk
            document_ids.append(doc_id)
            texts.append(content)

        # 生成向量
        embeddings = await self._generate_embeddings(texts)

        # 存储向量
        for doc_id, embedding in zip(document_ids, embeddings):
            self.embeddings[doc_id] = embedding

        return document_ids

    async def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """生成文本向量"""
        if self.embedding_service:
            # 使用外部向量化服务
            embeddings = await self.embedding_service.embed_texts(texts)
            return [np.array(emb) for emb in embeddings]
        else:
            # 简单的默认实现（基于TF-IDF的伪向量）
            return await self._generate_simple_embeddings(texts)

    async def _generate_simple_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """生成简单的文本向量（用于演示）"""
        # 这是一个简化的实现，实际应用中应该使用专业的向量化模型
        dimension = self.vector_index_config.dimension
        embeddings = []

        for text in texts:
            # 基于词频的简单向量
            words = text.lower().split()
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # 创建固定长度的向量
            vector = np.zeros(dimension)
            for i, word in enumerate(list(word_counts.keys())[:dimension]):
                if i < dimension:
                    vector[i] = word_counts[word] / len(words)

            # 归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            embeddings.append(vector)

        return embeddings

    async def _rebuild_index(self) -> None:
        """重建向量索引"""
        if not self.documents:
            return

        try:
            # 更新文档ID列表
            self.document_ids = list(self.documents.keys())

            # 构建向量矩阵
            embeddings_list = [self.embeddings[doc_id] for doc_id in self.document_ids]
            if embeddings_list:
                self.embedding_matrix = np.vstack(embeddings_list)
            else:
                self.embedding_matrix = np.array([]).reshape(0, self.vector_index_config.dimension)

            structured_logger.debug(
                f"重建向量索引完成，包含 {len(self.document_ids)} 个文档"
            )

        except Exception as e:
            structured_logger.error(f"重建向量索引失败: {e}")

    async def retrieve(
        self,
        query: RetrievalQuery
    ) -> RetrievalResult:
        """
        执行向量检索

        Args:
            query: 检索查询

        Returns:
            RetrievalResult: 检索结果
        """
        if not self._initialized:
            raise RuntimeError("向量检索器未初始化")

        start_time = time.time()

        try:
            # 检查缓存
            if self._query_cache and query.text in self._query_cache:
                cached_result = self._query_cache[query.text]
                structured_logger.debug(f"使用缓存结果: {query.text[:50]}...")
                return cached_result

            structured_logger.info(
                f"开始向量检索",
                extra={
                    "query_length": len(query.text),
                    "top_k": query.top_k,
                    "min_score": query.min_score,
                }
            )

            # 生成查询向量
            query_embedding = await self._generate_query_embedding(query)

            # 执行向量搜索
            scores, indices = await self._vector_search(query_embedding, query.top_k)

            # 构建结果
            chunks = []
            chunk_scores = []
            explanations = []

            for score, idx in zip(scores, indices):
                if idx < len(self.document_ids):
                    doc_id = self.document_ids[idx]
                    chunk = self.documents[doc_id]

                    # 应用最小分数阈值
                    if score >= query.min_score:
                        chunk.score = score
                        chunks.append(chunk)
                        chunk_scores.append(score)
                        explanations.append(f"向量相似度: {score:.3f}")

            # 限制结果数量
            if len(chunks) > query.max_results:
                chunks = chunks[:query.max_results]
                chunk_scores = chunk_scores[:query.max_results]
                explanations = explanations[:query.max_results]

            processing_time = (time.time() - start_time) * 1000

            result = RetrievalResult(
                chunks=chunks,
                query=query.text,
                strategy=RetrievalStrategy.VECTOR,
                total_found=len(chunks),
                search_time_ms=processing_time,
                scores=chunk_scores,
                explanations=explanations,
                metadata={
                    "embedding_dimension": len(query_embedding) if query_embedding is not None else 0,
                    "total_documents": len(self.documents),
                    "metric": self.vector_index_config.metric,
                },
                created_at=datetime.utcnow().isoformat(),
            )

            # 缓存结果
            if self._query_cache:
                self._query_cache[query.text] = result
                self._cache_timestamps[query.text] = time.time()

            structured_logger.info(
                f"向量检索完成",
                extra={
                    "results_count": len(chunks),
                    "processing_time_ms": processing_time,
                }
            )

            return result

        except Exception as e:
            structured_logger.error(f"向量检索失败: {e}")
            raise Exception(f"Vector retrieval failed: {e}")

    async def _generate_query_embedding(self, query: RetrievalQuery) -> np.ndarray:
        """生成查询向量"""
        if query.query_embedding:
            return np.array(query.query_embedding)

        # 使用向量化服务生成查询向量
        embeddings = await self._generate_embeddings([query.text])
        return embeddings[0] if embeddings else np.zeros(self.vector_index_config.dimension)

    async def _vector_search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, List[int]]:
        """执行向量搜索"""
        if self.embedding_matrix is None or self.embedding_matrix.size == 0:
            return np.array([]), []

        # 计算相似度
        if self.vector_index_config.metric == "cosine":
            # 余弦相似度
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return np.array([]), []

            normalized_query = query_embedding / query_norm

            # 计算与所有文档的余弦相似度
            similarities = np.dot(self.embedding_matrix, normalized_query)

            # 获取top-k结果
            if len(similarities) > top_k:
                top_indices = np.argpartition(similarities, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(-similarities[top_indices])]
            else:
                top_indices = np.argsort(-similarities)

            top_scores = similarities[top_indices]

        elif self.vector_index_config.metric == "euclidean":
            # 欧氏距离（转换为相似度）
            distances = np.linalg.norm(self.embedding_matrix - query_embedding, axis=1)
            similarities = 1.0 / (1.0 + distances)  # 转换为相似度

            if len(similarities) > top_k:
                top_indices = np.argpartition(similarities, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(-similarities[top_indices])]
            else:
                top_indices = np.argsort(-similarities)

            top_scores = similarities[top_indices]

        else:
            # 默认使用余弦相似度
            similarities = np.dot(self.embedding_matrix, query_embedding)
            if len(similarities) > top_k:
                top_indices = np.argpartition(similarities, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(-similarities[top_indices])]
            else:
                top_indices = np.argsort(-similarities)

            top_scores = similarities[top_indices]

        return top_scores, top_indices.tolist()

    async def batch_retrieve(
        self,
        queries: List[RetrievalQuery]
    ) -> List[RetrievalResult]:
        """
        批量向量检索

        Args:
            queries: 查询列表

        Returns:
            List[RetrievalResult]: 检索结果列表
        """
        if not self._initialized:
            raise RuntimeError("向量检索器未初始化")

        try:
            structured_logger.info(f"开始批量向量检索，查询数量: {len(queries)}")

            # 并行处理查询
            tasks = [self.retrieve(query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理异常
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    structured_logger.error(f"查询 {i} 处理失败: {result}")
                    # 创建空结果
                    valid_results.append(RetrievalResult(
                        chunks=[],
                        query=queries[i].text,
                        strategy=RetrievalStrategy.VECTOR,
                        total_found=0,
                        search_time_ms=0.0,
                        created_at=datetime.utcnow().isoformat(),
                    ))
                else:
                    valid_results.append(result)

            structured_logger.info(f"批量向量检索完成，成功处理 {len(valid_results)} 个查询")
            return valid_results

        except Exception as e:
            structured_logger.error(f"批量向量检索失败: {e}")
            raise Exception(f"Batch vector retrieval failed: {e}")

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """删除文档"""
        try:
            deleted_count = 0
            for doc_id in document_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                    deleted_count += 1
                if doc_id in self.embeddings:
                    del self.embeddings[doc_id]

            if deleted_count > 0:
                await self._rebuild_index()

            structured_logger.info(f"删除了 {deleted_count} 个文档")
            return True

        except Exception as e:
            structured_logger.error(f"删除文档失败: {e}")
            return False

    async def update_document(self, document_id: str, document: Dict[str, Any]) -> bool:
        """更新文档"""
        try:
            if document_id not in self.documents:
                return False

            # 先删除旧文档
            await self.delete_documents([document_id])

            # 添加新文档
            document["id"] = document_id
            await self.add_documents([document])

            return True

        except Exception as e:
            structured_logger.error(f"更新文档失败: {e}")
            return False

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        if document_id in self.documents:
            chunk = self.documents[document_id]
            return {
                "id": chunk.id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
                "score": chunk.score,
                "created_at": chunk.created_at,
            }
        return None

    async def search_similar(self, document_id: str, top_k: int = 10) -> List[DocumentChunk]:
        """搜索相似文档"""
        if document_id not in self.documents:
            return []

        try:
            # 获取文档向量
            doc_embedding = self.embeddings.get(document_id)
            if doc_embedding is None:
                return []

            # 执行相似度搜索
            scores, indices = await self._vector_search(doc_embedding, top_k + 1)  # +1 因为包含自己

            similar_chunks = []
            for score, idx in zip(scores, indices):
                if idx < len(self.document_ids):
                    similar_doc_id = self.document_ids[idx]
                    if similar_doc_id != document_id:  # 排除自己
                        chunk = self.documents[similar_doc_id]
                        chunk.score = score
                        similar_chunks.append(chunk)

            return similar_chunks[:top_k]

        except Exception as e:
            structured_logger.error(f"搜索相似文档失败: {e}")
            return []

    async def expand_query(self, query: str, max_terms: int = 5) -> List[str]:
        """查询扩展（简化实现）"""
        # 简单的查询扩展：基于同义词和相关词
        # 实际应用中可以使用更复杂的扩展策略
        expanded_terms = [query]

        # 基于词频的简单扩展
        words = query.lower().split()
        for word in words:
            # 这里可以集成同义词词典或词向量模型
            synonyms = await self._get_synonyms(word, max_terms // len(words))
            expanded_terms.extend(synonyms)

        return expanded_terms[:max_terms]

    async def _get_synonyms(self, word: str, max_synonyms: int) -> List[str]:
        """获取同义词（简化实现）"""
        # 这是一个简化的实现，实际应用中应该使用专业的同义词词典
        synonym_map = {
            "ai": ["artificial intelligence", "machine learning", "deep learning"],
            "computer": ["pc", "desktop", "laptop"],
            "phone": ["mobile", "smartphone", "cellphone"],
            "car": ["automobile", "vehicle", "auto"],
        }

        return synonym_map.get(word.lower(), [])[:max_synonyms]

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_documents": len(self.documents),
            "index_type": self.vector_index_config.index_type,
            "dimension": self.vector_index_config.dimension,
            "metric": self.vector_index_config.metric,
            "cache_size": len(self._query_cache) if self._query_cache else 0,
            "memory_usage_mb": self._estimate_memory_usage(),
            "initialized": self._initialized,
        }

    def _estimate_memory_usage(self) -> float:
        """估算内存使用量（MB）"""
        if self.embedding_matrix is None:
            return 0.0

        # 向量矩阵内存
        matrix_memory = self.embedding_matrix.nbytes / (1024 * 1024)

        # 文档数据内存
        docs_memory = len(str(self.documents)) / (1024 * 1024)

        return matrix_memory + docs_memory

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 执行简单检索测试
            test_query = RetrievalQuery(text="test", top_k=1)
            test_result = await self.retrieve(test_query)

            return {
                "status": "healthy",
                "initialized": self._initialized,
                "total_documents": len(self.documents),
                "index_dimension": self.vector_index_config.dimension,
                "cache_enabled": self._query_cache is not None,
                "test_retrieval_time_ms": test_result.search_time_ms,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "initialized": self._initialized,
            }

    async def cleanup(self) -> None:
        """清理资源"""
        try:
            self.documents.clear()
            self.embeddings.clear()
            self.document_ids.clear()
            self.embedding_matrix = None

            if self._query_cache:
                self._query_cache.clear()
            self._cache_timestamps.clear()

            self._initialized = False
            structured_logger.info("向量检索器清理完成")

        except Exception as e:
            structured_logger.error(f"向量检索器清理失败: {e}")