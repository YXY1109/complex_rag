"""
图检索器

基于图结构的文档检索实现。
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, deque

from ..interfaces.retriever_interface import (
    RetrieverInterface,
    RetrievalQuery,
    RetrievalResult,
    DocumentChunk,
    RetrievalStrategy,
    RetrieverConfig,
)
from ...graph_rag.interfaces.entity_interface import (
    EntityModel,
    RelationshipModel,
    CommunityModel,
    EntityType,
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.retriever.strategies.graph_retriever")


class GraphRetriever(RetrieverInterface):
    """
    图检索器

    基于实体关系和社区结构的图检索实现。
    """

    def __init__(self, config: RetrieverConfig, graph_service=None):
        """
        初始化图检索器

        Args:
            config: 检索器配置
            graph_service: 图服务实例
        """
        self.config = config
        self.graph_service = graph_service

        # 图数据结构
        self.entities: Dict[str, EntityModel] = {}
        self.relationships: Dict[str, RelationshipModel] = {}
        self.communities: Dict[str, CommunityModel] = {}
        self.document_entities: Dict[str, List[str]] = defaultdict(list)  # doc_id -> entity_ids
        self.entity_documents: Dict[str, List[str]] = defaultdict(list)  # entity_id -> doc_ids

        # 图邻接表
        self.entity_graph: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> neighbor_entity_ids
        self.community_entities: Dict[str, Set[str]] = defaultdict(set)  # community_id -> entity_ids

        # 缓存
        self._query_cache = {} if config.enable_caching else None
        self._cache_timestamps = {}

        self._initialized = False

    async def initialize(self) -> bool:
        """
        初始化图检索器

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "初始化图检索器",
                extra={
                    "cache_enabled": self.config.enable_caching,
                }
            )

            # 如果提供了图服务，从图服务加载数据
            if self.graph_service:
                await self._load_from_graph_service()

            self._initialized = True
            structured_logger.info("图检索器初始化成功")
            return True

        except Exception as e:
            structured_logger.error(f"图检索器初始化失败: {e}")
            return False

    async def _load_from_graph_service(self) -> None:
        """从图服务加载数据"""
        try:
            # 这里应该从图服务加载实体、关系和社区数据
            # 由于图服务可能还未实现，这里提供一个框架
            structured_logger.info("从图服务加载图数据")

            # TODO: 实现从图服务加载数据的逻辑
            # entities = await self.graph_service.get_all_entities()
            # relationships = await self.graph_service.get_all_relationships()
            # communities = await self.graph_service.get_all_communities()

            # 更新本地图结构
            # self._update_graph_structure(entities, relationships, communities)

        except Exception as e:
            structured_logger.warning(f"从图服务加载数据失败: {e}")

    def _update_graph_structure(
        self,
        entities: List[EntityModel],
        relationships: List[RelationshipModel],
        communities: List[CommunityModel]
    ) -> None:
        """更新图结构"""
        # 更新实体
        for entity in entities:
            self.entities[entity.id] = entity

        # 更新关系和构建邻接表
        for relationship in relationships:
            self.relationships[relationship.id] = relationship

            # 构建无向图的邻接表
            self.entity_graph[relationship.source_entity_id].add(relationship.target_entity_id)
            self.entity_graph[relationship.target_entity_id].add(relationship.source_entity_id)

        # 更新社区
        for community in communities:
            self.communities[community.id] = community
            self.community_entities[community.id] = set(community.entities)

    async def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        添加文档到图检索器

        Args:
            documents: 文档列表，包含实体和关系信息

        Returns:
            List[str]: 文档ID列表
        """
        if not self._initialized:
            raise RuntimeError("图检索器未初始化")

        document_ids = []

        try:
            structured_logger.info(f"开始添加 {len(documents)} 个文档到图检索器")

            for doc in documents:
                doc_id = doc.get("id") or f"doc_{len(self.document_entities)}_{int(time.time())}"

                # 创建文档片段
                chunk = DocumentChunk(
                    id=doc_id,
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    source=doc.get("source"),
                    chunk_index=doc.get("chunk_index", 0),
                    start_pos=doc.get("start_pos", 0),
                    end_pos=doc.get("end_pos", len(doc.get("content", ""))),
                    created_at=datetime.utcnow().isoformat(),
                )

                # 处理文档中的实体
                entities = doc.get("entities", [])
                entity_ids = []

                for entity_data in entities:
                    entity = EntityModel(**entity_data)
                    self.entities[entity.id] = entity
                    entity_ids.append(entity.id)

                    # 更新实体-文档映射
                    self.entity_documents[entity.id].append(doc_id)

                # 处理文档中的关系
                relationships = doc.get("relationships", [])
                for rel_data in relationships:
                    relationship = RelationshipModel(**rel_data)
                    self.relationships[relationship.id] = relationship

                    # 更新图邻接表
                    self.entity_graph[relationship.source_entity_id].add(relationship.target_entity_id)
                    self.entity_graph[relationship.target_entity_id].add(relationship.source_entity_id)

                # 更新文档-实体映射
                if entity_ids:
                    self.document_entities[doc_id] = entity_ids

                document_ids.append(doc_id)

            structured_logger.info(f"成功添加 {len(document_ids)} 个文档到图检索器")
            return document_ids

        except Exception as e:
            structured_logger.error(f"添加文档失败: {e}")
            raise Exception(f"Failed to add documents: {e}")

    async def retrieve(
        self,
        query: RetrievalQuery
    ) -> RetrievalResult:
        """
        执行图检索

        Args:
            query: 检索查询

        Returns:
            RetrievalResult: 检索结果
        """
        if not self._initialized:
            raise RuntimeError("图检索器未初始化")

        start_time = time.time()

        try:
            # 检查缓存
            cache_key = f"{query.text}_{query.top_k}_{query.min_score}"
            if self._query_cache and cache_key in self._query_cache:
                cached_result = self._query_cache[cache_key]
                structured_logger.debug(f"使用缓存结果: {query.text[:50]}...")
                return cached_result

            structured_logger.info(
                f"开始图检索",
                extra={
                    "query_length": len(query.text),
                    "top_k": query.top_k,
                    "min_score": query.min_score,
                }
            )

            # 从查询中抽取实体（简化实现）
            query_entities = await self._extract_query_entities(query.text)

            # 基于实体的图检索
            doc_scores = await self._graph_based_retrieval(query_entities, query)

            # 排序和构建结果
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

            chunks = []
            chunk_scores = []
            explanations = []

            for doc_id, score in sorted_docs:
                if score <= query.min_score:
                    break

                if doc_id in self.document_entities:
                    # 创建文档片段（这里简化处理，实际应该存储完整文档内容）
                    chunk = DocumentChunk(
                        id=doc_id,
                        content="",  # 图检索主要基于实体关系，不直接存储内容
                        metadata={"graph_score": score, "entities": self.document_entities[doc_id]},
                        score=score,
                        created_at=datetime.utcnow().isoformat(),
                    )

                    chunks.append(chunk)
                    chunk_scores.append(score)

                    # 生成解释
                    doc_entities = self.document_entities[doc_id]
                    common_entities = set(query_entities) & set(doc_entities)
                    explanation = f"图检索分数: {score:.3f}, 共同实体: {list(common_entities)}"
                    explanations.append(explanation)

                if len(chunks) >= query.max_results:
                    break

            processing_time = (time.time() - start_time) * 1000

            result = RetrievalResult(
                chunks=chunks,
                query=query.text,
                strategy=RetrievalStrategy.GRAPH,
                total_found=len(chunks),
                search_time_ms=processing_time,
                scores=chunk_scores,
                explanations=explanations,
                metadata={
                    "total_entities": len(self.entities),
                    "total_relationships": len(self.relationships),
                    "total_communities": len(self.communities),
                    "query_entities": query_entities,
                },
                created_at=datetime.utcnow().isoformat(),
            )

            # 缓存结果
            if self._query_cache:
                self._query_cache[cache_key] = result
                self._cache_timestamps[cache_key] = time.time()

            structured_logger.info(
                f"图检索完成",
                extra={
                    "results_count": len(chunks),
                    "processing_time_ms": processing_time,
                }
            )

            return result

        except Exception as e:
            structured_logger.error(f"图检索失败: {e}")
            raise Exception(f"Graph retrieval failed: {e}")

    async def _extract_query_entities(self, query_text: str) -> List[str]:
        """从查询中抽取实体（简化实现）"""
        # 这是一个简化的实现，实际应用中应该使用实体抽取服务
        query_entities = []

        # 基于现有实体列表进行匹配
        query_lower = query_text.lower()
        for entity_id, entity in self.entities.items():
            if entity.name.lower() in query_lower:
                query_entities.append(entity_id)

        return query_entities

    async def _graph_based_retrieval(
        self,
        query_entities: List[str],
        query: RetrievalQuery
    ) -> Dict[str, float]:
        """基于图的检索"""
        doc_scores = defaultdict(float)

        if not query_entities:
            return doc_scores

        # 方法1: 直接实体匹配
        for entity_id in query_entities:
            for doc_id in self.entity_documents.get(entity_id, []):
                doc_scores[doc_id] += 1.0

        # 方法2: 邻居实体匹配
        neighbor_entities = set()
        for entity_id in query_entities:
            neighbor_entities.update(self.entity_graph.get(entity_id, set()))

        for neighbor_id in neighbor_entities:
            neighbor_weight = 0.5  # 邻居实体权重较低
            for doc_id in self.entity_documents.get(neighbor_id, []):
                doc_scores[doc_id] += neighbor_weight

        # 方法3: 社区匹配
        entity_communities = defaultdict(set)
        for community_id, community_entities in self.community_entities.items():
            for entity_id in community_entities:
                entity_communities[entity_id].add(community_id)

        query_communities = set()
        for entity_id in query_entities:
            query_communities.update(entity_communities.get(entity_id, set()))

        for community_id in query_communities:
            community_weight = 0.3  # 社区权重更低
            community_entity_ids = self.community_entities.get(community_id, set())
            for entity_id in community_entity_ids:
                for doc_id in self.entity_documents.get(entity_id, []):
                    doc_scores[doc_id] += community_weight

        # 方法4: 图路径评分（简化实现）
        for doc_id in self.document_entities:
            doc_entities = set(self.document_entities[doc_id])
            path_score = self._calculate_path_score(query_entities, doc_entities)
            doc_scores[doc_id] += path_score

        return dict(doc_scores)

    def _calculate_path_score(self, query_entities: List[str], doc_entities: Set[str]) -> float:
        """计算图路径分数"""
        if not query_entities or not doc_entities:
            return 0.0

        total_path_score = 0.0
        connection_count = 0

        for query_entity in query_entities:
            for doc_entity in doc_entities:
                # 计算两个实体之间的最短路径
                path_length = self._shortest_path_length(query_entity, doc_entity)
                if path_length is not None:
                    # 路径越短，分数越高
                    score = 1.0 / (path_length + 1)
                    total_path_score += score
                    connection_count += 1

        if connection_count == 0:
            return 0.0

        return total_path_score / connection_count

    def _shortest_path_length(self, source: str, target: str) -> Optional[int]:
        """计算两个实体之间的最短路径长度"""
        if source == target:
            return 0

        if source not in self.entity_graph or target not in self.entity_graph:
            return None

        # BFS算法
        visited = set()
        queue = deque([(source, 0)])

        while queue:
            current, distance = queue.popleft()
            if current == target:
                return distance

            if current in visited:
                continue

            visited.add(current)

            for neighbor in self.entity_graph.get(current, set()):
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))

        return None

    async def batch_retrieve(
        self,
        queries: List[RetrievalQuery]
    ) -> List[RetrievalResult]:
        """批量图检索"""
        if not self._initialized:
            raise RuntimeError("图检索器未初始化")

        try:
            structured_logger.info(f"开始批量图检索，查询数量: {len(queries)}")

            # 并行处理查询
            tasks = [self.retrieve(query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理异常
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    structured_logger.error(f"查询 {i} 处理失败: {result}")
                    valid_results.append(RetrievalResult(
                        chunks=[],
                        query=queries[i].text,
                        strategy=RetrievalStrategy.GRAPH,
                        total_found=0,
                        search_time_ms=0.0,
                        created_at=datetime.utcnow().isoformat(),
                    ))
                else:
                    valid_results.append(result)

            structured_logger.info(f"批量图检索完成，成功处理 {len(valid_results)} 个查询")
            return valid_results

        except Exception as e:
            structured_logger.error(f"批量图检索失败: {e}")
            raise Exception(f"Batch graph retrieval failed: {e}")

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """删除文档"""
        try:
            deleted_count = 0

            for doc_id in document_ids:
                if doc_id in self.document_entities:
                    # 获取文档的实体
                    entity_ids = self.document_entities[doc_id]

                    # 从实体-文档映射中删除
                    for entity_id in entity_ids:
                        if entity_id in self.entity_documents:
                            self.entity_documents[entity_id].remove(doc_id)

                    # 删除文档-实体映射
                    del self.document_entities[doc_id]
                    deleted_count += 1

            structured_logger.info(f"删除了 {deleted_count} 个文档")
            return True

        except Exception as e:
            structured_logger.error(f"删除文档失败: {e}")
            return False

    async def update_document(self, document_id: str, document: Dict[str, Any]) -> bool:
        """更新文档"""
        try:
            if document_id not in self.document_entities:
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
        if document_id in self.document_entities:
            return {
                "id": document_id,
                "entities": self.document_entities[document_id],
                "metadata": {},
            }
        return None

    async def search_similar(self, document_id: str, top_k: int = 10) -> List[DocumentChunk]:
        """搜索相似文档"""
        if document_id not in self.document_entities:
            return []

        try:
            # 获取文档的实体
            doc_entities = set(self.document_entities[document_id])

            # 找到具有相似实体的文档
            similar_docs = []
            for other_doc_id, other_entities in self.document_entities.items():
                if other_doc_id == document_id:
                    continue

                other_entity_set = set(other_entities)
                # 计算Jaccard相似度
                intersection = len(doc_entities & other_entity_set)
                union = len(doc_entities | other_entity_set)

                if union > 0:
                    similarity = intersection / union
                    if similarity > 0.1:  # 最低相似度阈值
                        chunk = DocumentChunk(
                            id=other_doc_id,
                            content="",
                            metadata={"similarity": similarity, "entities": other_entities},
                            score=similarity,
                            created_at=datetime.utcnow().isoformat(),
                        )
                        similar_docs.append(chunk)

            # 按相似度排序
            similar_docs.sort(key=lambda x: x.score, reverse=True)
            return similar_docs[:top_k]

        except Exception as e:
            structured_logger.error(f"搜索相似文档失败: {e}")
            return []

    async def expand_query(self, query: str, max_terms: int = 5) -> List[str]:
        """基于图的查询扩展"""
        expanded_terms = [query]

        # 从查询中抽取实体
        query_entities = await self._extract_query_entities(query)

        # 找到相关实体
        related_entities = set()
        for entity_id in query_entities:
            # 邻居实体
            related_entities.update(self.entity_graph.get(entity_id, set()))

            # 同社区实体
            for community_id, community_entities in self.community_entities.items():
                if entity_id in community_entities:
                    related_entities.update(community_entities)

        # 获取实体名称
        for entity_id in list(related_entities)[:max_terms - 1]:
            if entity_id in self.entities:
                expanded_terms.append(self.entities[entity_id].name)

        return expanded_terms[:max_terms]

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "total_communities": len(self.communities),
            "total_documents": len(self.document_entities),
            "cache_size": len(self._query_cache) if self._query_cache else 0,
            "graph_density": self._calculate_graph_density(),
            "initialized": self._initialized,
        }

    def _calculate_graph_density(self) -> float:
        """计算图密度"""
        num_entities = len(self.entities)
        if num_entities < 2:
            return 0.0

        num_edges = len(self.relationships)
        max_edges = num_entities * (num_entities - 1) / 2

        return num_edges / max_edges if max_edges > 0 else 0.0

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 执行简单检索测试
            test_query = RetrievalQuery(text="test", top_k=1)
            test_result = await self.retrieve(test_query)

            return {
                "status": "healthy",
                "initialized": self._initialized,
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
                "total_documents": len(self.document_entities),
                "cache_enabled": self._query_cache is not None,
                "test_retrieval_time_ms": test_result.search_time_ms,
                "graph_density": self._calculate_graph_density(),
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
            self.entities.clear()
            self.relationships.clear()
            self.communities.clear()
            self.document_entities.clear()
            self.entity_documents.clear()
            self.entity_graph.clear()
            self.community_entities.clear()

            if self._query_cache:
                self._query_cache.clear()
            self._cache_timestamps.clear()

            self._initialized = False
            structured_logger.info("图检索器清理完成")

        except Exception as e:
            structured_logger.error(f"图检索器清理失败: {e}")