"""
社区发现模块

实现GraphRAG General模式的社区发现功能。
"""

import time
import math
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import asyncio

from ..interfaces.entity_interface import (
    EntityInterface,
    EntityModel,
    RelationshipModel,
    CommunityModel,
    CommunityDetectionRequest,
    CommunityDetectionResponse,
    EntityType,
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.graph_rag.general.community")


class CommunityDetector(EntityInterface):
    """
    社区发现器

    基于图算法的社区发现，支持Leiden、Louvain、Infomap等算法。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化社区发现器

        Args:
            config: 配置参数
        """
        self.config = config
        self.default_algorithm = config.get("algorithm", "leiden")
        self.default_resolution = config.get("resolution", 1.0)
        self.min_community_size = config.get("min_community_size", 3)
        self.max_communities = config.get("max_communities", 100)
        self.enable_hierarchical = config.get("enable_hierarchical", True)
        self.community_similarity_threshold = config.get("community_similarity_threshold", 0.3)

    async def initialize(self) -> bool:
        """
        初始化社区发现器

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "社区发现器初始化成功",
                extra={
                    "default_algorithm": self.default_algorithm,
                    "default_resolution": self.default_resolution,
                    "min_community_size": self.min_community_size,
                }
            )
            return True
        except Exception as e:
            structured_logger.error(f"社区发现器初始化失败: {e}")
            return False

    async def extract_entities(self, request: "EntityExtractionRequest") -> "EntityExtractionResponse":
        """实体抽取（占位实现）"""
        return CommunityDetectionResponse(
            communities=[],
            total_communities=0,
            total_entities_assigned=0,
            processing_time_ms=0.0,
            created_at=datetime.utcnow().isoformat(),
        )

    async def resolve_entities(self, request: "EntityResolutionRequest") -> "EntityResolutionResponse":
        """实体解析（占位实现）"""
        return EntityResolutionResponse(
            resolved_entities=[],
            merged_entities=[],
            resolution_mappings={},
            processing_time_ms=0.0,
            created_at=datetime.utcnow().isoformat(),
        )

    async def detect_communities(
        self,
        request: CommunityDetectionRequest
    ) -> CommunityDetectionResponse:
        """
        检测社区

        Args:
            request: 社区发现请求

        Returns:
            CommunityDetectionResponse: 社区检测结果
        """
        start_time = time.time()

        try:
            structured_logger.info(
                "开始社区发现",
                extra={
                    "entities_count": len(request.entities),
                    "relationships_count": len(request.relationships),
                    "algorithm": request.algorithm,
                    "min_community_size": request.min_community_size,
                }
            )

            # 构建图结构
            graph = await self._build_graph(request.entities, request.relationships)

            # 选择算法进行社区发现
            if request.algorithm == "leiden":
                communities = await self._leiden_algorithm(graph, request)
            elif request.algorithm == "louvain":
                communities = await self._louvain_algorithm(graph, request)
            elif request.algorithm == "infomap":
                communities = await self._infomap_algorithm(graph, request)
            else:
                # 默认使用Leiden算法
                communities = await self._leiden_algorithm(graph, request)

            # 后处理社区
            processed_communities = await self._post_process_communities(communities, request)

            # 如果启用分层社区发现
            if self.enable_hierarchical:
                processed_communities = await self._detect_hierarchical_communities(
                    processed_communities, graph, request
                )

            # 限制社区数量
            if request.max_communities and len(processed_communities) > request.max_communities:
                processed_communities = processed_communities[:request.max_communities]

            processing_time = (time.time() - start_time) * 1000

            # 计算统计信息
            total_entities_assigned = sum(len(community.entities) for community in processed_communities)

            structured_logger.info(
                "社区发现完成",
                extra={
                    "communities_count": len(processed_communities),
                    "total_entities_assigned": total_entities_assigned,
                    "processing_time_ms": round(processing_time, 2),
                }
            )

            return CommunityDetectionResponse(
                communities=processed_communities,
                total_communities=len(processed_communities),
                total_entities_assigned=total_entities_assigned,
                processing_time_ms=round(processing_time, 2),
                created_at=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            structured_logger.error(f"社区发现失败: {e}")
            raise Exception(f"Community detection failed: {e}")

    async def _build_graph(
        self,
        entities: List[EntityModel],
        relationships: List[RelationshipModel]
    ) -> Dict[str, Dict[str, Any]]:
        """构建图结构"""
        graph = {
            "nodes": {},
            "edges": [],
            "adjacency": defaultdict(set),
            "weights": defaultdict(lambda: defaultdict(float))
        }

        # 添加节点
        for entity in entities:
            graph["nodes"][entity.id] = {
                "entity": entity,
                "degree": 0,
                "weighted_degree": 0.0,
                "community": None
            }

        # 添加边
        for relationship in relationships:
            if (relationship.source_entity_id in graph["nodes"] and
                relationship.target_entity_id in graph["nodes"]):

                # 更新邻接表
                graph["adjacency"][relationship.source_entity_id].add(relationship.target_entity_id)
                graph["adjacency"][relationship.target_entity_id].add(relationship.source_entity_id)

                # 更新权重
                weight = relationship.confidence
                graph["weights"][relationship.source_entity_id][relationship.target_entity_id] = weight
                graph["weights"][relationship.target_entity_id][relationship.source_entity_id] = weight

                # 更新度数
                graph["nodes"][relationship.source_entity_id]["degree"] += 1
                graph["nodes"][relationship.target_entity_id]["degree"] += 1
                graph["nodes"][relationship.source_entity_id]["weighted_degree"] += weight
                graph["nodes"][relationship.target_entity_id]["weighted_degree"] += weight

                graph["edges"].append(relationship)

        return graph

    async def _leiden_algorithm(
        self,
        graph: Dict[str, Dict[str, Any]],
        request: CommunityDetectionRequest
    ) -> List[CommunityModel]:
        """Leiden社区发现算法"""
        # 初始化每个节点为独立社区
        communities = {node_id: i for i, node_id in enumerate(graph["nodes"].keys())}

        improved = True
        iteration = 0
        max_iterations = 100

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # 随机节点顺序
            node_order = list(graph["nodes"].keys())
            import random
            random.shuffle(node_order)

            for node_id in node_order:
                current_community = communities[node_id]
                neighbor_communities = self._get_neighbor_communities(node_id, communities, graph)

                if not neighbor_communities:
                    continue

                # 计算移动到邻居社区的增益
                best_community = current_community
                best_gain = 0.0

                for neighbor_community in neighbor_communities:
                    if neighbor_community == current_community:
                        continue

                    gain = self._calculate_modularity_gain(
                        node_id, current_community, neighbor_community, communities, graph, request
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_community = neighbor_community

                # 移动到最佳社区
                if best_community != current_community:
                    communities[node_id] = best_community
                    improved = True

        # 聚合社区
        community_groups = defaultdict(list)
        for node_id, community_id in communities.items():
            community_groups[community_id].append(node_id)

        # 构建社区模型
        community_models = []
        for community_id, entity_ids in community_groups.items():
            if len(entity_ids) >= request.min_community_size:
                community = await self._create_community_model(
                    community_id, entity_ids, graph, request
                )
                community_models.append(community)

        return community_models

    async def _louvain_algorithm(
        self,
        graph: Dict[str, Dict[str, Any]],
        request: CommunityDetectionRequest
    ) -> List[CommunityModel]:
        """Louvain社区发现算法"""
        # 简化实现，实际Louvain算法更复杂
        return await self._leiden_algorithm(graph, request)

    async def _infomap_algorithm(
        self,
        graph: Dict[str, Dict[str, Any]],
        request: CommunityDetectionRequest
    ) -> List[CommunityModel]:
        """Infomap社区发现算法"""
        # 简化实现，基于连通分量
        visited = set()
        communities = []

        for node_id in graph["nodes"]:
            if node_id in visited:
                continue

            # 深度优先搜索找连通分量
            component = []
            stack = [node_id]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)
                component.append(current)

                for neighbor in graph["adjacency"][current]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            if len(component) >= request.min_community_size:
                community = await self._create_community_model(
                    len(communities), component, graph, request
                )
                communities.append(community)

        return communities

    def _get_neighbor_communities(
        self,
        node_id: str,
        communities: Dict[str, int],
        graph: Dict[str, Dict[str, Any]]
    ) -> Set[int]:
        """获取节点的邻居社区"""
        neighbor_communities = set()
        for neighbor_id in graph["adjacency"][node_id]:
            neighbor_communities.add(communities[neighbor_id])
        return neighbor_communities

    def _calculate_modularity_gain(
        self,
        node_id: str,
        current_community: int,
        target_community: int,
        communities: Dict[str, int],
        graph: Dict[str, Dict[str, Any]],
        request: CommunityDetectionRequest
    ) -> float:
        """计算模块度增益"""
        # 简化的模块度计算
        resolution = request.resolution or self.default_resolution

        # 计算内部边权重
        internal_weight = 0.0
        for neighbor_id in graph["adjacency"][node_id]:
            if communities[neighbor_id] == target_community:
                internal_weight += graph["weights"][node_id][neighbor_id]

        # 计算总权重
        total_weight = sum(graph["weights"][node_id].values())
        target_community_weight = sum(
            graph["nodes"][nid]["weighted_degree"]
            for nid, cid in communities.items()
            if cid == target_community
        )

        # 模块度增益
        graph_total_weight = sum(
            graph["nodes"][nid]["weighted_degree"]
            for nid in graph["nodes"]
        )

        if graph_total_weight == 0:
            return 0.0

        gain = (internal_weight -
                resolution * total_weight * target_community_weight / (2 * graph_total_weight))

        return gain

    async def _create_community_model(
        self,
        community_id: int,
        entity_ids: List[str],
        graph: Dict[str, Dict[str, Any]],
        request: CommunityDetectionRequest
    ) -> CommunityModel:
        """创建社区模型"""
        # 获取社区中的实体
        community_entities = [graph["nodes"][eid]["entity"] for eid in entity_ids]

        # 计算社区特征
        entity_types = Counter(entity.type for entity in community_entities)
        dominant_type = entity_types.most_common(1)[0][0] if entity_types else EntityType.ORGANIZATION

        # 生成社区名称
        community_name = await self._generate_community_name(community_entities, dominant_type)

        # 生成社区描述
        community_description = await self._generate_community_description(community_entities)

        # 计算社区属性
        properties = {
            "entity_count": len(community_entities),
            "dominant_entity_type": dominant_type.value,
            "entity_types": {et.value: count for et, count in entity_types.items()},
            "average_confidence": sum(e.confidence for e in community_entities) / len(community_entities),
            "algorithm": request.algorithm,
            "resolution": request.resolution or self.default_resolution,
        }

        return CommunityModel(
            id=f"community_{community_id}_{int(time.time())}",
            name=community_name,
            description=community_description,
            entities=entity_ids,
            properties=properties,
            level=0,
            size=len(community_entities),
            created_at=datetime.utcnow().isoformat(),
        )

    async def _generate_community_name(
        self,
        entities: List[EntityModel],
        dominant_type: EntityType
    ) -> str:
        """生成社区名称"""
        if not entities:
            return "Empty Community"

        # 基于主导类型和实体名称生成
        type_names = {
            EntityType.PERSON: "People",
            EntityType.ORGANIZATION: "Organizations",
            EntityType.LOCATION: "Locations",
            EntityType.EVENT: "Events",
            EntityType.PRODUCT: "Products",
        }

        base_name = type_names.get(dominant_type, "Entities")

        # 如果实体数量少，可以使用实体名称
        if len(entities) <= 3:
            entity_names = [e.name for e in entities[:3]]
            return f"{base_name}: {', '.join(entity_names)}"

        return f"{base_name} Community ({len(entities)} entities)"

    async def _generate_community_description(self, entities: List[EntityModel]) -> str:
        """生成社区描述"""
        if not entities:
            return ""

        entity_types = Counter(entity.type for entity in entities)
        total_confidence = sum(e.confidence for e in entities)
        avg_confidence = total_confidence / len(entities)

        description_parts = []
        description_parts.append(f"Community with {len(entities)} entities")

        if entity_types:
            type_info = ", ".join([f"{count} {et.value}" for et, count in entity_types.most_common(3)])
            description_parts.append(f"including {type_info}")

        description_parts.append(f"with average confidence {avg_confidence:.2f}")

        return ". ".join(description_parts) + "."

    async def _post_process_communities(
        self,
        communities: List[CommunityModel],
        request: CommunityDetectionRequest
    ) -> List[CommunityModel]:
        """后处理社区"""
        # 过滤小社区
        filtered_communities = [
            community for community in communities
            if community.size >= request.min_community_size
        ]

        # 按大小排序
        filtered_communities.sort(key=lambda x: x.size, reverse=True)

        return filtered_communities

    async def _detect_hierarchical_communities(
        self,
        communities: List[CommunityModel],
        graph: Dict[str, Dict[str, Any]],
        request: CommunityDetectionRequest
    ) -> List[CommunityModel]:
        """检测分层社区"""
        # 简化实现：为每个社区分配层级
        for i, community in enumerate(communities):
            if community.size > 20:
                community.level = 0  # 高级社区
            elif community.size > 10:
                community.level = 1  # 中级社区
            else:
                community.level = 2  # 基础社区

        return communities

    async def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """更新实体属性"""
        return True

    async def get_entity(self, entity_id: str) -> Optional[EntityModel]:
        """获取实体"""
        return None

    async def get_entities_by_type(self, entity_type: EntityType, limit: int = 100) -> List[EntityModel]:
        """按类型获取实体"""
        return []

    async def search_entities(self, query: str, entity_types: List[EntityType] = None, limit: int = 100) -> List[EntityModel]:
        """搜索实体"""
        return []

    async def get_relationships(self, entity_id: str, relationship_type: str = None) -> List[RelationshipModel]:
        """获取实体的关系"""
        return []

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {}

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "provider": "CommunityDetector",
            "default_algorithm": self.default_algorithm,
            "min_community_size": self.min_community_size,
        }

    async def cleanup(self) -> None:
        """清理资源"""
        pass