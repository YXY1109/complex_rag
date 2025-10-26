"""
实体解析模块

实现GraphRAG General模式的实体解析和消歧功能。
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import asyncio

from ..interfaces.entity_interface import (
    EntityInterface,
    EntityModel,
    RelationshipModel,
    EntityResolutionRequest,
    EntityResolutionResponse,
    EntityType,
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.graph_rag.general.resolution")


class EntityResolver(EntityInterface):
    """
    实体解析器

    基于多种相似度计算策略的实体解析和消歧。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化实体解析器

        Args:
            config: 配置参数
        """
        self.config = config
        self.similarity_threshold = config.get("similarity_threshold", 0.8)
        self.enable_semantic_similarity = config.get("enable_semantic_similarity", True)
        self.enable_type_matching = config.get("enable_type_matching", True)
        self.enable_property_matching = config.get("enable_property_matching", True)
        self.max_candidates = config.get("max_candidates", 10)
        self.resolution_strategies = config.get("resolution_strategies", [
            "name_similarity",
            "type_priority",
            "property_similarity",
            "context_similarity"
        ])

    async def initialize(self) -> bool:
        """
        初始化实体解析器

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "实体解析器初始化成功",
                extra={
                    "similarity_threshold": self.similarity_threshold,
                    "resolution_strategies": self.resolution_strategies,
                }
            )
            return True
        except Exception as e:
            structured_logger.error(f"实体解析器初始化失败: {e}")
            return False

    async def extract_entities(
        self, request: "EntityExtractionRequest"
    ) -> "EntityExtractionResponse":
        """实体抽取（占位实现）"""
        return EntityExtractionResponse(
            entities=[],
            relationships=[],
            total_entities=0,
            total_relationships=0,
            processing_time_ms=0.0,
            created_at=datetime.utcnow().isoformat(),
        )

    async def resolve_entities(
        self,
        request: EntityResolutionRequest
    ) -> EntityResolutionResponse:
        """
        解析实体（去重、合并）

        Args:
            request: 实体解析请求

        Returns:
            EntityResolutionResponse: 解析结果
        """
        start_time = time.time()

        try:
            structured_logger.info(
                "开始实体解析",
                extra={
                    "input_entities_count": len(request.entities),
                    "existing_entities_count": len(request.existing_entities),
                    "resolution_strategy": request.resolution_strategy,
                    "similarity_threshold": request.similarity_threshold,
                }
            )

            # 构建候选实体映射
            candidate_groups = await self._build_candidate_groups(
                request.entities, request.existing_entities, request
            )

            # 解析每个候选组
            resolved_entities = []
            merged_entities = []
            resolution_mappings = {}

            for group_id, group in candidate_groups.items():
                if len(group) == 1:
                    # 单个实体，直接使用
                    entity = group[0]
                    resolved_entities.append(entity)
                else:
                    # 多个相似实体，需要合并
                    merged_entity, merge_mappings = await self._merge_entity_group(
                        group, request
                    )
                    resolved_entities.append(merged_entity)
                    merged_entities.append(merged_entity)
                    resolution_mappings.update(merge_mappings)

            processing_time = (time.time() - start_time) * 1000

            structured_logger.info(
                "实体解析完成",
                extra={
                    "resolved_entities_count": len(resolved_entities),
                    "merged_entities_count": len(merged_entities),
                    "resolution_mappings_count": len(resolution_mappings),
                    "processing_time_ms": round(processing_time, 2),
                }
            )

            return EntityResolutionResponse(
                resolved_entities=resolved_entities,
                merged_entities=merged_entities,
                resolution_mappings=resolution_mappings,
                processing_time_ms=round(processing_time, 2),
                created_at=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            structured_logger.error(f"实体解析失败: {e}")
            raise Exception(f"Entity resolution failed: {e}")

    async def _build_candidate_groups(
        self,
        entities: List[EntityModel],
        existing_entities: List[EntityModel],
        request: EntityResolutionRequest
    ) -> Dict[str, List[EntityModel]]:
        """构建候选实体组"""
        all_entities = entities + existing_entities
        processed = set()
        groups = {}
        group_id_counter = 0

        for i, entity in enumerate(all_entities):
            if entity.id in processed:
                continue

            # 创建新组
            group_id = f"group_{group_id_counter}"
            group_id_counter += 1
            groups[group_id] = [entity]
            processed.add(entity.id)

            # 查找相似实体
            for j, other_entity in enumerate(all_entities):
                if (other_entity.id in processed or
                    other_entity.id == entity.id):
                    continue

                similarity = await self._calculate_entity_similarity(
                    entity, other_entity, request
                )

                if similarity >= request.similarity_threshold:
                    groups[group_id].append(other_entity)
                    processed.add(other_entity.id)

        return groups

    async def _merge_entity_group(
        self,
        entities: List[EntityModel],
        request: EntityResolutionRequest
    ) -> Tuple[EntityModel, Dict[str, str]]:
        """合并实体组"""
        # 选择最佳代表实体
        representative_entity = await self._select_representative_entity(entities)

        # 合并属性
        merged_properties = await self._merge_entity_properties(entities)

        # 合并关系
        merged_relationships = await self._merge_entity_relationships(entities, request)

        # 创建合并后的实体
        merged_entity = EntityModel(
            id=representative_entity.id,
            name=representative_entity.name,
            type=representative_entity.type,
            description=representative_entity.description,
            properties=merged_properties,
            source_text=representative_entity.source_text,
            start_char=representative_entity.start_char,
            end_char=representative_entity.end_char,
            confidence=max(e.confidence for e in entities),
            created_at=datetime.utcnow().isoformat(),
        )

        # 创建映射关系
        resolution_mappings = {}
        for entity in entities:
            if entity.id != representative_entity.id:
                resolution_mappings[entity.id] = representative_entity.id

        return merged_entity, resolution_mappings

    async def _select_representative_entity(
        self, entities: List[EntityModel]
    ) -> EntityModel:
        """选择代表实体"""
        # 基于置信度和名称完整性选择
        def entity_score(entity: EntityModel) -> float:
            score = entity.confidence

            # 名称长度和完整性奖励
            if len(entity.name) >= 3 and entity.name[0].isupper():
                score += 0.2

            # 描述存在奖励
            if entity.description:
                score += 0.1

            # 属性数量奖励
            score += len(entity.properties) * 0.01

            return score

        return max(entities, key=entity_score)

    async def _merge_entity_properties(
        self, entities: List[EntityModel]
    ) -> Dict[str, Any]:
        """合并实体属性"""
        merged_properties = {}
        property_sources = defaultdict(list)

        # 收集所有属性
        for entity in entities:
            for key, value in entity.properties.items():
                if key not in merged_properties:
                    merged_properties[key] = value
                property_sources[key].append((value, entity.confidence))

        # 处理冲突属性
        for key, sources in property_sources.items():
            if len(sources) > 1:
                # 选择置信度最高的值
                merged_properties[key] = max(sources, key=lambda x: x[1])[0]
                merged_properties[f"{key}_merged_from"] = len(sources)

        # 添加合并元数据
        merged_properties["merge_count"] = len(entities)
        merged_properties["merge_confidence"] = sum(e.confidence for e in entities) / len(entities)

        return merged_properties

    async def _merge_entity_relationships(
        self, entities: List[EntityModel], request: EntityResolutionRequest
    ) -> List[RelationshipModel]:
        """合并实体关系"""
        # 这里简化实现，实际场景中需要更复杂的关系合并逻辑
        return []

    async def _calculate_entity_similarity(
        self,
        entity1: EntityModel,
        entity2: EntityModel,
        request: EntityResolutionRequest
    ) -> float:
        """计算实体相似度"""
        similarity_scores = []

        # 名称相似度
        name_similarity = self._calculate_name_similarity(entity1.name, entity2.name)
        similarity_scores.append(("name", name_similarity, 0.4))

        # 类型匹配
        if self.enable_type_matching:
            type_similarity = 1.0 if entity1.type == entity2.type else 0.0
            similarity_scores.append(("type", type_similarity, 0.2))

        # 属性相似度
        if self.enable_property_matching:
            property_similarity = self._calculate_property_similarity(
                entity1.properties, entity2.properties
            )
            similarity_scores.append(("property", property_similarity, 0.2))

        # 描述相似度
        if entity1.description and entity2.description:
            desc_similarity = self._calculate_text_similarity(
                entity1.description, entity2.description
            )
            similarity_scores.append(("description", desc_similarity, 0.2))

        # 加权平均
        total_weight = sum(weight for _, _, weight in similarity_scores)
        if total_weight == 0:
            return 0.0

        weighted_similarity = sum(score * weight for _, score, weight in similarity_scores)
        return weighted_similarity / total_weight

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """计算名称相似度"""
        # 标准化名称
        name1_norm = name1.lower().strip()
        name2_norm = name2.lower().strip()

        if name1_norm == name2_norm:
            return 1.0

        # Jaccard相似度
        words1 = set(name1_norm.split())
        words2 = set(name2_norm.split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        jaccard_similarity = len(intersection) / len(union)

        # 编辑距离相似度
        edit_distance_similarity = self._calculate_edit_distance_similarity(name1_norm, name2_norm)

        # 取最高值
        return max(jaccard_similarity, edit_distance_similarity)

    def _calculate_edit_distance_similarity(self, s1: str, s2: str) -> float:
        """计算编辑距离相似度"""
        len1, len2 = len(s1), len(s2)
        if len1 == 0:
            return 0.0 if len2 > 0 else 1.0
        if len2 == 0:
            return 0.0

        # 动态规划计算编辑距离
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # 删除
                        dp[i][j-1] + 1,    # 插入
                        dp[i-1][j-1] + 1   # 替换
                    )

        max_len = max(len1, len2)
        return 1.0 - (dp[len1][len2] / max_len)

    def _calculate_property_similarity(
        self, props1: Dict[str, Any], props2: Dict[str, Any]
    ) -> float:
        """计算属性相似度"""
        if not props1 and not props2:
            return 1.0
        if not props1 or not props2:
            return 0.0

        common_keys = set(props1.keys()) & set(props2.keys())
        all_keys = set(props1.keys()) | set(props2.keys())

        if not all_keys:
            return 0.0

        # 计算共同属性值的相似度
        similarity_sum = 0.0
        for key in common_keys:
            val1, val2 = props1[key], props2[key]
            if isinstance(val1, str) and isinstance(val2, str):
                similarity_sum += self._calculate_text_similarity(val1, val2)
            elif val1 == val2:
                similarity_sum += 1.0
            else:
                similarity_sum += 0.0

        # 键的相似度
        key_similarity = len(common_keys) / len(all_keys)

        # 值的相似度
        value_similarity = similarity_sum / len(common_keys) if common_keys else 0.0

        return (key_similarity + value_similarity) / 2.0

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        if not text1 or not text2:
            return 0.0

        # 简单的词汇重叠相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    async def detect_communities(self, request) -> "CommunityDetectionResponse":
        """社区发现（占位实现）"""
        return CommunityDetectionResponse(
            communities=[],
            total_communities=0,
            total_entities_assigned=0,
            processing_time_ms=100.0,
            created_at=datetime.utcnow().isoformat(),
        )

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
            "provider": "EntityResolver",
            "similarity_threshold": self.similarity_threshold,
            "resolution_strategies": self.resolution_strategies,
        }

    async def cleanup(self) -> None:
        """清理资源"""
        pass