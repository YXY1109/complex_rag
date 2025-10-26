"""
轻量级实体抽取模块

实现GraphRAG Light模式的快速实体抽取功能。
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

from ..interfaces.entity_interface import (
    EntityInterface,
    EntityType,
    EntityModel,
    RelationshipModel,
    EntityExtractionRequest,
    EntityExtractionResponse,
    EntityResolutionRequest,
    EntityResolutionResponse
)
from ...infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.graph_rag.light.extraction")


class LightEntityExtractor(EntityInterface):
    """
    轻量级实体抽取器

    基于规则和简单模式的快速实体抽取，优化性能和资源使用。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化轻量级实体抽取器

        Args:
            config: 配置参数
        """
        self.config = config
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.min_entity_length = config.get("min_entity_length", 2)
        self.max_entities = config.get("max_entities", 50)
        self.enable_relationship_extraction = config.get("enable_relationships", False)  # Light模式默认关闭关系抽取
        self.entity_patterns = self._load_light_entity_patterns()
        self.common_entities = self._load_common_entities()

    def _load_light_entity_patterns(self) -> Dict[EntityType, List[str]]:
        """加载轻量级实体模式（简化版）"""
        return {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # 简单姓名模式
            ],
            EntityType.ORGANIZATION: [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company)\b',  # 公司后缀
            ],
            EntityType.LOCATION: [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b',  # 城市, 州
            ],
            EntityType.DATE: [
                r'\b\d{4}-\d{2}-\d{2}\b',  # ISO日期格式
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # M/D/YYYY格式
            ],
            EntityType.MONEY: [
                r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?\b',  # $金额
            ],
        }

    def _load_common_entities(self) -> Dict[EntityType, List[str]]:
        """加载常见实体列表（用于快速匹配）"""
        return {
            EntityType.ORGANIZATION: [
                "Apple", "Google", "Microsoft", "Amazon", "Facebook", "Twitter",
                "LinkedIn", "GitHub", "IBM", "Oracle", "Intel", "NVIDIA"
            ],
            EntityType.LOCATION: [
                "New York", "Los Angeles", "San Francisco", "Chicago", "Boston",
                "Seattle", "Washington", "London", "Paris", "Tokyo", "Beijing"
            ],
            EntityType.PRODUCT: [
                "iPhone", "iPad", "MacBook", "Windows", "Android", "Chrome"
            ],
        }

    async def initialize(self) -> bool:
        """
        初始化轻量级实体抽取器

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "轻量级实体抽取器初始化成功",
                extra={
                    "confidence_threshold": self.confidence_threshold,
                    "max_entities": self.max_entities,
                    "relationship_extraction": self.enable_relationship_extraction,
                }
            )
            return True
        except Exception as e:
            structured_logger.error(f"轻量级实体抽取器初始化失败: {e}")
            return False

    async def extract_entities(
        self,
        request: EntityExtractionRequest
    ) -> EntityExtractionResponse:
        """
        执行轻量级实体抽取

        Args:
            request: 实体抽取请求

        Returns:
            EntityExtractionResponse: 抽取结果
        """
        start_time = time.time()

        try:
            structured_logger.info(
                "开始轻量级实体抽取",
                extra={
                    "text_length": len(request.text),
                    "entity_types": [t.value for t in request.entity_types] if request.entity_types else [],
                }
            )

            # 快速文本预处理
            normalized_text = self._preprocess_text(request.text)

            # 第一轮：快速模式匹配
            entities = await self._fast_pattern_extraction(normalized_text, request)

            # 第二轮：常见实体匹配
            additional_entities = await self._common_entity_extraction(normalized_text, request)
            entities.extend(additional_entities)

            # 去重和过滤
            entities = await self._deduplicate_entities(entities)

            # 限制数量
            if len(entities) > self.max_entities:
                entities = sorted(entities, key=lambda x: x.confidence, reverse=True)[:self.max_entities]

            # Light模式下简化关系抽取
            relationships = []
            if self.enable_relationship_extraction:
                relationships = await self._simple_relationship_extraction(entities, normalized_text)

            processing_time = (time.time() - start_time) * 1000

            structured_logger.info(
                "轻量级实体抽取完成",
                extra={
                    "entities_count": len(entities),
                    "relationships_count": len(relationships),
                    "processing_time_ms": round(processing_time, 2),
                }
            )

            return EntityExtractionResponse(
                entities=entities,
                relationships=relationships,
                total_entities=len(entities),
                total_relationships=len(relationships),
                processing_time_ms=round(processing_time, 2),
                created_at=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            structured_logger.error(f"轻量级实体抽取失败: {e}")
            raise Exception(f"Light entity extraction failed: {e}")

    def _preprocess_text(self, text: str) -> str:
        """快速文本预处理"""
        # 简单的标准化处理
        # 保留句子边界，去除多余空格
        normalized = re.sub(r'\s+', ' ', text.strip())
        return normalized

    async def _fast_pattern_extraction(
        self,
        text: str,
        request: EntityExtractionRequest
    ) -> List[EntityModel]:
        """快速模式匹配抽取"""
        entities = []
        entity_types = request.entity_types or list(self.entity_patterns.keys())

        # 并行处理不同实体类型
        extraction_tasks = []
        for entity_type in entity_types:
            if entity_type in self.entity_patterns:
                task = self._extract_entities_by_type(text, entity_type)
                extraction_tasks.append(task)

        # 等待所有抽取任务完成
        extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # 合并结果
        for result in extraction_results:
            if isinstance(result, list):
                entities.extend(result)
            elif isinstance(result, Exception):
                structured_logger.warning(f"实体抽取任务异常: {result}")

        return entities

    async def _extract_entities_by_type(
        self,
        text: str,
        entity_type: EntityType
    ) -> List[EntityModel]:
        """按类型抽取实体"""
        entities = []
        patterns = self.entity_patterns[entity_type]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_name = match.group().strip()

                if len(entity_name) < self.min_entity_length:
                    continue

                # 计算置信度（简化版）
                confidence = self._calculate_light_confidence(entity_name, entity_type, text)

                if confidence >= self.confidence_threshold:
                    entity = EntityModel(
                        id=f"light_entity_{len(entities)}_{int(time.time())}",
                        name=entity_name,
                        type=entity_type,
                        source_text=match.group(),
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=confidence,
                        created_at=datetime.utcnow().isoformat(),
                    )
                    entities.append(entity)

        return entities

    async def _common_entity_extraction(
        self,
        text: str,
        request: EntityExtractionRequest
    ) -> List[EntityModel]:
        """常见实体匹配"""
        entities = []
        entity_types = request.entity_types or list(self.common_entities.keys())

        for entity_type in entity_types:
            if entity_type not in self.common_entities:
                continue

            for common_entity in self.common_entities[entity_type]:
                # 查找所有出现位置
                start_pos = 0
                while True:
                    pos = text.lower().find(common_entity.lower(), start_pos)
                    if pos == -1:
                        break

                    # 检查边界
                    if self._is_word_boundary(text, pos, len(common_entity)):
                        confidence = 0.9  # 常见实体高置信度

                        entity = EntityModel(
                            id=f"common_entity_{len(entities)}_{int(time.time())}",
                            name=common_entity,
                            type=entity_type,
                            source_text=text[pos:pos+len(common_entity)],
                            start_char=pos,
                            end_char=pos+len(common_entity),
                            confidence=confidence,
                            created_at=datetime.utcnow().isoformat(),
                        )
                        entities.append(entity)

                    start_pos = pos + 1

        return entities

    def _is_word_boundary(self, text: str, start: int, length: int) -> bool:
        """检查是否为词边界"""
        # 检查开始边界
        if start > 0 and text[start-1].isalnum():
            return False

        # 检查结束边界
        end = start + length
        if end < len(text) and text[end].isalnum():
            return False

        return True

    async def _deduplicate_entities(self, entities: List[EntityModel]) -> List[EntityModel]:
        """去重实体"""
        seen = set()
        deduplicated = []

        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
            else:
                # 如果已存在，选择置信度更高的
                for i, existing_entity in enumerate(deduplicated):
                    if (existing_entity.name.lower() == entity.name.lower() and
                        existing_entity.type == entity.type):
                        if entity.confidence > existing_entity.confidence:
                            deduplicated[i] = entity
                        break

        return deduplicated

    async def _simple_relationship_extraction(
        self,
        entities: List[EntityModel],
        text: str
    ) -> List[RelationshipModel]:
        """简单关系抽取"""
        relationships = []

        # 简化的关系模式
        simple_patterns = [
            (r'(\w+)\s+works\s+at\s+(\w+)', 'works_at'),
            (r'(\w+)\s+located\s+in\s+(\w+)', 'located_in'),
            (r'(\w+)\s+CEO\s+of\s+(\w+)', 'ceo_of'),
        ]

        for pattern, rel_type in simple_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                source_name = match.group(1).strip()
                target_name = match.group(2).strip()

                # 查找匹配的实体
                source_entity = next(
                    (e for e in entities if e.name.lower() == source_name.lower()), None
                )
                target_entity = next(
                    (e for e in entities if e.name.lower() == target_name.lower()), None
                )

                if source_entity and target_entity:
                    relationship = RelationshipModel(
                        id=f"light_rel_{len(relationships)}_{int(time.time())}",
                        source_entity_id=source_entity.id,
                        target_entity_id=target_entity.id,
                        relationship_type=rel_type,
                        description=match.group().strip(),
                        source_text=match.group(),
                        confidence=0.7,  # 固定置信度
                        created_at=datetime.utcnow().isoformat(),
                    )
                    relationships.append(relationship)

        return relationships

    def _calculate_light_confidence(
        self,
        entity_name: str,
        entity_type: EntityType,
        text: str
    ) -> float:
        """计算轻量级置信度"""
        base_confidence = 0.6

        # 基于实体类型调整
        type_confidence_map = {
            EntityType.PERSON: 0.7,
            EntityType.ORGANIZATION: 0.8,
            EntityType.LOCATION: 0.7,
            EntityType.DATE: 0.9,
            EntityType.MONEY: 0.8,
        }

        type_confidence = type_confidence_map.get(entity_type, base_confidence)

        # 基于首字母大写调整
        if entity_name and entity_name[0].isupper():
            base_confidence += 0.1

        # 基于常见实体调整
        if entity_type in self.common_entities:
            if entity_name in self.common_entities[entity_type]:
                base_confidence += 0.2

        return min(1.0, base_confidence * type_confidence)

    async def resolve_entities(self, request: EntityResolutionRequest) -> EntityResolutionResponse:
        """实体解析（简化版）"""
        # 简单的基于名称的实体解析
        resolved_entities = []
        resolution_mappings = {}
        seen_names = {}

        for entity in request.entities:
            name_key = entity.name.lower().strip()

            if name_key in seen_names:
                # 合并到已存在的实体
                existing_entity = seen_names[name_key]
                resolution_mappings[entity.id] = existing_entity.id

                # 更新置信度为更高的值
                if entity.confidence > existing_entity.confidence:
                    existing_entity.confidence = entity.confidence
            else:
                seen_names[name_key] = entity
                resolved_entities.append(entity)

        return EntityResolutionResponse(
            resolved_entities=resolved_entities,
            merged_entities=[],
            resolution_mappings=resolution_mappings,
            processing_time_ms=50.0,
            created_at=datetime.utcnow().isoformat(),
        )

    async def detect_communities(self, request) -> "CommunityDetectionResponse":
        """社区发现（简化版）"""
        # Light模式下使用简单的聚类
        communities = []
        entity_type_groups = {}

        # 按实体类型分组
        for entity in request.entities:
            if entity.type not in entity_type_groups:
                entity_type_groups[entity.type] = []
            entity_type_groups[entity.type].append(entity.id)

        # 为每个类型创建社区
        for entity_type, entity_ids in entity_type_groups.items():
            if len(entity_ids) >= 3:  # 最小社区大小
                from ..interfaces.entity_interface import CommunityModel
                community = CommunityModel(
                    id=f"light_community_{len(communities)}_{int(time.time())}",
                    name=f"{entity_type.value} Community",
                    description=f"Community of {len(entity_ids)} {entity_type.value} entities",
                    entities=entity_ids,
                    properties={"entity_type": entity_type.value, "size": len(entity_ids)},
                    level=0,
                    size=len(entity_ids),
                    created_at=datetime.utcnow().isoformat(),
                )
                communities.append(community)

        from ..interfaces.entity_interface import CommunityDetectionResponse
        return CommunityDetectionResponse(
            communities=communities,
            total_communities=len(communities),
            total_entities_assigned=sum(len(c.entities) for c in communities),
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
            "provider": "LightEntityExtractor",
            "confidence_threshold": self.confidence_threshold,
            "max_entities": self.max_entities,
            "mode": "light",
        }

    async def cleanup(self) -> None:
        """清理资源"""
        pass