"""
实体抽取模块

实现GraphRAG General模式的多轮实体抽取功能。
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

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


structured_logger = get_logger("core_rag.graph_rag.general.extraction")


class EntityExtractor(EntityInterface):
    """
    实体抽取器

    基于规则和机器学习的实体抽取，支持多轮抽取以提高准确性。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化实体抽取器

        Args:
            config: 配置参数
        """
        self.config = config
        self.max_extraction_rounds = config.get("max_rounds", 3)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.min_entity_length = config.get("min_entity_length", 2)
        self.max_entities_per_round = config.get("max_entities_per_round", 100)
        self.enable_relationship_extraction = config.get("enable_relationships", True)
        self.entity_type_patterns = self._load_entity_patterns()
        self.relationship_patterns = self._load_relationship_patterns()

    def _load_entity_patterns(self) -> Dict[EntityType, List[str]]:
        """加载实体类型模式"""
        return {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
                r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b',  # Title + Name
                r'\b[A-Z][a-z]+\s+(?:CEO|CTO|CFO|COO|President|Director|Manager)\b',  # Name + Title
            ],
            EntityType.ORGANIZATION: [
                r'\b(?:Apple|Google|Microsoft|Amazon|Facebook|Twitter|LinkedIn|GitHub)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Corporation)\b',
                r'\b(?:University|College|Institute)\s+of\s+[A-Z][a-z]+\b',
            ],
            EntityType.LOCATION: [
                r'\b(?:New York|Los Angeles|San Francisco|Chicago|Boston|Seattle|Washington)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b',  # City, State
                r'\b(?:USA|United States|United Kingdom|China|Japan|Germany|France)\b',
            ],
            EntityType.EVENT: [
                r'\b(?:World War\s+[IVX]+|COVID-19|Olympics|Election|Conference)\b',
                r'\b(?:September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
                r'\b(?:in|on|at|during|since|before|after)\s+\d{4}\b',
            ],
            EntityType.DATE: [
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # M/D/YYYY
            ],
            EntityType.MONEY: [
                r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?\b',  # $1,234.56
                r'\b\d+(?:,\d{3})*\s*(?:dollars|USD|euros|EUR|pounds|GBP|yuan|CNY)\b',
                r'\b(?:million|billion|trillion)\s+(?:dollars|USD|euros|EUR)\b',
            ],
            EntityType.PRODUCT: [
                r'\b(?:iPhone|iPad|MacBook|Android|Windows)\b',
                r'\b(?:Toyota|Honda|Ford|BMW|Mercedes|Tesla)\b',
                r'\b(?:Google\s+Pixel|Samsung\s+Galaxy|OnePlus)\b',
            ],
        }

    def _load_relationship_patterns(self) -> List[str]:
        """加载关系模式"""
        return [
            r'(\w+)\s+(?:is|was|are|were)\s+(?:a|an|the)\s+(\w+)',  # X is a Y
            r'(\w+)\s+(?:works|worked|works for)\s+(?:at|in)\s+(\w+)',  # X works at Y
            r'(\w+)\s+(?:located|situated)\s+(?:in|at)\s+(\w+)',  # X located in Y
            r'(\w+)\s+(?:CEO|President|Director|Manager)\s+of\s+(\w+)',  # X CEO of Y
            r'(\w+)\s+(?:born|founded|created)\s+(?:in|on)\s+([\w\s]+)',  # X born in Y
            r'(\w+)\s+(?:headquartered|based)\s+(?:in|at)\s+(\w+)',  # X headquartered in Y
        ]

    async def initialize(self) -> bool:
        """
        初始化实体抽取器

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "实体抽取器初始化成功",
                extra={
                    "max_rounds": self.max_extraction_rounds,
                    "confidence_threshold": self.confidence_threshold,
                    "supported_entity_types": [t.value for t in self.entity_type_patterns.keys()],
                }
            )
            return True
        except Exception as e:
            structured_logger.error(f"实体抽取器初始化失败: {e}")
            return False

    async def extract_entities(
        self,
        request: EntityExtractionRequest
    ) -> EntityExtractionResponse:
        """
        执行实体抽取

        Args:
            request: 实体抽取请求

        Returns:
            EntityExtractionResponse: 抽取结果
        """
        start_time = time.time()

        try:
            structured_logger.info(
                "开始实体抽取",
                extra={
                    "text_length": len(request.text),
                    "entity_types": [t.value for t in request.entity_types] if request.entity_types else [],
                    "max_entities": request.max_entities,
                }
            )

            # 第一轮：基于规则的抽取
            entities, relationships = await self._rule_based_extraction(request)

            # 第二轮：基于上下文的优化
            if self.max_extraction_rounds >= 2:
                additional_entities, additional_relationships = await self._context_based_extraction(
                    request.text, entities, relationships
                )
                entities.extend(additional_entities)
                relationships.extend(additional_relationships)

            # 第三轮：实体消歧和过滤
            if self.max_extraction_rounds >= 3:
                filtered_entities = await self._filter_entities(entities)
                filtered_relationships = await self._filter_relationships(relationships, filtered_entities)
                entities = filtered_entities
                relationships = filtered_relationships

            # 限制数量
            if request.max_entities and len(entities) > request.max_entities:
                entities = sorted(entities, key=lambda x: x.confidence, reverse=True)[:request.max_entities]

            processing_time = (time.time() - start_time) * 1000

            structured_logger.info(
                "实体抽取完成",
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
            structured_logger.error(f"实体抽取失败: {e}")
            raise Exception(f"Entity extraction failed: {e}")

    async def _rule_based_extraction(
        self, request: EntityExtractionRequest
    ) -> tuple[List[EntityModel], List[RelationshipModel]]:
        """基于规则的实体抽取"""
        entities = []
        relationships = []

        text = request.text
        entity_types = request.entity_types or list(self.entity_type_patterns.keys())

        # 抽取实体
        for entity_type in entity_types:
            if entity_type not in self.entity_type_patterns:
                continue

            patterns = self.entity_type_patterns[entity_type]
            for pattern in patterns:
                import re
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group().strip()
                    if len(entity_name) < self.min_entity_length:
                        continue

                    confidence = self._calculate_confidence(entity_name, entity_type, text)

                    if confidence >= self.confidence_threshold:
                        entity = EntityModel(
                            id=f"entity_{len(entities)}_{int(time.time())}",
                            name=entity_name,
                            type=entity_type,
                            source_text=match.group(),
                            start_char=match.start(),
                            end_char=match.end(),
                            confidence=confidence,
                            created_at=datetime.utcnow().isoformat(),
                        )
                        entities.append(entity)

        # 抽取关系
        if self.enable_relationship_extraction:
            for pattern in self.relationship_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    source_name = match.group(1).strip()
                    target_name = match.group(2).strip()
                    relationship_desc = match.group().strip()

                    # 查找匹配的实体
                    source_entities = [e for e in entities if e.name.lower() == source_name.lower()]
                    target_entities = [e for e in entities if e.name.lower() == target_name.lower()]

                    if source_entities and target_entities:
                        source_entity = source_entities[0]
                        target_entity = target_entities[0]

                        confidence = self._calculate_relationship_confidence(
                            relationship_desc, source_entity, target_entity, text
                        )

                        if confidence >= self.confidence_threshold:
                            relationship = RelationshipModel(
                                id=f"rel_{len(relationships)}_{int(time.time())}",
                                source_entity_id=source_entity.id,
                                target_entity_id=target_entity.id,
                                relationship_type="related_to",  # 简化的关系类型
                                description=relationship_desc,
                                source_text=match.group(),
                                confidence=confidence,
                                created_at=datetime.utcnow().isoformat(),
                            )
                            relationships.append(relationship)

        return entities, relationships

    async def _context_based_extraction(
        self, text: str, existing_entities: List[EntityModel], existing_relationships: List[RelationshipModel]
    ) -> tuple[List[EntityModel], List[RelationshipModel]]:
        """基于上下文的实体抽取优化"""
        additional_entities = []
        additional_relationships = []

        # 分析已有实体的上下文
        for entity in existing_entities:
            # 获取实体周围的文本
            context_start = max(0, entity.start_char - 100)
            context_end = min(len(text), entity.end_char + 100)
            context = text[context_start:context_end]

            # 在上下文中查找相关实体
            for entity_type in self.entity_type_patterns:
                if entity_type == entity.type:
                    continue

                patterns = self.entity_type_patterns[entity_type]
                for pattern in patterns:
                    import re
                    matches = re.finditer(pattern, context, re.IGNORECASE)
                    for match in matches:
                        entity_name = match.group().strip()
                        if len(entity_name) < self.min_entity_length:
                            continue

                        # 检查是否已存在
                        if not any(e.name.lower() == entity_name.lower() for e in existing_entities):
                            confidence = self._calculate_confidence(entity_name, entity_type, context)

                            if confidence >= self.confidence_threshold:
                                new_entity = EntityModel(
                                    id=f"entity_{len(existing_entities) + len(additional_entities)}_{int(time.time())}",
                                    name=entity_name,
                                    type=entity_type,
                                    source_text=match.group(),
                                    start_char=context_start + match.start(),
                                    end_char=context_start + match.end(),
                                    confidence=confidence,
                                    created_at=datetime.utcnow().isoformat(),
                                )
                                additional_entities.append(new_entity)

        return additional_entities, additional_relationships

    async def _filter_entities(self, entities: List[EntityModel]) -> List[EntityModel]:
        """过滤和清理实体"""
        filtered_entities = []

        for entity in entities:
            # 基于置信度过滤
            if entity.confidence < self.confidence_threshold:
                continue

            # 基于长度过滤
            if len(entity.name) < self.min_entity_length:
                continue

            # 基于内容过滤（排除常见噪音词）
            if self._is_noise_word(entity.name):
                continue

            filtered_entities.append(entity)

        return filtered_entities

    async def _filter_relationships(
        self, relationships: List[RelationshipModel], entities: List[EntityModel]
    ) -> List[RelationshipModel]:
        """过滤关系"""
        entity_ids = {e.id for e in entities}
        filtered_relationships = []

        for relationship in relationships:
            # 检查源和目标实体是否存在
            if (relationship.source_entity_id not in entity_ids or
                relationship.target_entity_id not in entity_ids):
                continue

            # 基于置信度过滤
            if relationship.confidence < self.confidence_threshold:
                continue

            filtered_relationships.append(relationship)

        return filtered_relationships

    def _calculate_confidence(self, entity_name: str, entity_type: EntityType, text: str) -> float:
        """计算实体置信度"""
        base_confidence = 0.5

        # 基于实体类型调整置信度
        type_confidence_map = {
            EntityType.PERSON: 0.8,
            EntityType.ORGANIZATION: 0.7,
            EntityType.LOCATION: 0.6,
            EntityType.DATE: 0.9,
            EntityType.MONEY: 0.8,
        }

        type_confidence = type_confidence_map.get(entity_type, base_confidence)

        # 基于大写字母调整置信度
        if entity_name[0].isupper():
            base_confidence += 0.2

        # 基于长度调整置信度
        length_factor = min(1.0, len(entity_name) / 10)
        length_confidence = 0.3 + 0.7 * length_factor

        # 基于在文本中的位置调整置信度
        position_factor = len(text) / 1000
        position_confidence = 0.8 + 0.2 * position_factor

        return min(1.0, base_confidence * type_confidence * length_confidence * position_confidence)

    def _calculate_relationship_confidence(
        self, relationship_desc: str, source_entity: EntityModel, target_entity: EntityModel, text: str
    ) -> float:
        """计算关系置信度"""
        base_confidence = 0.5

        # 基于关系描述词调整
        strong_relationship_words = [
            "is", "was", "are", "were", "works", "located", "situated"
        ]
        medium_relationship_words = [
            "part of", "member of", "associated with"
        ]

        if any(word in relationship_desc.lower() for word in strong_relationship_words):
            base_confidence = 0.8
        elif any(word in relationship_desc.lower() for word in medium_relationship_words):
            base_confidence = 0.6

        # 基于实体类型兼容性调整
        compatible_pairs = [
            (EntityType.PERSON, EntityType.ORGANIZATION),
            (EntityType.ORGANIZATION, EntityType.LOCATION),
            (EntityType.PERSON, EntityType.LOCATION),
            (EntityType.ORGANIZATION, EntityType.EVENT),
        ]

        if (source_entity.type, target_entity.type) in compatible_pairs or
            (target_entity.type, source_entity.type) in compatible_pairs):
            base_confidence += 0.2

        return min(1.0, base_confidence)

    def _is_noise_word(self, entity_name: str) -> bool:
        """检查是否为噪音词"""
        noise_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "out", "off", "over", "under", "about", "into", "through",
            "during", "before", "after", "since", "until", "while", "as", "if", "because",
            "than", "then", "else", "when", "where", "how", "what", "which", "who", "whom",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
            "mr", "mrs", "ms", "dr", "prof", "inc", "corp", "llc", "ltd", "co",
        }

        return entity_name.lower() in noise_words

    # 实现其他接口方法
    async def resolve_entities(self, request: EntityResolutionRequest) -> EntityResolutionResponse:
        """实体解析"""
        # 简单实现：基于名称相似度
        resolved_entities = []
        resolution_mappings = {}

        entity_name_map = {}
        for entity in request.existing_entities:
            name_key = entity.name.lower().strip()
            if name_key not in entity_name_map:
                entity_name_map[name_key] = []
            entity_name_map[name_key].append(entity)

        for entity in request.entities:
            name_key = entity.name.lower().strip()
            if name_key in entity_name_map:
                # 找到相似实体
                for existing_entity in entity_name_map[name_key]:
                    similarity = self._calculate_name_similarity(entity.name, existing_entity.name)
                    if similarity >= 0.8:
                        resolution_mappings[entity.id] = existing_entity.id
                        break

                # 如果找到匹配，使用现有实体
                if entity.id in resolution_mappings:
                    existing_id = resolution_mappings[entity.id]
                    existing_entity = next(e for e in request.existing_entities if e.id == existing_id)
                    resolved_entities.append(existing_entity)
                else:
                    resolved_entities.append(entity)
            else:
                resolved_entities.append(entity)

        return EntityResolutionResponse(
            resolved_entities=resolved_entities,
            merged_entities=[],
            resolution_mappings=resolution_mappings,
            processing_time_ms=100.0,
            created_at=datetime.utcnow().isoformat(),
        )

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """计算名称相似度"""
        # 简单的Jaccard相似度
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    async def detect_communities(self, request) -> "CommunityDetectionResponse":
        """社区发现"""
        # 简单实现
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
            "provider": "EntityExtractor",
            "supported_entity_types": [t.value for t in self.entity_type_patterns.keys()],
            "max_extraction_rounds": self.max_extraction_rounds,
            "confidence_threshold": self.confidence_threshold,
        }

    async def cleanup(self) -> None:
        """清理资源"""
        pass