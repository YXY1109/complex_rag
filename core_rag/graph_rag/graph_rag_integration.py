"""
GraphRAG集成层

将GraphRAG功能集成到现有的RAG架构中。
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .entity_extraction_service import EntityExtractionService, GraphRAGMode
from .interfaces.entity_interface import (
    EntityExtractionRequest,
    EntityResolutionRequest,
    CommunityDetectionRequest,
    EntityModel,
    RelationshipModel,
    CommunityModel,
    EntityType,
)
from .interfaces.search_interface import (
    SearchInterface,
    SearchRequest,
    SearchResponse,
    SearchResult,
    CommunitySearchRequest,
    CommunitySearchResponse,
    PathSearchRequest,
    PathSearchResponse,
)
from .interfaces.graph_interface import (
    GraphInterface,
    NodeModel,
    EdgeModel,
    GraphQuery,
    GraphPath,
    GraphNodeType,
)
from ..infrastructure.monitoring.loguru_logger import get_logger


structured_logger = get_logger("core_rag.graph_rag.graph_rag_integration")


class ProcessingLevel(str, Enum):
    """处理级别"""
    BASIC = "basic"        # 基础实体抽取
    ENHANCED = "enhanced"  # 增强处理（包含关系抽取）
    FULL = "full"         # 完整处理（包含社区发现）


@dataclass
class GraphRAGConfig:
    """GraphRAG配置"""
    mode: GraphRAGMode = GraphRAGMode.GENERAL
    processing_level: ProcessingLevel = ProcessingLevel.ENHANCED
    enable_caching: bool = True
    enable_parallel_processing: bool = True
    confidence_threshold: float = 0.7
    similarity_threshold: float = 0.8
    min_community_size: int = 3
    max_entities: int = 100
    entity_types: List[EntityType] = None

    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = [
                EntityType.PERSON,
                EntityType.ORGANIZATION,
                EntityType.LOCATION,
                EntityType.DATE,
                EntityType.MONEY,
            ]


class GraphRAGProcessor:
    """
    GraphRAG处理器

    集成GraphRAG功能到RAG流水线中。
    """

    def __init__(self, config: GraphRAGConfig):
        """
        初始化GraphRAG处理器

        Args:
            config: GraphRAG配置
        """
        self.config = config
        self.entity_service = None
        self.search_service = None
        self.graph_service = None
        self._initialized = False

    async def initialize(self) -> bool:
        """
        初始化GraphRAG处理器

        Returns:
            bool: 初始化是否成功
        """
        try:
            structured_logger.info(
                "初始化GraphRAG处理器",
                extra={
                    "mode": self.config.mode,
                    "processing_level": self.config.processing_level,
                    "confidence_threshold": self.config.confidence_threshold,
                }
            )

            # 初始化实体抽取服务
            entity_config = {
                "default_mode": self.config.mode,
                "enable_caching": self.config.enable_caching,
                "enable_parallel_processing": self.config.enable_parallel_processing,
                "confidence_threshold": self.config.confidence_threshold,
                "similarity_threshold": self.config.similarity_threshold,
                "min_community_size": self.config.min_community_size,
                "max_entities": self.config.max_entities,
            }

            self.entity_service = EntityExtractionService(entity_config)
            await self.entity_service.initialize()

            # TODO: 初始化搜索服务和图服务（这些在后续任务中实现）
            self.search_service = None
            self.graph_service = None

            self._initialized = True
            structured_logger.info("GraphRAG处理器初始化成功")
            return True

        except Exception as e:
            structured_logger.error(f"GraphRAG处理器初始化失败: {e}")
            return False

    async def process_document(
        self,
        text: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理文档，抽取实体和关系

        Args:
            text: 文档文本
            document_id: 文档ID
            metadata: 文档元数据

        Returns:
            Dict[str, Any]: 处理结果
        """
        if not self._initialized:
            raise RuntimeError("GraphRAG处理器未初始化")

        try:
            structured_logger.info(
                "开始处理文档",
                extra={
                    "document_id": document_id,
                    "text_length": len(text),
                    "processing_level": self.config.processing_level,
                }
            )

            # 创建实体抽取请求
            extraction_request = EntityExtractionRequest(
                text=text,
                entity_types=self.config.entity_types,
                confidence_threshold=self.config.confidence_threshold,
                max_entities=self.config.max_entities,
                context=document_id,
            )

            # 执行处理流水线
            if self.config.processing_level == ProcessingLevel.BASIC:
                result = await self._basic_processing(extraction_request)
            elif self.config.processing_level == ProcessingLevel.ENHANCED:
                result = await self._enhanced_processing(extraction_request)
            elif self.config.processing_level == ProcessingLevel.FULL:
                result = await self._full_processing(extraction_request)
            else:
                raise ValueError(f"不支持的处理级别: {self.config.processing_level}")

            # 添加文档信息
            result["document_id"] = document_id
            result["metadata"] = metadata or {}

            structured_logger.info(
                "文档处理完成",
                extra={
                    "document_id": document_id,
                    "entities_count": len(result.get("entities", [])),
                    "relationships_count": len(result.get("relationships", [])),
                    "communities_count": len(result.get("communities", [])),
                }
            )

            return result

        except Exception as e:
            structured_logger.error(f"文档处理失败: {e}")
            raise Exception(f"Document processing failed: {e}")

    async def _basic_processing(self, request: EntityExtractionRequest) -> Dict[str, Any]:
        """基础处理：仅实体抽取"""
        response = await self.entity_service.extract_entities(request, self.config.mode)

        return {
            "entities": response.entities,
            "relationships": response.relationships,
            "processing_stats": {
                "entities_count": response.total_entities,
                "relationships_count": response.total_relationships,
                "processing_time_ms": response.processing_time_ms,
                "level": "basic",
            },
        }

    async def _enhanced_processing(self, request: EntityExtractionRequest) -> Dict[str, Any]:
        """增强处理：实体抽取 + 实体解析"""
        # 创建解析请求
        resolution_request = EntityResolutionRequest(
            entities=request.entity_types,  # 这个会在实际处理中被替换
            existing_entities=[],
            resolution_strategy="name_similarity",
            similarity_threshold=self.config.similarity_threshold,
        )

        # 执行一站式处理
        result = await self.entity_service.extract_and_resolve(
            extraction_request=request,
            resolution_request=resolution_request,
            mode=self.config.mode
        )

        # 转换结果格式
        extraction_response = result["extraction"]
        resolution_response = result.get("resolution")

        entities = extraction_response.entities
        if resolution_response:
            entities = resolution_response.resolved_entities

        return {
            "entities": entities,
            "relationships": extraction_response.relationships,
            "merged_entities": resolution_response.merged_entities if resolution_response else [],
            "resolution_mappings": resolution_response.resolution_mappings if resolution_response else {},
            "processing_stats": {
                "entities_count": len(entities),
                "relationships_count": len(extraction_response.relationships),
                "merged_entities_count": len(resolution_response.merged_entities) if resolution_response else 0,
                "processing_time_ms": extraction_response.processing_time_ms,
                "level": "enhanced",
            },
        }

    async def _full_processing(self, request: EntityExtractionRequest) -> Dict[str, Any]:
        """完整处理：实体抽取 + 解析 + 社区发现"""
        # 创建解析请求
        resolution_request = EntityResolutionRequest(
            entities=request.entity_types,  # 这个会在实际处理中被替换
            existing_entities=[],
            resolution_strategy="name_similarity",
            similarity_threshold=self.config.similarity_threshold,
        )

        # 创建社区发现请求
        community_request = CommunityDetectionRequest(
            entities=request.entity_types,  # 这个会在实际处理中被替换
            relationships=[],
            algorithm="leiden",
            min_community_size=self.config.min_community_size,
        )

        # 执行一站式处理
        result = await self.entity_service.extract_and_resolve(
            extraction_request=request,
            resolution_request=resolution_request,
            community_request=community_request,
            mode=self.config.mode
        )

        # 转换结果格式
        extraction_response = result["extraction"]
        resolution_response = result.get("resolution")
        community_response = result.get("community")

        entities = extraction_response.entities
        if resolution_response:
            entities = resolution_response.resolved_entities

        return {
            "entities": entities,
            "relationships": extraction_response.relationships,
            "merged_entities": resolution_response.merged_entities if resolution_response else [],
            "resolution_mappings": resolution_response.resolution_mappings if resolution_response else {},
            "communities": community_response.communities if community_response else [],
            "processing_stats": {
                "entities_count": len(entities),
                "relationships_count": len(extraction_response.relationships),
                "merged_entities_count": len(resolution_response.merged_entities) if resolution_response else 0,
                "communities_count": len(community_response.communities) if community_response else 0,
                "processing_time_ms": extraction_response.processing_time_ms,
                "level": "full",
            },
        }

    async def search_entities(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 20
    ) -> List[EntityModel]:
        """
        搜索实体

        Args:
            query: 搜索查询
            entity_types: 实体类型过滤
            limit: 限制数量

        Returns:
            List[EntityModel]: 匹配的实体
        """
        if not self._initialized:
            raise RuntimeError("GraphRAG处理器未初始化")

        try:
            # 使用实体抽取服务的搜索功能
            entities = await self.entity_service.light_extractor.search_entities(
                query, entity_types, limit
            )

            return entities

        except Exception as e:
            structured_logger.error(f"实体搜索失败: {e}")
            return []

    async def get_entity_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None
    ) -> List[RelationshipModel]:
        """
        获取实体的关系

        Args:
            entity_id: 实体ID
            relationship_type: 关系类型过滤

        Returns:
            List[RelationshipModel]: 关系列表
        """
        if not self._initialized:
            raise RuntimeError("GraphRAG处理器未初始化")

        try:
            # 使用实体抽取服务的关系查询功能
            relationships = await self.entity_service.general_extractor.get_relationships(
                entity_id, relationship_type
            )

            return relationships

        except Exception as e:
            structured_logger.error(f"获取实体关系失败: {e}")
            return []

    async def process_query(
        self,
        query: str,
        context_entities: Optional[List[EntityModel]] = None,
        max_context_entities: int = 10
    ) -> Dict[str, Any]:
        """
        处理查询，提取查询中的实体

        Args:
            query: 用户查询
            context_entities: 上下文实体
            max_context_entities: 最大上下文实体数量

        Returns:
            Dict[str, Any]: 查询处理结果
        """
        if not self._initialized:
            raise RuntimeError("GraphRAG处理器未初始化")

        try:
            structured_logger.info(
                "开始处理查询",
                extra={
                    "query_length": len(query),
                    "context_entities_count": len(context_entities) if context_entities else 0,
                }
            )

            # 从查询中抽取实体
            extraction_request = EntityExtractionRequest(
                text=query,
                entity_types=self.config.entity_types,
                confidence_threshold=self.config.confidence_threshold * 0.8,  # 查询中使用较低的阈值
                max_entities=20,
            )

            extraction_response = await self.entity_service.extract_entities(
                extraction_request, GraphRAGMode.LIGHT  # 查询处理使用Light模式
            )

            # 与上下文实体合并
            all_entities = extraction_response.entities.copy()
            if context_entities:
                # 去重合并
                seen_names = {e.name.lower() for e in all_entities}
                for context_entity in context_entities[:max_context_entities]:
                    if context_entity.name.lower() not in seen_names:
                        all_entities.append(context_entity)
                        seen_names.add(context_entity.name.lower())

            structured_logger.info(
                "查询处理完成",
                extra={
                    "query_entities_count": len(extraction_response.entities),
                    "total_entities_count": len(all_entities),
                }
            )

            return {
                "query_entities": extraction_response.entities,
                "context_entities": context_entities or [],
                "all_entities": all_entities,
                "query_relationships": extraction_response.relationships,
                "processing_stats": {
                    "query_entities_count": len(extraction_response.entities),
                    "total_entities_count": len(all_entities),
                    "processing_time_ms": extraction_response.processing_time_ms,
                },
            }

        except Exception as e:
            structured_logger.error(f"查询处理失败: {e}")
            return {
                "query_entities": [],
                "context_entities": context_entities or [],
                "all_entities": context_entities or [],
                "query_relationships": [],
                "error": str(e),
            }

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._initialized:
            return {"error": "GraphRAG处理器未初始化"}

        try:
            entity_stats = await self.entity_service.get_statistics()

            return {
                "config": {
                    "mode": self.config.mode,
                    "processing_level": self.config.processing_level,
                    "confidence_threshold": self.config.confidence_threshold,
                    "similarity_threshold": self.config.similarity_threshold,
                    "min_community_size": self.config.min_community_size,
                    "max_entities": self.config.max_entities,
                },
                "entity_service_stats": entity_stats,
                "status": "initialized",
            }

        except Exception as e:
            structured_logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            entity_health = await self.entity_service.health_check()

            return {
                "status": "healthy" if entity_health.get("status") == "healthy" else "degraded",
                "initialized": self._initialized,
                "entity_service_health": entity_health,
                "config": {
                    "mode": self.config.mode,
                    "processing_level": self.config.processing_level,
                },
            }

        except Exception as e:
            structured_logger.error(f"健康检查失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "initialized": self._initialized,
            }

    async def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.entity_service:
                await self.entity_service.cleanup()

            # TODO: 清理其他服务

            self._initialized = False
            structured_logger.info("GraphRAG处理器清理完成")

        except Exception as e:
            structured_logger.error(f"GraphRAG处理器清理失败: {e}")


class GraphRAGFactory:
    """GraphRAG处理器工厂"""

    @staticmethod
    def create_processor(config: Dict[str, Any]) -> GraphRAGProcessor:
        """
        创建GraphRAG处理器

        Args:
            config: 配置字典

        Returns:
            GraphRAGProcessor: 处理器实例
        """
        # 转换配置
        graphrag_config = GraphRAGConfig(
            mode=GraphRAGMode(config.get("mode", "general")),
            processing_level=ProcessingLevel(config.get("processing_level", "enhanced")),
            enable_caching=config.get("enable_caching", True),
            enable_parallel_processing=config.get("enable_parallel_processing", True),
            confidence_threshold=config.get("confidence_threshold", 0.7),
            similarity_threshold=config.get("similarity_threshold", 0.8),
            min_community_size=config.get("min_community_size", 3),
            max_entities=config.get("max_entities", 100),
            entity_types=[EntityType(t) for t in config.get("entity_types", [])] or None,
        )

        return GraphRAGProcessor(graphrag_config)