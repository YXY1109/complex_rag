"""
实体抽取接口

定义实体抽取和关系解析的抽象接口。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class EntityType(str, Enum):
    """实体类型"""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "GPE"
    EVENT = "EVENT"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    PRODUCT = "PRODUCT"
    FACILITY = "FACILITY"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    NORP = "NORP"
    QUANTITY = "QUANTITY"
    ORDINAL = "ORDINAL"
    CARDINAL = "CARDINAL"


class EntityModel(BaseModel):
    """实体模型"""
    id: str
    name: str
    type: EntityType
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_text: str
    start_char: int
    end_char: int
    confidence: float = Field(default=1.0)
    created_at: str


class RelationshipModel(BaseModel):
    """关系模型"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_text: str
    confidence: float = Field(default=1.0)
    created_at: str


class EntityExtractionRequest(BaseModel):
    """实体抽取请求"""
    text: str
    entity_types: List[EntityType] = Field(default_factory=list)
    max_entities: Optional[int] = None
    confidence_threshold: float = Field(default=0.5)
    context: Optional[str] = None
    user_id: Optional[str] = None


class EntityExtractionResponse(BaseModel):
    """实体抽取响应"""
    entities: List[EntityModel]
    relationships: List[RelationshipModel]
    total_entities: int
    total_relationships: int
    processing_time_ms: float
    created_at: str


class EntityResolutionRequest(BaseModel):
    """实体解析请求"""
    entities: List[EntityModel]
    existing_entities: List[EntityModel] = Field(default_factory=list)
    resolution_strategy: str = Field(default="name_similarity")  # name_similarity, type_priority
    similarity_threshold: float = Field(default=0.8)
    user_id: Optional[str] = None


class EntityResolutionResponse(BaseModel):
    """实体解析响应"""
    resolved_entities: List[EntityModel]
    merged_entities: List[EntityModel] = Field(default_factory=list)
    resolution_mappings: Dict[str, str] = Field(default_factory=dict)  # old_id -> new_id
    processing_time_ms: float
    created_at: str


class CommunityDetectionRequest(BaseModel):
    """社区发现请求"""
    entities: List[EntityModel]
    relationships: List[RelationshipModel]
    algorithm: str = Field(default="leiden")  # leiden, louvain, infomap
    resolution: Optional[float] = None
    min_community_size: int = Field(default=2)
    max_communities: Optional[int] = None
    user_id: Optional[str] = None


class CommunityModel(BaseModel):
    """社区模型"""
    id: str
    name: str
    description: Optional[str] = None
    entities: List[str]  # entity IDs
    properties: Dict[str, Any] = Field(default_factory=dict)
    level: int = Field(default=0)
    size: int
    created_at: str


class CommunityDetectionResponse(BaseModel):
    """社区发现响应"""
    communities: List[CommunityModel]
    total_communities: int
    total_entities_assigned: int
    processing_time_ms: float
    created_at: str


class EntityInterface(ABC):
    """
    实体抽取接口抽象类

    定义实体抽取、关系解析、社区发现等功能的接口。
    """

    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化实体抽取器

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    async def extract_entities(
        self,
        request: EntityExtractionRequest
    ) -> EntityExtractionResponse:
        """
        抽取实体和关系

        Args:
            request: 实体抽取请求

        Returns:
            EntityExtractionResponse: 抽取结果
        """
        pass

    @abstractmethod
    async def resolve_entities(
        self,
        request: EntityResolutionRequest
    ) -> EntityDetectionResponse:
        """
        解析实体（去重、合并）

        Args:
            request: 实体解析请求

        Returns:
            EntityResolutionResponse: 解析结果
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def update_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """
        更新实体属性

        Args:
            entity_id: 实体ID
            properties: 属性字典

        Returns:
            bool: 更新是否成功
        """
        pass

    @abstractmethod
    async def get_entity(
        self,
        entity_id: str
    ) -> Optional[EntityModel]:
        """
        获取实体

        Args:
            entity_id: 实体ID

        Returns:
            Optional[EntityModel]: 实体模型
        """
        pass

    @abstractmethod
    async def get_entities_by_type(
        self,
        entity_type: EntityType,
        limit: int = 100
    ) -> List[EntityModel]:
        """
        按类型获取实体

        Args:
            entity_type: 实体类型
            limit: 限制数量

        Returns:
            List[EntityModel]: 实体列表
        """
        pass

    @abstractmethod
    async def search_entities(
        self,
        query: str,
        entity_types: List[EntityType] = None,
        limit: int = 100
    ) -> List[EntityModel]:
        """
        搜索实体

        Args:
            query: 搜索查询
            entity_types: 实体类型过滤
            limit: 限制数量

        Returns:
            List[EntityModel]: 实体列表
        """
        pass

    @abstractmethod
    async def get_relationships(
        self,
        entity_id: str,
        relationship_type: str = None
    ) -> List[RelationshipModel]:
        """
        获取实体的关系

        Args:
            entity_id: 实体ID
            relationship_type: 关系类型过滤

        Returns:
            List[RelationshipModel]: 关系列表
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        获取实体统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict[str, Any]: 健康状态
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """清理资源"""
        pass


class EntityException(Exception):
    """实体抽取异常基类"""
    pass


class ExtractionError(EntityException):
    """抽取异常"""
    pass


class ResolutionError(EntityException):
    """解析异常"""
    pass


class CommunityDetectionError(EntityException):
    """社区发现异常"""
    pass


class InvalidEntityTypeError(EntityException):
    """无效实体类型异常"""
    pass