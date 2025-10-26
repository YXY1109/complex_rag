"""
GraphRAG配置模块

提供GraphRAG组件的配置管理和默认设置。
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .interfaces.entity_interface import EntityType


class GraphRAGMode(str, Enum):
    """GraphRAG模式"""
    GENERAL = "general"
    LIGHT = "light"


class ProcessingLevel(str, Enum):
    """处理级别"""
    BASIC = "basic"        # 基础实体抽取
    ENHANCED = "enhanced"  # 增强处理（包含关系抽取）
    FULL = "full"         # 完整处理（包含社区发现）


class CommunityAlgorithm(str, Enum):
    """社区发现算法"""
    LEIDEN = "leiden"
    LOUVAIN = "louvain"
    INFOMAP = "infomap"


@dataclass
class EntityExtractionConfig:
    """实体抽取配置"""
    confidence_threshold: float = 0.7
    min_entity_length: int = 2
    max_entities: int = 100
    max_extraction_rounds: int = 3
    enable_relationship_extraction: bool = True
    entity_types: List[EntityType] = field(default_factory=lambda: [
        EntityType.PERSON,
        EntityType.ORGANIZATION,
        EntityType.LOCATION,
        EntityType.DATE,
        EntityType.MONEY,
    ])


@dataclass
class EntityResolutionConfig:
    """实体解析配置"""
    similarity_threshold: float = 0.8
    resolution_strategy: str = "name_similarity"
    enable_semantic_similarity: bool = True
    enable_type_matching: bool = True
    enable_property_matching: bool = True
    max_candidates: int = 10


@dataclass
class CommunityDetectionConfig:
    """社区发现配置"""
    algorithm: CommunityAlgorithm = CommunityAlgorithm.LEIDEN
    resolution: Optional[float] = None
    min_community_size: int = 3
    max_communities: Optional[int] = None
    enable_hierarchical: bool = True
    community_similarity_threshold: float = 0.3


@dataclass
class SearchConfig:
    """搜索配置"""
    default_search_type: str = "hybrid"
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    default_limit: int = 10
    default_threshold: float = 0.7
    enable_explanation: bool = True


@dataclass
class GraphConfig:
    """图配置"""
    default_node_type: str = "entity"
    default_edge_weight: float = 1.0
    enable_node_properties: bool = True
    enable_edge_properties: bool = True
    max_path_depth: int = 5


@dataclass
class GraphRAGServiceConfig:
    """GraphRAG服务配置"""
    mode: GraphRAGMode = GraphRAGMode.GENERAL
    processing_level: ProcessingLevel = ProcessingLevel.ENHANCED
    enable_caching: bool = True
    enable_parallel_processing: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 秒

    # 子配置
    entity_extraction: EntityExtractionConfig = field(default_factory=EntityExtractionConfig)
    entity_resolution: EntityResolutionConfig = field(default_factory=EntityResolutionConfig)
    community_detection: CommunityDetectionConfig = field(default_factory=CommunityDetectionConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)


class ConfigDefaults:
    """配置默认值"""

    # General模式默认配置
    GENERAL_CONFIG = {
        "mode": GraphRAGMode.GENERAL,
        "processing_level": ProcessingLevel.FULL,
        "entity_extraction": {
            "confidence_threshold": 0.5,
            "max_extraction_rounds": 3,
            "enable_relationship_extraction": True,
            "max_entities": 100,
        },
        "entity_resolution": {
            "similarity_threshold": 0.8,
            "enable_semantic_similarity": True,
            "enable_type_matching": True,
        },
        "community_detection": {
            "algorithm": CommunityAlgorithm.LEIDEN,
            "min_community_size": 3,
            "enable_hierarchical": True,
        },
    }

    # Light模式默认配置
    LIGHT_CONFIG = {
        "mode": GraphRAGMode.LIGHT,
        "processing_level": ProcessingLevel.BASIC,
        "entity_extraction": {
            "confidence_threshold": 0.6,
            "max_entities": 50,
            "enable_relationship_extraction": False,
        },
        "entity_resolution": {
            "similarity_threshold": 0.7,
            "enable_semantic_similarity": False,
        },
        "community_detection": {
            "algorithm": CommunityAlgorithm.INFOMAP,
            "min_community_size": 5,
        },
    }

    # 开发环境配置
    DEVELOPMENT_CONFIG = {
        "enable_caching": False,
        "enable_parallel_processing": False,
        "entity_extraction": {
            "max_entities": 20,
        },
        "community_detection": {
            "max_communities": 10,
        },
    }

    # 生产环境配置
    PRODUCTION_CONFIG = {
        "enable_caching": True,
        "enable_parallel_processing": True,
        "cache_size": 10000,
        "cache_ttl": 7200,
        "entity_extraction": {
            "max_entities": 200,
        },
    }


class ConfigManager:
    """配置管理器"""

    @staticmethod
    def create_config(
        mode: GraphRAGMode = GraphRAGMode.GENERAL,
        environment: str = "production",
        custom_overrides: Optional[Dict[str, Any]] = None
    ) -> GraphRAGServiceConfig:
        """
        创建GraphRAG配置

        Args:
            mode: GraphRAG模式
            environment: 环境类型 (development, production)
            custom_overrides: 自定义覆盖配置

        Returns:
            GraphRAGServiceConfig: 配置实例
        """
        # 选择基础配置
        if mode == GraphRAGMode.GENERAL:
            base_config = ConfigDefaults.GENERAL_CONFIG.copy()
        elif mode == GraphRAGMode.LIGHT:
            base_config = ConfigDefaults.LIGHT_CONFIG.copy()
        else:
            raise ValueError(f"不支持的GraphRAG模式: {mode}")

        # 应用环境配置
        if environment == "development":
            env_config = ConfigDefaults.DEVELOPMENT_CONFIG.copy()
        elif environment == "production":
            env_config = ConfigDefaults.PRODUCTION_CONFIG.copy()
        else:
            env_config = {}

        # 合并配置
        config_dict = ConfigManager._merge_configs(base_config, env_config)

        # 应用自定义覆盖
        if custom_overrides:
            config_dict = ConfigManager._merge_configs(config_dict, custom_overrides)

        # 转换为配置对象
        return ConfigManager._dict_to_config(config_dict)

    @staticmethod
    def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> GraphRAGServiceConfig:
        """将字典转换为配置对象"""
        # 转换子配置
        entity_extraction_dict = config_dict.get("entity_extraction", {})
        entity_resolution_dict = config_dict.get("entity_resolution", {})
        community_detection_dict = config_dict.get("community_detection", {})
        search_dict = config_dict.get("search", {})
        graph_dict = config_dict.get("graph", {})

        return GraphRAGServiceConfig(
            mode=GraphRAGMode(config_dict.get("mode", "general")),
            processing_level=ProcessingLevel(config_dict.get("processing_level", "enhanced")),
            enable_caching=config_dict.get("enable_caching", True),
            enable_parallel_processing=config_dict.get("enable_parallel_processing", True),
            cache_size=config_dict.get("cache_size", 1000),
            cache_ttl=config_dict.get("cache_ttl", 3600),
            entity_extraction=EntityExtractionConfig(
                confidence_threshold=entity_extraction_dict.get("confidence_threshold", 0.7),
                min_entity_length=entity_extraction_dict.get("min_entity_length", 2),
                max_entities=entity_extraction_dict.get("max_entities", 100),
                max_extraction_rounds=entity_extraction_dict.get("max_extraction_rounds", 3),
                enable_relationship_extraction=entity_extraction_dict.get("enable_relationship_extraction", True),
                entity_types=[EntityType(t) for t in entity_extraction_dict.get("entity_types", [])] or [
                    EntityType.PERSON,
                    EntityType.ORGANIZATION,
                    EntityType.LOCATION,
                    EntityType.DATE,
                    EntityType.MONEY,
                ],
            ),
            entity_resolution=EntityResolutionConfig(
                similarity_threshold=entity_resolution_dict.get("similarity_threshold", 0.8),
                resolution_strategy=entity_resolution_dict.get("resolution_strategy", "name_similarity"),
                enable_semantic_similarity=entity_resolution_dict.get("enable_semantic_similarity", True),
                enable_type_matching=entity_resolution_dict.get("enable_type_matching", True),
                enable_property_matching=entity_resolution_dict.get("enable_property_matching", True),
                max_candidates=entity_resolution_dict.get("max_candidates", 10),
            ),
            community_detection=CommunityDetectionConfig(
                algorithm=CommunityAlgorithm(community_detection_dict.get("algorithm", "leiden")),
                resolution=community_detection_dict.get("resolution"),
                min_community_size=community_detection_dict.get("min_community_size", 3),
                max_communities=community_detection_dict.get("max_communities"),
                enable_hierarchical=community_detection_dict.get("enable_hierarchical", True),
                community_similarity_threshold=community_detection_dict.get("community_similarity_threshold", 0.3),
            ),
            search=SearchConfig(
                default_search_type=search_dict.get("default_search_type", "hybrid"),
                semantic_weight=search_dict.get("semantic_weight", 0.7),
                keyword_weight=search_dict.get("keyword_weight", 0.3),
                default_limit=search_dict.get("default_limit", 10),
                default_threshold=search_dict.get("default_threshold", 0.7),
                enable_explanation=search_dict.get("enable_explanation", True),
            ),
            graph=GraphConfig(
                default_node_type=graph_dict.get("default_node_type", "entity"),
                default_edge_weight=graph_dict.get("default_edge_weight", 1.0),
                enable_node_properties=graph_dict.get("enable_node_properties", True),
                enable_edge_properties=graph_dict.get("enable_edge_properties", True),
                max_path_depth=graph_dict.get("max_path_depth", 5),
            ),
        )

    @staticmethod
    def get_default_config() -> GraphRAGServiceConfig:
        """获取默认配置"""
        return ConfigManager.create_config()

    @staticmethod
    def get_light_config() -> GraphRAGServiceConfig:
        """获取Light模式配置"""
        return ConfigManager.create_config(mode=GraphRAGMode.LIGHT)

    @staticmethod
    def get_development_config() -> GraphRAGServiceConfig:
        """获取开发环境配置"""
        return ConfigManager.create_config(environment="development")

    @staticmethod
    def validate_config(config: GraphRAGServiceConfig) -> List[str]:
        """
        验证配置

        Args:
            config: 配置实例

        Returns:
            List[str]: 验证错误列表
        """
        errors = []

        # 验证基本参数
        if not 0.0 <= config.entity_extraction.confidence_threshold <= 1.0:
            errors.append("entity_extraction.confidence_threshold 必须在 [0.0, 1.0] 范围内")

        if config.entity_extraction.max_entities <= 0:
            errors.append("entity_extraction.max_entities 必须大于 0")

        if not 0.0 <= config.entity_resolution.similarity_threshold <= 1.0:
            errors.append("entity_resolution.similarity_threshold 必须在 [0.0, 1.0] 范围内")

        if config.community_detection.min_community_size < 2:
            errors.append("community_detection.min_community_size 必须至少为 2")

        if config.search.default_limit <= 0:
            errors.append("search.default_limit 必须大于 0")

        if not 0.0 <= config.search.default_threshold <= 1.0:
            errors.append("search.default_threshold 必须在 [0.0, 1.0] 范围内")

        # 验证权重
        total_weight = config.search.semantic_weight + config.search.keyword_weight
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"search.semantic_weight + search.keyword_weight 必须约等于 1.0，当前为 {total_weight}")

        return errors