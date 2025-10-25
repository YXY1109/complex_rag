"""
GraphRAG Configuration (RAGFlow Reference)

This module contains configuration for RAGFlow GraphRAG capabilities.
"""

from pydantic import Field

from ..settings import BaseConfig


class GraphRAGConfig(BaseConfig):
    """GraphRAG configuration based on RAGFlow best practices."""

    # GraphRAG Mode
    graphrag_mode: str = Field(default="light", env="GRAPHRAG_MODE")  # light, general
    graphrag_enabled: bool = Field(default=True, env="GRAPHRAG_ENABLED")

    # Entity Extraction
    entity_extraction_enabled: bool = Field(default=True, env="GRAPHRAG_ENTITY_EXTRACTION_ENABLED")
    entity_model: str = Field(default="gpt-4-turbo-preview", env="GRAPHRAG_ENTITY_MODEL")
    entity_types: list[str] = Field(
        default=["PERSON", "ORG", "GPE", "EVENT", "DATE", "PRODUCT", "WORK_OF_ART", "LAW", "LANGUAGE", "MONEY"],
        env="GRAPHRAG_ENTITY_TYPES"
    )
    entity_confidence_threshold: float = Field(default=0.7, env="GRAPHRAG_ENTITY_CONFIDENCE_THRESHOLD")
    entity_max_entities_per_doc: int = Field(default=100, env="GRAPHRAG_MAX_ENTITIES_PER_DOC")

    # Relationship Extraction
    relationship_extraction_enabled: bool = Field(default=True, env="GRAPHRAG_RELATIONSHIP_EXTRACTION_ENABLED")
    relationship_model: str = Field(default="gpt-4-turbo-preview", env="GRAPHRAG_RELATIONSHIP_MODEL")
    relationship_types: list[str] = Field(
        default=["PART_OF", "WORKS_FOR", "LOCATED_IN", "CREATED_BY", "MARRIED_TO", "RELATED_TO"],
        env="GRAPHRAG_RELATIONSHIP_TYPES"
    )
    relationship_confidence_threshold: float = Field(default=0.6, env="GRAPHRAG_RELATIONSHIP_CONFIDENCE_THRESHOLD")
    relationship_max_relationships_per_doc: int = Field(default=200, env="GRAPHRAG_MAX_RELATIONSHIPS_PER_DOC")

    # Light Mode Configuration (Single-pass extraction)
    light_mode_extraction_model: str = Field(default="gpt-3.5-turbo", env="GRAPHRAG_LIGHT_MODEL")
    light_mode_max_tokens: int = Field(default=4000, env="GRAPHRAG_LIGHT_MAX_TOKENS")
    light_mode_temperature: float = Field(default=0.1, env="GRAPHRAG_LIGHT_TEMPERATURE")
    light_mode_batch_size: int = Field(default=10, env="GRAPHRAG_LIGHT_BATCH_SIZE")

    # General Mode Configuration (Multi-pass extraction)
    general_mode_entity_model: str = Field(default="gpt-4-turbo-preview", env="GRAPHRAG_GENERAL_ENTITY_MODEL")
    general_mode_resolution_model: str = Field(default="gpt-4-turbo-preview", env="GRAPHRAG_GENERAL_RESOLUTION_MODEL")
    general_mode_community_model: str = Field(default="gpt-4-turbo-preview", env="GRAPHRAG_GENERAL_COMMUNITY_MODEL")
    general_mode_max_iterations: int = Field(default=5, env="GRAPHRAG_GENERAL_MAX_ITERATIONS")
    general_mode_overlap_threshold: float = Field(default=0.8, env="GRAPHRAG_GENERAL_OVERLAP_THRESHOLD")

    # Community Detection
    community_detection_enabled: bool = Field(default=True, env="GRAPHRAG_COMMUNITY_DETECTION_ENABLED")
    community_algorithm: str = Field(default="louvain", env="GRAPHRAG_COMMUNITY_ALGORITHM")  # louvain, leiden
    community_resolution: float = Field(default=1.0, env="GRAPHRAG_COMMUNITY_RESOLUTION")
    community_min_size: int = Field(default=3, env="GRAPHRAG_COMMUNITY_MIN_SIZE")
    community_max_size: int = Field(default=100, env="GRAPHRAG_COMMUNITY_MAX_SIZE")

    # Graph Storage
    graph_storage_backend: str = Field(default="neo4j", env="GRAPHRAG_GRAPH_STORAGE_BACKEND")  # neo4j, networkx, arangodb
    graph_storage_config: dict = Field(
        default={
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password",
                "database": "neo4j"
            }
        },
        env="GRAPHRAG_GRAPH_STORAGE_CONFIG"
    )

    # Graph Indexing
    graph_indexing_enabled: bool = Field(default=True, env="GRAPHRAG_GRAPH_INDEXING_ENABLED")
    entity_index_fields: list[str] = Field(
        default=["name", "type", "description", "source_documents"],
        env="GRAPHRAG_ENTITY_INDEX_FIELDS"
    )
    relationship_index_fields: list[str] = Field(
        default=["source", "target", "type", "description", "weight"],
        env="GRAPHRAG_RELATIONSHIP_INDEX_FIELDS"
    )

    # Graph Search
    graph_search_enabled: bool = Field(default=True, env="GRAPHRAG_GRAPH_SEARCH_ENABLED")
    graph_search_algorithm: str = Field(default="personalized_pagerank", env="GRAPHRAG_GRAPH_SEARCH_ALGORITHM")
    graph_search_max_hops: int = Field(default=3, env="GRAPHRAG_GRAPH_SEARCH_MAX_HOPS")
    graph_search_max_results: int = Field(default=20, env="GRAPHRAG_GRAPH_SEARCH_MAX_RESULTS")
    graph_search_damping_factor: float = Field(default=0.85, env="GRAPHRAG_GRAPH_SEARCH_DAMPING_FACTOR")

    # Graph Visualization
    graph_visualization_enabled: bool = Field(default=False, env="GRAPHRAG_GRAPH_VISUALIZATION_ENABLED")
    graph_visualization_max_nodes: int = Field(default=100, env="GRAPHRAG_GRAPH_VISUALIZATION_MAX_NODES")
    graph_visualization_layout: str = Field(default="spring", env="GRAPHRAG_GRAPH_VISUALIZATION_LAYOUT")

    # Processing Settings
    max_concurrent_extractions: int = Field(default=5, env="GRAPHRAG_MAX_CONCURRENT_EXTRACTIONS")
    extraction_timeout: int = Field(default=300, env="GRAPHRAG_EXTRACTION_TIMEOUT")  # 5 minutes
    processing_batch_size: int = Field(default=10, env="GRAPHRAG_PROCESSING_BATCH_SIZE")
    memory_limit_mb: int = Field(default=8192, env="GRAPHRAG_MEMORY_LIMIT_MB")

    # Quality Control
    quality_check_enabled: bool = Field(default=True, env="GRAPHRAG_QUALITY_CHECK_ENABLED")
    min_entity_occurrences: int = Field(default=2, env="GRAPHRAG_MIN_ENTITY_OCCURRENCES")
    min_relationship_occurrences: int = Field(default=1, env="GRAPHRAG_MIN_RELATIONSHIP_OCCURRENCES")
    duplicate_entity_threshold: float = Field(default=0.9, env="GRAPHRAG_DUPLICATE_ENTITY_THRESHOLD")

    # Cache Settings
    cache_enabled: bool = Field(default=True, env="GRAPHRAG_CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, env="GRAPHRAG_CACHE_TTL")  # 1 hour
    cache_graph_ttl: int = Field(default=86400, env="GRAPHRAG_CACHE_GRAPH_TTL")  # 24 hours
    cache_dir: str = Field(default="cache/graphrag", env="GRAPHRAG_CACHE_DIR")

    # Update Strategy
    graph_update_strategy: str = Field(default="incremental", env="GRAPHRAG_GRAPH_UPDATE_STRATEGY")  # incremental, full_rebuild
    graph_update_interval: int = Field(default=3600, env="GRAPHRAG_GRAPH_UPDATE_INTERVAL")  # 1 hour
    graph_pruning_enabled: bool = Field(default=True, env="GRAPHRAG_GRAPH_PRUNING_ENABLED")
    graph_pruning_threshold: float = Field(default=0.1, env="GRAPHRAG_GRAPH_PRUNING_THRESHOLD")

    # Integration Settings
    integration_with_vector_search: bool = Field(default=True, env="GRAPHRAG_INTEGRATION_WITH_VECTOR_SEARCH")
    graph_weight_in_search: float = Field(default=0.3, env="GRAPHRAG_GRAPH_WEIGHT_IN_SEARCH")
    vector_weight_in_search: float = Field(default=0.7, env="GRAPHRAG_VECTOR_WEIGHT_IN_SEARCH")

    # Logging and Monitoring
    logging_enabled: bool = Field(default=True, env="GRAPHRAG_LOGGING_ENABLED")
    log_level: str = Field(default="INFO", env="GRAPHRAG_LOG_LEVEL")
    metrics_enabled: bool = Field(default=True, env="GRAPHRAG_METRICS_ENABLED")
    metrics_collection_interval: int = Field(default=60, env="GRAPHRAG_METRICS_COLLECTION_INTERVAL")

    def get_light_mode_config(self) -> dict:
        """Get Light mode configuration."""
        return {
            "mode": "light",
            "extraction_model": self.light_mode_extraction_model,
            "max_tokens": self.light_mode_max_tokens,
            "temperature": self.light_mode_temperature,
            "batch_size": self.light_mode_batch_size,
            "entity_types": self.entity_types,
            "relationship_types": self.relationship_types,
        }

    def get_general_mode_config(self) -> dict:
        """Get General mode configuration."""
        return {
            "mode": "general",
            "entity_model": self.general_mode_entity_model,
            "resolution_model": self.general_mode_resolution_model,
            "community_model": self.general_mode_community_model,
            "max_iterations": self.general_mode_max_iterations,
            "overlap_threshold": self.general_mode_overlap_threshold,
            "entity_types": self.entity_types,
            "relationship_types": self.relationship_types,
        }

    def get_entity_extraction_config(self) -> dict:
        """Get entity extraction configuration."""
        return {
            "enabled": self.entity_extraction_enabled,
            "model": self.entity_model,
            "entity_types": self.entity_types,
            "confidence_threshold": self.entity_confidence_threshold,
            "max_entities_per_doc": self.entity_max_entities_per_doc,
        }

    def get_relationship_extraction_config(self) -> dict:
        """Get relationship extraction configuration."""
        return {
            "enabled": self.relationship_extraction_enabled,
            "model": self.relationship_model,
            "relationship_types": self.relationship_types,
            "confidence_threshold": self.relationship_confidence_threshold,
            "max_relationships_per_doc": self.relationship_max_relationships_per_doc,
        }

    def get_community_detection_config(self) -> dict:
        """Get community detection configuration."""
        return {
            "enabled": self.community_detection_enabled,
            "algorithm": self.community_algorithm,
            "resolution": self.community_resolution,
            "min_size": self.community_min_size,
            "max_size": self.community_max_size,
        }

    def get_graph_search_config(self) -> dict:
        """Get graph search configuration."""
        return {
            "enabled": self.graph_search_enabled,
            "algorithm": self.graph_search_algorithm,
            "max_hops": self.graph_search_max_hops,
            "max_results": self.graph_search_max_results,
            "damping_factor": self.graph_search_damping_factor,
        }

    def get_storage_config(self) -> dict:
        """Get graph storage configuration."""
        return self.graph_storage_config.get(self.graph_storage_backend, {})

    def is_light_mode(self) -> bool:
        """Check if GraphRAG is in Light mode."""
        return self.graphrag_mode.lower() == "light"

    def is_general_mode(self) -> bool:
        """Check if GraphRAG is in General mode."""
        return self.graphrag_mode.lower() == "general"


# Global GraphRAG configuration instance
graphrag_config = GraphRAGConfig()