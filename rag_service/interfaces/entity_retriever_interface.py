"""
Entity Retriever Interface Abstract Class

This module defines the abstract interface for entity-based retrievers.
All entity retriever implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time
import re
from dataclasses import dataclass


class EntityType(str, Enum):
    """Entity types."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    PRODUCT = "product"
    EVENT = "event"
    DATE = "date"
    TIME = "time"
    MONEY = "money"
    QUANTITY = "quantity"
    PERCENTAGE = "percentage"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    CUSTOM = "custom"


class RelationType(str, Enum):
    """Relation types between entities."""
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    MEMBER_OF = "member_of"
    CREATED_BY = "created_by"
    OWNS = "owns"
    KNOWS = "knows"
    MANAGES = "manages"
    COLLABORATES_WITH = "collaborates_with"
    CUSTOM = "custom"


class EntityMatchingStrategy(str, Enum):
    """Entity matching strategies."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    PHONETIC = "phonetic"
    ABBREVIATION = "abbreviation"
    HYBRID = "hybrid"


class GraphTraversalMode(str, Enum):
    """Graph traversal modes."""
    BFS = "bfs"  # Breadth-First Search
    DFS = "dfs"  # Depth-First Search
    SHORTEST_PATH = "shortest_path"
    ALL_PATHS = "all_paths"
    NEIGHBORS = "neighbors"


@dataclass
class EntityMention:
    """Entity mention in text."""
    text: str
    entity_type: EntityType
    start_position: int
    end_position: int
    confidence: float
    entity_id: Optional[str] = None
    canonical_name: Optional[str] = None
    aliases: Optional[List[str]] = None
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EntityRelation:
    """Relation between two entities."""
    subject_id: str
    object_id: str
    relation_type: RelationType
    confidence: float
    source_text: Optional[str] = None
    source_document_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Entity:
    """Entity representation."""
    id: str
    name: str
    entity_type: EntityType
    canonical_name: Optional[str] = None
    aliases: List[str] = None
    description: Optional[str] = None
    properties: Dict[str, Any] = None
    embedding: Optional[List[float]] = None
    mentions: List[EntityMention] = None
    relations: List[EntityRelation] = None
    created_at: float = None
    updated_at: float = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EntityResult:
    """Entity search result."""
    entity: Entity
    score: float
    match_type: EntityMatchingStrategy
    match_details: Optional[Dict[str, Any]] = None
    path_from_query: Optional[List[str]] = None  # For graph traversal


class EntityRetrieverConfig(BaseModel):
    """Entity retriever configuration."""
    provider: str
    graph_name: str
    entity_types: List[EntityType] = Field(default_factory=list)
    relation_types: List[RelationType] = Field(default_factory=list)
    matching_strategy: EntityMatchingStrategy = EntityMatchingStrategy.HYBRID
    fuzzy_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    semantic_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_relations_per_entity: int = Field(default=100, ge=1)
    max_path_length: int = Field(default=5, ge=1)
    enable_entity_linking: bool = True
    enable_relation_extraction: bool = True
    enable_coreference_resolution: bool = True
    enable_temporal_reasoning: bool = False
    enable_spatial_reasoning: bool = False
    cache_enabled: bool = True
    cache_size: int = Field(default=1000, ge=1)
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    debug_mode: bool = False
    custom_entity_types: Optional[Dict[str, EntityType]] = None
    custom_relation_types: Optional[Dict[str, RelationType]] = None
    custom_options: Optional[Dict[str, Any]] = None


class EntityQuery(BaseModel):
    """Entity query model."""
    name: Optional[str] = None
    entity_type: Optional[EntityType] = None
    properties: Optional[Dict[str, Any]] = None
    aliases: Optional[List[str]] = None
    description_contains: Optional[str] = None
    embedding: Optional[List[float]] = None
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class RelationQuery(BaseModel):
    """Relation query model."""
    subject_id: Optional[str] = None
    object_id: Optional[str] = None
    relation_type: Optional[RelationType] = None
    subject_type: Optional[EntityType] = None
    object_type: Optional[EntityType] = None
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class GraphQuery(BaseModel):
    """Graph traversal query."""
    start_entity_id: str
    traversal_mode: GraphTraversalMode = GraphTraversalMode.BFS
    max_depth: int = Field(default=3, ge=1, le=10)
    relation_types: Optional[List[RelationType]] = None
    entity_types: Optional[List[EntityType]] = None
    target_entity_type: Optional[EntityType] = None
    max_results: int = Field(default=100, ge=1)
    include_relations: bool = True
    include_paths: bool = True


class EntitySearchRequest(BaseModel):
    """Entity search request."""
    query: Union[str, EntityQuery, GraphQuery]
    top_k: int = Field(default=10, ge=1, le=1000)
    entity_types: Optional[List[EntityType]] = None
    include_relations: bool = False
    include_mentions: bool = False
    include_embeddings: bool = False
    matching_strategy: Optional[EntityMatchingStrategy] = None
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    filter_relations: Optional[RelationQuery] = None
    search_params: Optional[Dict[str, Any]] = None
    custom_options: Optional[Dict[str, Any]] = None


class EntitySearchResponse(BaseModel):
    """Entity search response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: Optional[str] = None
    results: List[EntityResult] = Field(default_factory=list)
    total_results: int = 0
    total_entities: Optional[int] = None
    search_time_ms: Optional[float] = None
    query_info: Optional[Dict[str, Any]] = None
    search_params_used: Optional[Dict[str, Any]] = None
    related_entities: Optional[Dict[str, List[Entity]]] = None
    entity_graph: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class BatchEntitySearchRequest(BaseModel):
    """Batch entity search request."""
    queries: List[Union[str, EntityQuery, GraphQuery]]
    top_k: int = Field(default=10, ge=1, le=1000)
    entity_types: Optional[List[EntityType]] = None
    include_relations: bool = False
    parallel_search: bool = True
    max_workers: int = Field(default=4, ge=1, le=16)
    search_params: Optional[Dict[str, Any]] = None


class BatchEntitySearchResponse(BaseModel):
    """Batch entity search response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    responses: List[EntitySearchResponse] = Field(default_factory=list)
    total_queries: int = 0
    total_results: int = 0
    total_time_ms: Optional[float] = None
    parallel_used: bool = False
    errors: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class EntityRetrieverCapabilities(BaseModel):
    """Entity retriever capabilities."""
    provider: str
    supported_entity_types: List[EntityType]
    supported_relation_types: List[RelationType]
    supported_matching_strategies: List[EntityMatchingStrategy]
    supported_traversal_modes: List[GraphTraversalMode]
    supports_entity_linking: bool
    supports_relation_extraction: bool
    supports_coreference_resolution: bool
    supports_temporal_reasoning: bool
    supports_spatial_reasoning: bool
    supports_graph_traversal: bool
    supports_semantic_search: bool
    supports_fuzzy_matching: bool
    supports_phonetic_matching: bool
    supports_batch_search: bool
    supports_streaming: bool
    supports_async: bool
    supports_filtering: bool
    supports_custom_entities: bool
    supports_custom_relations: bool
    max_entities_per_type: Optional[int] = None
    max_relations_per_query: Optional[int] = None
    accuracy_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speed_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    gpu_accelerated: bool = False


class EntityGraphStats(BaseModel):
    """Entity graph statistics."""
    graph_name: str
    total_entities: int
    total_relations: int
    entity_type_counts: Dict[str, int]
    relation_type_counts: Dict[str, int]
    average_degree: Optional[float] = None
    connected_components: Optional[int] = None
    graph_density: Optional[float] = None
    index_size_mb: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    last_updated: Optional[float] = None
    build_time_ms: Optional[float] = None
    average_query_time_ms: Optional[float] = None
    queries_per_second: Optional[float] = None


class EntityRetrieverInterface(ABC):
    """
    Abstract interface for entity-based retrievers.

    This class defines the contract that all entity retriever implementations must follow.
    It provides a unified interface for entity search and graph traversal while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: EntityRetrieverConfig):
        """Initialize the entity retriever with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.graph_name = config.graph_name
        self.entity_types = config.entity_types
        self.relation_types = config.relation_types
        self.matching_strategy = config.matching_strategy
        self.fuzzy_threshold = config.fuzzy_threshold
        self.semantic_threshold = config.semantic_threshold
        self.max_relations_per_entity = config.max_relations_per_entity
        self.max_path_length = config.max_path_length
        self.enable_entity_linking = config.enable_entity_linking
        self.enable_relation_extraction = config.enable_relation_extraction
        self.enable_coreference_resolution = config.enable_coreference_resolution
        self.enable_temporal_reasoning = config.enable_temporal_reasoning
        self.enable_spatial_reasoning = config.enable_spatial_reasoning
        self.cache_enabled = config.cache_enabled
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self._capabilities: Optional[EntityRetrieverCapabilities] = None
        self._graph_stats: Optional[EntityGraphStats] = None

    @property
    @abstractmethod
    def capabilities(self) -> EntityRetrieverCapabilities:
        """
        Get the capabilities of this entity retriever.

        Returns:
            EntityRetrieverCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def initialize_graph(self) -> bool:
        """
        Initialize the entity graph.

        Returns:
            bool: True if initialization successful

        Raises:
            EntityRetrieverException: If initialization fails
        """
        pass

    @abstractmethod
    async def search(
        self,
        request: EntitySearchRequest,
        **kwargs
    ) -> EntitySearchResponse:
        """
        Search for entities.

        Args:
            request: Entity search request
            **kwargs: Additional provider-specific parameters

        Returns:
            EntitySearchResponse: Search results

        Raises:
            EntityRetrieverException: If search fails
        """
        pass

    async def search_entities(
        self,
        query: Union[str, EntityQuery],
        top_k: int = 10,
        entity_types: Optional[List[EntityType]] = None,
        **kwargs
    ) -> EntitySearchResponse:
        """
        Search for entities (simplified interface).

        Args:
            query: Search query
            top_k: Number of results to return
            entity_types: Entity types to filter by
            **kwargs: Additional parameters

        Returns:
            EntitySearchResponse: Search results
        """
        request = EntitySearchRequest(
            query=query,
            top_k=top_k,
            entity_types=entity_types,
            **kwargs
        )

        return await self.search(request)

    async def find_entity_by_name(
        self,
        name: str,
        entity_type: Optional[EntityType] = None,
        matching_strategy: Optional[EntityMatchingStrategy] = None,
        **kwargs
    ) -> EntitySearchResponse:
        """
        Find entity by exact name.

        Args:
            name: Entity name to search for
            entity_type: Entity type filter
            matching_strategy: Matching strategy
            **kwargs: Additional parameters

        Returns:
            EntitySearchResponse: Search results
        """
        entity_query = EntityQuery(
            name=name,
            entity_type=entity_type
        )

        request = EntitySearchRequest(
            query=entity_query,
            top_k=1,
            matching_strategy=matching_strategy or EntityMatchingStrategy.EXACT,
            **kwargs
        )

        return await self.search(request)

    async def find_similar_entities(
        self,
        entity_id: str,
        top_k: int = 10,
        entity_types: Optional[List[EntityType]] = None,
        **kwargs
    ) -> EntitySearchResponse:
        """
        Find entities similar to a given entity.

        Args:
            entity_id: ID of the reference entity
            top_k: Number of results to return
            entity_types: Entity types to filter by
            **kwargs: Additional parameters

        Returns:
            EntitySearchResponse: Similar entities
        """
        # This requires getting the entity first, then searching by embedding
        # Implementation-specific
        raise NotImplementedError("find_similar_entities must be implemented by subclasses")

    async def traverse_graph(
        self,
        query: GraphQuery,
        **kwargs
    ) -> EntitySearchResponse:
        """
        Traverse the entity graph.

        Args:
            query: Graph traversal query
            **kwargs: Additional parameters

        Returns:
            EntitySearchResponse: Graph traversal results
        """
        request = EntitySearchRequest(
            query=query,
            include_relations=True,
            include_paths=True,
            **kwargs
        )

        return await self.search(request)

    async def find_related_entities(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        max_depth: int = 2,
        top_k: int = 50,
        **kwargs
    ) -> EntitySearchResponse:
        """
        Find entities related to a given entity.

        Args:
            entity_id: ID of the source entity
            relation_types: Relation types to follow
            max_depth: Maximum traversal depth
            top_k: Maximum number of results
            **kwargs: Additional parameters

        Returns:
            EntitySearchResponse: Related entities
        """
        graph_query = GraphQuery(
            start_entity_id=entity_id,
            traversal_mode=GraphTraversalMode.BFS,
            max_depth=max_depth,
            relation_types=relation_types,
            max_results=top_k
        )

        return await self.traverse_graph(graph_query, **kwargs)

    async def find_path_between_entities(
        self,
        start_entity_id: str,
        end_entity_id: str,
        max_path_length: Optional[int] = None,
        **kwargs
    ) -> EntitySearchResponse:
        """
        Find shortest path between two entities.

        Args:
            start_entity_id: ID of the start entity
            end_entity_id: ID of the end entity
            max_path_length: Maximum path length
            **kwargs: Additional parameters

        Returns:
            EntitySearchResponse: Path between entities
        """
        graph_query = GraphQuery(
            start_entity_id=start_entity_id,
            traversal_mode=GraphTraversalMode.SHORTEST_PATH,
            max_depth=max_path_length or self.max_path_length,
            max_results=1
        )

        # This is a simplified approach - actual implementation would be more complex
        response = await self.traverse_graph(graph_query, **kwargs)

        # Filter results to find paths to the target entity
        filtered_results = [
            result for result in response.results
            if result.entity.id == end_entity_id
        ]

        response.results = filtered_results
        response.total_results = len(filtered_results)

        return response

    async def batch_search(
        self,
        request: BatchEntitySearchRequest,
        **kwargs
    ) -> BatchEntitySearchResponse:
        """
        Search for multiple queries.

        Args:
            request: Batch search request
            **kwargs: Additional provider-specific parameters

        Returns:
            BatchEntitySearchResponse: Batch search results

        Raises:
            EntityRetrieverException: If batch search fails
        """
        if not self.capabilities.supports_batch_search:
            # Fall back to sequential processing
            responses = []
            total_time = 0

            for i, query in enumerate(request.queries):
                single_request = EntitySearchRequest(
                    query=query,
                    top_k=request.top_k,
                    entity_types=request.entity_types,
                    include_relations=request.include_relations,
                    search_params=request.search_params,
                )

                start_time = time.time()
                response = await self.search(single_request, **kwargs)
                end_time = time.time()

                response.query_id = f"batch_query_{i}"
                responses.append(response)
                total_time += (end_time - start_time) * 1000

            return BatchEntitySearchResponse(
                responses=responses,
                total_queries=len(request.queries),
                total_results=sum(len(r.results) for r in responses),
                total_time_ms=total_time,
                parallel_used=False,
            )

        # Parallel processing
        import asyncio
        semaphore = asyncio.Semaphore(request.max_workers)

        async def process_single_query(query: Union[str, EntityQuery, GraphQuery], index: int) -> EntitySearchResponse:
            async with semaphore:
                single_request = EntitySearchRequest(
                    query=query,
                    top_k=request.top_k,
                    entity_types=request.entity_types,
                    include_relations=request.include_relations,
                    search_params=request.search_params,
                )

                response = await self.search(single_request, **kwargs)
                response.query_id = f"batch_query_{index}"
                return response

        start_time = time.time()

        if request.parallel_search:
            tasks = [
                process_single_query(query, i)
                for i, query in enumerate(request.queries)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            responses = []
            for i, query in enumerate(request.queries):
                response = await process_single_query(query, i)
                responses.append(response)

        end_time = time.time()

        # Handle exceptions
        valid_responses = []
        errors = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                errors.append(f"Query {i} failed: {str(response)}")
            else:
                valid_responses.append(response)

        return BatchEntitySearchResponse(
            responses=valid_responses,
            total_queries=len(request.queries),
            total_results=sum(len(r.results) for r in valid_responses),
            total_time_ms=(end_time - start_time) * 1000,
            parallel_used=request.parallel_search,
            errors=errors,
        )

    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity name for matching.

        Args:
            name: Entity name to normalize

        Returns:
            str: Normalized entity name
        """
        # Basic normalization
        normalized = name.strip().lower()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove leading/trailing punctuation
        normalized = normalized.strip('.,;:!?()[]{}"\'')

        return normalized

    def _calculate_fuzzy_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate fuzzy similarity between two names.

        Args:
            name1: First name
            name2: Second name

        Returns:
            float: Similarity score (0 to 1)
        """
        # Simple Levenshtein distance implementation
        # In practice, you'd use a library like fuzzywuzzy or python-Levenshtein

        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        distance = levenshtein_distance(name1.lower(), name2.lower())
        max_len = max(len(name1), len(name2))

        if max_len == 0:
            return 1.0

        return 1.0 - (distance / max_len)

    def _calculate_semantic_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate semantic similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            float: Cosine similarity (0 to 1)
        """
        import numpy as np

        vec1 = np.array(embedding1, dtype=np.float32)
        vec2 = np.array(embedding2, dtype=np.float32)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def get_graph_stats(self) -> EntityGraphStats:
        """
        Get graph statistics.

        Returns:
            EntityGraphStats: Graph statistics

        Raises:
            EntityRetrieverException: If stats retrieval fails
        """
        if self._graph_stats is None:
            # Default implementation - override in subclasses
            return EntityGraphStats(
                graph_name=self.graph_name,
                total_entities=0,
                total_relations=0,
                entity_type_counts={},
                relation_type_counts={},
            )
        return self._graph_stats

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the entity retriever.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            # Create a simple test query
            test_request = EntitySearchRequest(
                query=EntityQuery(name="test_entity"),
                top_k=5
            )

            start_time = time.time()
            response = await self.search(test_request)
            end_time = time.time()

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "graph_name": self.graph_name,
                "response_time_ms": (end_time - start_time) * 1000,
                "test_results_count": len(response.results),
                "matching_strategy": self.matching_strategy,
                "total_entities": response.total_entities,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "graph_name": self.graph_name,
                "error": str(e)
            }

    def supports_entity_type(self, entity_type: EntityType) -> bool:
        """
        Check if the retriever supports a specific entity type.

        Args:
            entity_type: Entity type to check

        Returns:
            bool: True if entity type is supported
        """
        return entity_type in self.capabilities.supported_entity_types

    def supports_relation_type(self, relation_type: RelationType) -> bool:
        """
        Check if the retriever supports a specific relation type.

        Args:
            relation_type: Relation type to check

        Returns:
            bool: True if relation type is supported
        """
        return relation_type in self.capabilities.supported_relation_types

    def supports_matching_strategy(self, strategy: EntityMatchingStrategy) -> bool:
        """
        Check if the retriever supports a specific matching strategy.

        Args:
            strategy: Matching strategy to check

        Returns:
            bool: True if strategy is supported
        """
        return strategy in self.capabilities.supported_matching_strategies

    async def validate_request(
        self,
        request: EntitySearchRequest
    ) -> bool:
        """
        Validate an entity search request.

        Args:
            request: Request to validate

        Returns:
            bool: True if request is valid

        Raises:
            ValueError: If request is invalid
        """
        # Check query
        if request.query is None:
            raise ValueError("Query cannot be None")

        # Check top_k
        if request.top_k < 1 or request.top_k > 1000:
            raise ValueError("top_k must be between 1 and 1000")

        # Check similarity threshold
        if request.similarity_threshold is not None and (request.similarity_threshold < 0.0 or request.similarity_threshold > 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        # Check graph query parameters
        if isinstance(request.query, GraphQuery):
            if request.query.max_depth < 1 or request.query.max_depth > 10:
                raise ValueError("Graph query max_depth must be between 1 and 10")
            if request.query.max_results < 1 or request.query.max_results > 1000:
                raise ValueError("Graph query max_results must be between 1 and 1000")

        return True

    def get_retriever_info(self) -> Dict[str, Any]:
        """
        Get information about the entity retriever.

        Returns:
            Dict[str, Any]: Retriever information
        """
        return {
            "provider": self.provider_name,
            "graph_name": self.graph_name,
            "matching_strategy": self.matching_strategy,
            "capabilities": self.capabilities.dict(),
            "config": {
                "entity_types": self.entity_types,
                "relation_types": self.relation_types,
                "fuzzy_threshold": self.fuzzy_threshold,
                "semantic_threshold": self.semantic_threshold,
                "enable_entity_linking": self.enable_entity_linking,
                "enable_relation_extraction": self.enable_relation_extraction,
                "enable_coreference_resolution": self.enable_coreference_resolution,
                "enable_temporal_reasoning": self.enable_temporal_reasoning,
                "enable_spatial_reasoning": self.enable_spatial_reasoning,
                "cache_enabled": self.cache_enabled,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
        }


class EntityRetrieverException(Exception):
    """Exception raised by entity retrievers."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        graph_name: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.provider = provider
        self.graph_name = graph_name
        self.error_code = error_code


class GraphNotFoundException(EntityRetrieverException):
    """Exception raised when graph is not found."""
    pass


class EntityNotFoundException(EntityRetrieverException):
    """Exception raised when entity is not found."""
    pass


class RelationNotFoundException(EntityRetrieverException):
    """Exception raised when relation is not found."""
    pass


class GraphTraversalException(EntityRetrieverException):
    """Exception raised when graph traversal fails."""
    pass


class EntityLinkingException(EntityRetrieverException):
    """Exception raised when entity linking fails."""
    pass