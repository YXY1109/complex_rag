"""
Vector Retriever Interface Abstract Class

This module defines the abstract interface for vector-based retrievers.
All vector retriever implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time
import numpy as np
from dataclasses import dataclass


class RetrieverType(str, Enum):
    """Retriever types."""
    VECTOR = "vector"
    KEYWORD = "keyword"
    ENTITY = "entity"
    RAPTOR = "raptor"
    HYBRID = "hybrid"


class DistanceMetric(str, Enum):
    """Distance metrics for vector similarity."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"


class SearchMode(str, Enum):
    """Search modes."""
    ANN = "ann"  # Approximate Nearest Neighbor
    EXACT = "exact"  # Exact search
    HYBRID = "hybrid"  # Combination of ANN and exact


class RerankStrategy(str, Enum):
    """Reranking strategies."""
    NONE = "none"
    SIMILARITY = "similarity"
    DIVERSITY = "diversity"
    RELEVANCE = "relevance"
    CUSTOM = "custom"


@dataclass
class VectorResult:
    """Vector search result."""
    id: str
    content: str
    vector: Optional[List[float]] = None
    score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    distance: Optional[float] = None
    index: Optional[int] = None


class VectorRetrieverConfig(BaseModel):
    """Vector retriever configuration."""
    provider: str
    index_name: str
    dimension: int = Field(ge=1)
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    search_mode: SearchMode = SearchMode.ANN
    ef_search: Optional[int] = Field(default=None, ge=1)  # For HNSW index
    ef_construction: Optional[int] = Field(default=None, ge=1)  # For HNSW index
    m: Optional[int] = Field(default=None, ge=2)  # For HNSW index
    max_connections: Optional[int] = Field(default=None, ge=1)
    batch_size: int = Field(default=100, ge=1, le=1000)
    normalize_vectors: bool = True
    use_quantization: bool = False
    quantization_type: Optional[str] = None  # "scalar", "product", etc.
    cache_enabled: bool = True
    cache_size: int = Field(default=1000, ge=1)
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    debug_mode: bool = False
    custom_options: Optional[Dict[str, Any]] = None


class VectorSearchRequest(BaseModel):
    """Vector search request."""
    query_vector: List[float]
    top_k: int = Field(default=10, ge=1, le=1000)
    filter: Optional[Dict[str, Any]] = None
    include_vectors: bool = False
    include_metadata: bool = True
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rerank_strategy: RerankStrategy = RerankStrategy.NONE
    diversity_penalty: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    return_distances: bool = False
    search_params: Optional[Dict[str, Any]] = None
    custom_options: Optional[Dict[str, Any]] = None


class VectorSearchResponse(BaseModel):
    """Vector search response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: Optional[str] = None
    results: List[VectorResult] = Field(default_factory=list)
    total_results: int = 0
    search_time_ms: Optional[float] = None
    index_info: Optional[Dict[str, Any]] = None
    search_params_used: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class BatchVectorSearchRequest(BaseModel):
    """Batch vector search request."""
    query_vectors: List[List[float]]
    top_k: int = Field(default=10, ge=1, le=1000)
    filter: Optional[Dict[str, Any]] = None
    include_vectors: bool = False
    include_metadata: bool = True
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rerank_strategy: RerankStrategy = RerankStrategy.NONE
    parallel_search: bool = True
    max_workers: int = Field(default=4, ge=1, le=16)
    search_params: Optional[Dict[str, Any]] = None


class BatchVectorSearchResponse(BaseModel):
    """Batch vector search response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    responses: List[VectorSearchResponse] = Field(default_factory=list)
    total_queries: int = 0
    total_results: int = 0
    total_time_ms: Optional[float] = None
    parallel_used: bool = False
    errors: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class VectorRetrieverCapabilities(BaseModel):
    """Vector retriever capabilities."""
    provider: str
    supported_distance_metrics: List[DistanceMetric]
    supported_search_modes: List[SearchMode]
    supported_rerank_strategies: List[RerankStrategy]
    max_dimension: Optional[int] = None
    max_top_k: Optional[int] = None
    supports_batch_search: bool
    supports_streaming: bool
    supports_async: bool
    supports_filtering: bool
    supports_reranking: bool
    supports_diversification: bool
    supports_quantization: bool
    supports_incremental_updates: bool
    supports_real_time_search: bool
    requires_indexing: bool
    memory_mb_per_dimension: Optional[float] = None
    accuracy_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speed_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    gpu_accelerated: bool = False


class VectorIndexStats(BaseModel):
    """Vector index statistics."""
    index_name: str
    total_vectors: int
    dimension: int
    distance_metric: DistanceMetric
    index_size_mb: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    last_updated: Optional[float] = None
    build_time_ms: Optional[float] = None
    average_query_time_ms: Optional[float] = None
    queries_per_second: Optional[float] = None
    index_type: Optional[str] = None
    compression_ratio: Optional[float] = None


class VectorRetrieverInterface(ABC):
    """
    Abstract interface for vector-based retrievers.

    This class defines the contract that all vector retriever implementations must follow.
    It provides a unified interface for vector similarity search while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: VectorRetrieverConfig):
        """Initialize the vector retriever with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.index_name = config.index_name
        self.dimension = config.dimension
        self.distance_metric = config.distance_metric
        self.search_mode = config.search_mode
        self.ef_search = config.ef_search
        self.normalize_vectors = config.normalize_vectors
        self.use_quantization = config.use_quantization
        self.cache_enabled = config.cache_enabled
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self._capabilities: Optional[VectorRetrieverCapabilities] = None
        self._index_stats: Optional[VectorIndexStats] = None

    @property
    @abstractmethod
    def capabilities(self) -> VectorRetrieverCapabilities:
        """
        Get the capabilities of this vector retriever.

        Returns:
            VectorRetrieverCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def initialize_index(self) -> bool:
        """
        Initialize the vector index.

        Returns:
            bool: True if initialization successful

        Raises:
            VectorRetrieverException: If initialization fails
        """
        pass

    @abstractmethod
    async def search(
        self,
        request: VectorSearchRequest,
        **kwargs
    ) -> VectorSearchResponse:
        """
        Search for similar vectors.

        Args:
            request: Vector search request
            **kwargs: Additional provider-specific parameters

        Returns:
            VectorSearchResponse: Search results

        Raises:
            VectorRetrieverException: If search fails
        """
        pass

    async def search_vector(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        **kwargs
    ) -> VectorSearchResponse:
        """
        Search for similar vectors (simplified interface).

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filter: Optional filter criteria
            threshold: Optional similarity threshold
            **kwargs: Additional parameters

        Returns:
            VectorSearchResponse: Search results
        """
        request = VectorSearchRequest(
            query_vector=query_vector,
            top_k=top_k,
            filter=filter,
            threshold=threshold,
            **kwargs
        )

        return await self.search(request)

    async def batch_search(
        self,
        request: BatchVectorSearchRequest,
        **kwargs
    ) -> BatchVectorSearchResponse:
        """
        Search for multiple query vectors.

        Args:
            request: Batch search request
            **kwargs: Additional provider-specific parameters

        Returns:
            BatchVectorSearchResponse: Batch search results

        Raises:
            VectorRetrieverException: If batch search fails
        """
        if not self.capabilities.supports_batch_search:
            # Fall back to sequential processing
            responses = []
            total_time = 0

            for i, query_vector in enumerate(request.query_vectors):
                single_request = VectorSearchRequest(
                    query_vector=query_vector,
                    top_k=request.top_k,
                    filter=request.filter,
                    include_vectors=request.include_vectors,
                    include_metadata=request.include_metadata,
                    threshold=request.threshold,
                    rerank_strategy=request.rerank_strategy,
                    search_params=request.search_params,
                )

                start_time = time.time()
                response = await self.search(single_request, **kwargs)
                end_time = time.time()

                response.query_id = f"batch_query_{i}"
                responses.append(response)
                total_time += (end_time - start_time) * 1000

            return BatchVectorSearchResponse(
                responses=responses,
                total_queries=len(request.query_vectors),
                total_results=sum(len(r.results) for r in responses),
                total_time_ms=total_time,
                parallel_used=False,
            )

        # Parallel processing
        import asyncio
        semaphore = asyncio.Semaphore(request.max_workers)

        async def process_single_query(query_vector: List[float], index: int) -> VectorSearchResponse:
            async with semaphore:
                single_request = VectorSearchRequest(
                    query_vector=query_vector,
                    top_k=request.top_k,
                    filter=request.filter,
                    include_vectors=request.include_vectors,
                    include_metadata=request.include_metadata,
                    threshold=request.threshold,
                    rerank_strategy=request.rerank_strategy,
                    search_params=request.search_params,
                )

                response = await self.search(single_request, **kwargs)
                response.query_id = f"batch_query_{index}"
                return response

        start_time = time.time()

        if request.parallel_search:
            tasks = [
                process_single_query(query_vector, i)
                for i, query_vector in enumerate(request.query_vectors)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            responses = []
            for i, query_vector in enumerate(request.query_vectors):
                response = await process_single_query(query_vector, i)
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

        return BatchVectorSearchResponse(
            responses=valid_responses,
            total_queries=len(request.query_vectors),
            total_results=sum(len(r.results) for r in valid_responses),
            total_time_ms=(end_time - start_time) * 1000,
            parallel_used=request.parallel_search,
            errors=errors,
        )

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize a vector to unit length.

        Args:
            vector: Input vector

        Returns:
            List[float]: Normalized vector
        """
        if not self.normalize_vectors:
            return vector

        vector_array = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vector_array)

        if norm == 0:
            return vector

        return (vector_array / norm).tolist()

    def _compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Cosine similarity (0 to 1)
        """
        vec1_array = np.array(vec1, dtype=np.float32)
        vec2_array = np.array(vec2, dtype=np.float32)

        dot_product = np.dot(vec1_array, vec2_array)
        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _apply_reranking(
        self,
        results: List[VectorResult],
        query_vector: List[float],
        strategy: RerankStrategy,
        diversity_penalty: Optional[float] = None
    ) -> List[VectorResult]:
        """
        Apply reranking strategy to search results.

        Args:
            results: Initial search results
            query_vector: Query vector for relevance scoring
            strategy: Reranking strategy
            diversity_penalty: Penalty for similar results

        Returns:
            List[VectorResult]: Reranked results
        """
        if strategy == RerankStrategy.NONE:
            return results

        if strategy == RerankStrategy.SIMILARITY:
            # Sort by similarity score (already done by most vector databases)
            return sorted(results, key=lambda x: x.score, reverse=True)

        elif strategy == RerankStrategy.DIVERSITY and diversity_penalty is not None:
            # Apply Maximal Marginal Relevance (MMR)
            selected = []
            remaining = results.copy()

            while remaining and len(selected) < len(results):
                # Select best candidate
                best_idx = 0
                best_score = -float('inf')

                for i, candidate in enumerate(remaining):
                    # Relevance score
                    relevance = candidate.score

                    # Diversity penalty
                    diversity = 0.0
                    for selected_item in selected:
                        similarity = self._compute_cosine_similarity(
                            candidate.vector or [],
                            selected_item.vector or []
                        )
                        diversity = max(diversity, similarity)

                    # MMR score
                    mmr_score = relevance - diversity_penalty * diversity

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i

                # Move best candidate to selected
                selected.append(remaining.pop(best_idx))

            return selected

        elif strategy == RerankStrategy.RELEVANCE:
            # Custom relevance scoring (implementation-specific)
            return results  # Placeholder

        elif strategy == RerankStrategy.CUSTOM:
            # Custom reranking (implementation-specific)
            return results  # Placeholder

        return results

    async def get_index_stats(self) -> VectorIndexStats:
        """
        Get index statistics.

        Returns:
            VectorIndexStats: Index statistics

        Raises:
            VectorRetrieverException: If stats retrieval fails
        """
        if self._index_stats is None:
            # Default implementation - override in subclasses
            return VectorIndexStats(
                index_name=self.index_name,
                total_vectors=0,
                dimension=self.dimension,
                distance_metric=self.distance_metric,
            )
        return self._index_stats

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the vector retriever.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            # Create a simple test vector
            test_vector = [0.1] * self.dimension
            test_request = VectorSearchRequest(
                query_vector=test_vector,
                top_k=5
            )

            start_time = time.time()
            response = await self.search(test_request)
            end_time = time.time()

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "index_name": self.index_name,
                "response_time_ms": (end_time - start_time) * 1000,
                "test_results_count": len(response.results),
                "dimension": self.dimension,
                "distance_metric": self.distance_metric,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "index_name": self.index_name,
                "error": str(e)
            }

    def supports_distance_metric(self, metric: DistanceMetric) -> bool:
        """
        Check if the retriever supports a specific distance metric.

        Args:
            metric: Distance metric to check

        Returns:
            bool: True if metric is supported
        """
        return metric in self.capabilities.supported_distance_metrics

    def supports_search_mode(self, mode: SearchMode) -> bool:
        """
        Check if the retriever supports a specific search mode.

        Args:
            mode: Search mode to check

        Returns:
            bool: True if mode is supported
        """
        return mode in self.capabilities.supported_search_modes

    def supports_rerank_strategy(self, strategy: RerankStrategy) -> bool:
        """
        Check if the retriever supports a specific rerank strategy.

        Args:
            strategy: Rerank strategy to check

        Returns:
            bool: True if strategy is supported
        """
        return strategy in self.capabilities.supported_rerank_strategies

    async def validate_request(
        self,
        request: VectorSearchRequest
    ) -> bool:
        """
        Validate a vector search request.

        Args:
            request: Request to validate

        Returns:
            bool: True if request is valid

        Raises:
            ValueError: If request is invalid
        """
        # Check query vector dimension
        if len(request.query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(request.query_vector)} "
                f"does not match index dimension {self.dimension}"
            )

        # Check top_k
        if request.top_k < 1 or request.top_k > 1000:
            raise ValueError("top_k must be between 1 and 1000")

        # Check threshold
        if request.threshold is not None and (request.threshold < 0.0 or request.threshold > 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")

        # Check diversity penalty
        if request.diversity_penalty is not None and (request.diversity_penalty < 0.0 or request.diversity_penalty > 1.0):
            raise ValueError("diversity_penalty must be between 0.0 and 1.0")

        return True

    def get_retriever_info(self) -> Dict[str, Any]:
        """
        Get information about the vector retriever.

        Returns:
            Dict[str, Any]: Retriever information
        """
        return {
            "provider": self.provider_name,
            "index_name": self.index_name,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "search_mode": self.search_mode,
            "capabilities": self.capabilities.dict(),
            "config": {
                "normalize_vectors": self.normalize_vectors,
                "use_quantization": self.use_quantization,
                "cache_enabled": self.cache_enabled,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
        }


class VectorRetrieverException(Exception):
    """Exception raised by vector retrievers."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        index_name: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.provider = provider
        self.index_name = index_name
        self.error_code = error_code


class IndexNotFoundException(VectorRetrieverException):
    """Exception raised when index is not found."""
    pass


class DimensionMismatchException(VectorRetrieverException):
    """Exception raised when vector dimensions don't match."""
    pass


class SearchTimeoutException(VectorRetrieverException):
    """Exception raised when search times out."""
    pass


class InsufficientResultsException(VectorRetrieverException):
    """Exception raised when search returns insufficient results."""
    pass