"""
RAPTOR Retriever Interface Abstract Class

This module defines the abstract interface for RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) retrievers.
All RAPTOR retriever implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time
import json
from dataclasses import dataclass


class TreeLevel(str, Enum):
    """Tree levels in RAPTOR hierarchy."""
    LEAF = "leaf"  # Original document chunks
    LEVEL_1 = "level_1"  # First level abstraction
    LEVEL_2 = "level_2"  # Second level abstraction
    LEVEL_3 = "level_3"  # Third level abstraction
    LEVEL_4 = "level_4"  # Fourth level abstraction
    LEVEL_5 = "level_5"  # Fifth level abstraction
    ROOT = "root"  # Highest level summary


class ClusteringMethod(str, Enum):
    """Clustering methods for tree construction."""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    AGGLOMERATIVE = "agglomerative"
    SPECTRAL = "spectral"
    CUSTOM = "custom"


class SummarizationMethod(str, Enum):
    """Summarization methods for node creation."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class RetrievalStrategy(str, Enum):
    """Retrieval strategies for RAPTOR."""
    TREE_TRAVERSAL = "tree_traversal"
    MULTI_LEVEL = "multi_level"
    ADAPTIVE = "adaptive"
    BEST_MATCH = "best_match"
    CUSTOM = "custom"


class TreeTraversalMode(str, Enum):
    """Tree traversal modes."""
    TOP_DOWN = "top_down"
    BOTTOM_UP = "bottom_up"
    BREADTH_FIRST = "breadth_first"
    ADAPTIVE = "adaptive"


@dataclass
class TreeNode:
    """Tree node in RAPTOR hierarchy."""
    id: str
    content: str
    level: TreeLevel
    children_ids: List[str] = None
    parent_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: float = None
    cluster_id: Optional[str] = None
    relevance_score: float = 0.0
    visit_count: int = 0


@dataclass
class TreePath:
    """Path through the RAPTOR tree."""
    nodes: List[TreeNode]
    scores: List[float]
    total_score: float
    traversal_mode: TreeTraversalMode


@dataclass
class RAPTORResult:
    """RAPTOR search result."""
    node: TreeNode
    score: float
    path: Optional[TreePath] = None
    level_scores: Optional[Dict[TreeLevel, float]] = None
    context: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class RAPTRetrieverConfig(BaseModel):
    """RAPTOR retriever configuration."""
    provider: str
    tree_name: str
    max_tree_depth: int = Field(default=5, ge=2, le=10)
    cluster_size: int = Field(default=10, ge=2, le=50)
    clustering_method: ClusteringMethod = ClusteringMethod.KMEANS
    summarization_method: SummarizationMethod = SummarizationMethod.ABSTRACTIVE
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE
    traversal_mode: TreeTraversalMode = TreeTraversalMode.ADAPTIVE
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    diversity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    adaptive_threshold: bool = True
    include_leaf_content: bool = True
    include_summaries: bool = True
    max_results_per_level: int = Field(default=5, ge=1, le=20)
    context_window_size: int = Field(default=4000, ge=1000)
    enable_tree_pruning: bool = True
    enable_dynamic_clustering: bool = False
    enable_incremental_updates: bool = False
    cache_enabled: bool = True
    cache_size: int = Field(default=1000, ge=1)
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    debug_mode: bool = False
    custom_options: Optional[Dict[str, Any]] = None


class RAPTORQuery(BaseModel):
    """RAPTOR query model."""
    query: str
    query_embedding: Optional[List[float]] = None
    target_levels: Optional[List[TreeLevel]] = None
    exclude_levels: Optional[List[TreeLevel]] = None
    max_depth: Optional[int] = None
    context_preference: str = Field(default="balanced")  # "detailed", "summary", "balanced"


class TreeSearchRequest(BaseModel):
    """Tree search request."""
    start_node_id: Optional[str] = None
    target_level: Optional[TreeLevel] = None
    max_depth: int = Field(default=3, ge=1, le=10)
    traversal_mode: TreeTraversalMode = TreeTraversalMode.ADAPTIVE
    include_siblings: bool = False
    include_parent: bool = True
    branching_factor: int = Field(default=3, ge=1, le=10)


class RAPTORSearchRequest(BaseModel):
    """RAPTOR search request."""
    query: Union[str, RAPTORQuery]
    top_k: int = Field(default=10, ge=1, le=1000)
    retrieval_strategy: Optional[RetrievalStrategy] = None
    traversal_mode: Optional[TreeTraversalMode] = None
    target_levels: Optional[List[TreeLevel]] = None
    exclude_levels: Optional[List[TreeLevel]] = None
    include_paths: bool = False
    include_context: bool = True
    include_leaf_content: bool = None
    include_summaries: bool = None
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    diversity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_results_per_level: Optional[int] = None
    adaptive_search: bool = True
    search_params: Optional[Dict[str, Any]] = None
    custom_options: Optional[Dict[str, Any]] = None


class RAPTORSearchResponse(BaseModel):
    """RAPTOR search response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: Optional[str] = None
    results: List[RAPTORResult] = Field(default_factory=list)
    total_results: int = 0
    tree_paths: Optional[List[TreePath]] = None
    level_results: Optional[Dict[TreeLevel, List[RAPTORResult]]] = None
    search_time_ms: Optional[float] = None
    query_info: Optional[Dict[str, Any]] = None
    search_params_used: Optional[Dict[str, Any]] = None
    tree_stats: Optional[Dict[str, Any]] = None
    adaptive_decisions: Optional[List[Dict[str, Any]]] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class BatchRAPTORSearchRequest(BaseModel):
    """Batch RAPTOR search request."""
    queries: List[Union[str, RAPTORQuery]]
    top_k: int = Field(default=10, ge=1, le=1000)
    retrieval_strategy: Optional[RetrievalStrategy] = None
    target_levels: Optional[List[TreeLevel]] = None
    parallel_search: bool = True
    max_workers: int = Field(default=4, ge=1, le=16)
    search_params: Optional[Dict[str, Any]] = None


class BatchRAPTORSearchResponse(BaseModel):
    """Batch RAPTOR search response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    responses: List[RAPTORSearchResponse] = Field(default_factory=list)
    total_queries: int = 0
    total_results: int = 0
    total_time_ms: Optional[float] = None
    parallel_used: bool = False
    errors: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class TreeConstructionRequest(BaseModel):
    """Tree construction request."""
    documents: List[str]
    document_ids: Optional[List[str]] = None
    chunk_size: int = Field(default=512, ge=100, le=2048)
    overlap: int = Field(default=50, ge=0, le=200)
    max_tree_depth: Optional[int] = None
    clustering_method: Optional[ClusteringMethod] = None
    summarization_method: Optional[SummarizationMethod] = None
    cluster_size: Optional[int] = None
    custom_options: Optional[Dict[str, Any]] = None


class TreeConstructionResponse(BaseModel):
    """Tree construction response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tree_id: str
    total_nodes: int
    tree_depth: int
    leaf_nodes: int
    construction_time_ms: Optional[float] = None
    tree_stats: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class RAPTRetrieverCapabilities(BaseModel):
    """RAPTOR retriever capabilities."""
    provider: str
    supported_clustering_methods: List[ClusteringMethod]
    supported_summarization_methods: List[SummarizationMethod]
    supported_retrieval_strategies: List[RetrievalStrategy]
    supported_traversal_modes: List[TreeTraversalMode]
    max_tree_depth: int
    max_cluster_size: int
    supports_tree_construction: bool
    supports_incremental_updates: bool
    supports_dynamic_clustering: bool
    supports_adaptive_search: bool
    supports_tree_pruning: bool
    supports_multi_query: bool
    supports_batch_search: bool
    supports_streaming: bool
    supports_async: bool
    supports_custom_clustering: bool
    supports_custom_summarization: bool
    supports_context_generation: bool
    supports_path_reconstruction: bool
    accuracy_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speed_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    memory_efficient: bool = False
    gpu_accelerated: bool = False


class RAPTORTreeStats(BaseModel):
    """RAPTOR tree statistics."""
    tree_name: str
    total_nodes: int
    tree_depth: int
    nodes_per_level: Dict[str, int]
    leaf_nodes: int
    root_nodes: int
    average_branching_factor: Optional[float] = None
    tree_size_mb: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    last_updated: Optional[float] = None
    construction_time_ms: Optional[float] = None
    average_query_time_ms: Optional[float] = None
    queries_per_second: Optional[float] = None
    clustering_info: Optional[Dict[str, Any]] = None
    summarization_info: Optional[Dict[str, Any]] = None


class RAPTRetrieverInterface(ABC):
    """
    Abstract interface for RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) retrievers.

    This class defines the contract that all RAPTOR retriever implementations must follow.
    It provides a unified interface for hierarchical document retrieval while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: RAPTRetrieverConfig):
        """Initialize the RAPTOR retriever with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.tree_name = config.tree_name
        self.max_tree_depth = config.max_tree_depth
        self.cluster_size = config.cluster_size
        self.clustering_method = config.clustering_method
        self.summarization_method = config.summarization_method
        self.retrieval_strategy = config.retrieval_strategy
        self.traversal_mode = config.traversal_mode
        self.similarity_threshold = config.similarity_threshold
        self.diversity_threshold = config.diversity_threshold
        self.adaptive_threshold = config.adaptive_threshold
        self.include_leaf_content = config.include_leaf_content
        self.include_summaries = config.include_summaries
        self.max_results_per_level = config.max_results_per_level
        self.context_window_size = config.context_window_size
        self.enable_tree_pruning = config.enable_tree_pruning
        self.enable_dynamic_clustering = config.enable_dynamic_clustering
        self.enable_incremental_updates = config.enable_incremental_updates
        self.cache_enabled = config.cache_enabled
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self._capabilities: Optional[RAPTRetrieverCapabilities] = None
        self._tree_stats: Optional[RAPTORTreeStats] = None

    @property
    @abstractmethod
    def capabilities(self) -> RAPTRetrieverCapabilities:
        """
        Get the capabilities of this RAPTOR retriever.

        Returns:
            RAPTRetrieverCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def initialize_tree(self) -> bool:
        """
        Initialize the RAPTOR tree.

        Returns:
            bool: True if initialization successful

        Raises:
            RAPTRetrieverException: If initialization fails
        """
        pass

    @abstractmethod
    async def construct_tree(
        self,
        request: TreeConstructionRequest,
        **kwargs
    ) -> TreeConstructionResponse:
        """
        Construct a RAPTOR tree from documents.

        Args:
            request: Tree construction request
            **kwargs: Additional provider-specific parameters

        Returns:
            TreeConstructionResponse: Construction result

        Raises:
            RAPTRetrieverException: If construction fails
        """
        pass

    @abstractmethod
    async def search(
        self,
        request: RAPTORSearchRequest,
        **kwargs
    ) -> RAPTORSearchResponse:
        """
        Search in the RAPTOR tree.

        Args:
            request: RAPTOR search request
            **kwargs: Additional provider-specific parameters

        Returns:
            RAPTORSearchResponse: Search results

        Raises:
            RAPTRetrieverException: If search fails
        """
        pass

    async def search_raptor(
        self,
        query: str,
        top_k: int = 10,
        target_levels: Optional[List[TreeLevel]] = None,
        **kwargs
    ) -> RAPTORSearchResponse:
        """
        Search in RAPTOR tree (simplified interface).

        Args:
            query: Search query
            top_k: Number of results to return
            target_levels: Target tree levels
            **kwargs: Additional parameters

        Returns:
            RAPTORSearchResponse: Search results
        """
        request = RAPTORSearchRequest(
            query=query,
            top_k=top_k,
            target_levels=target_levels,
            **kwargs
        )

        return await self.search(request)

    async def multi_level_search(
        self,
        query: str,
        levels: List[TreeLevel],
        top_k_per_level: int = 5,
        **kwargs
    ) -> RAPTORSearchResponse:
        """
        Search across multiple tree levels.

        Args:
            query: Search query
            levels: Tree levels to search
            top_k_per_level: Results per level
            **kwargs: Additional parameters

        Returns:
            RAPTORSearchResponse: Multi-level search results
        """
        request = RAPTORSearchRequest(
            query=query,
            retrieval_strategy=RetrievalStrategy.MULTI_LEVEL,
            target_levels=levels,
            max_results_per_level=top_k_per_level,
            **kwargs
        )

        return await self.search(request)

    async def tree_traversal_search(
        self,
        query: str,
        start_node_id: Optional[str] = None,
        traversal_mode: TreeTraversalMode = TreeTraversalMode.ADAPTIVE,
        max_depth: int = 3,
        **kwargs
    ) -> RAPTORSearchResponse:
        """
        Search using tree traversal.

        Args:
            query: Search query
            start_node_id: Starting node ID
            traversal_mode: Traversal mode
            max_depth: Maximum traversal depth
            **kwargs: Additional parameters

        Returns:
            RAPTORSearchResponse: Tree traversal results
        """
        request = RAPTORSearchRequest(
            query=query,
            retrieval_strategy=RetrievalStrategy.TREE_TRAVERSAL,
            traversal_mode=traversal_mode,
            include_paths=True,
            search_params={
                "start_node_id": start_node_id,
                "max_depth": max_depth,
                **kwargs.get("search_params", {})
            },
            **kwargs
        )

        return await self.search(request)

    async def adaptive_search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> RAPTORSearchResponse:
        """
        Perform adaptive search that selects optimal strategy.

        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional parameters

        Returns:
            RAPTORSearchResponse: Adaptive search results
        """
        request = RAPTORSearchRequest(
            query=query,
            top_k=top_k,
            retrieval_strategy=RetrievalStrategy.ADAPTIVE,
            adaptive_search=True,
            include_paths=True,
            **kwargs
        )

        return await self.search(request)

    async def batch_search(
        self,
        request: BatchRAPTORSearchRequest,
        **kwargs
    ) -> BatchRAPTORSearchResponse:
        """
        Search for multiple queries.

        Args:
            request: Batch search request
            **kwargs: Additional provider-specific parameters

        Returns:
            BatchRAPTORSearchResponse: Batch search results

        Raises:
            RAPTRetrieverException: If batch search fails
        """
        if not self.capabilities.supports_batch_search:
            # Fall back to sequential processing
            responses = []
            total_time = 0

            for i, query in enumerate(request.queries):
                single_request = RAPTORSearchRequest(
                    query=query,
                    top_k=request.top_k,
                    retrieval_strategy=request.retrieval_strategy,
                    target_levels=request.target_levels,
                    search_params=request.search_params,
                )

                start_time = time.time()
                response = await self.search(single_request, **kwargs)
                end_time = time.time()

                response.query_id = f"batch_query_{i}"
                responses.append(response)
                total_time += (end_time - start_time) * 1000

            return BatchRAPTORSearchResponse(
                responses=responses,
                total_queries=len(request.queries),
                total_results=sum(len(r.results) for r in responses),
                total_time_ms=total_time,
                parallel_used=False,
            )

        # Parallel processing
        import asyncio
        semaphore = asyncio.Semaphore(request.max_workers)

        async def process_single_query(query: Union[str, RAPTORQuery], index: int) -> RAPTORSearchResponse:
            async with semaphore:
                single_request = RAPTORSearchRequest(
                    query=query,
                    top_k=request.top_k,
                    retrieval_strategy=request.retrieval_strategy,
                    target_levels=request.target_levels,
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

        return BatchRAPTORSearchResponse(
            responses=valid_responses,
            total_queries=len(request.queries),
            total_results=sum(len(r.results) for r in valid_responses),
            total_time_ms=(end_time - start_time) * 1000,
            parallel_used=request.parallel_search,
            errors=errors,
        )

    def _compute_node_similarity(
        self,
        query_embedding: List[float],
        node_embedding: List[float]
    ) -> float:
        """
        Compute similarity between query and node embeddings.

        Args:
            query_embedding: Query embedding
            node_embedding: Node embedding

        Returns:
            float: Cosine similarity (0 to 1)
        """
        import numpy as np

        vec1 = np.array(query_embedding, dtype=np.float32)
        vec2 = np.array(node_embedding, dtype=np.float32)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _select_optimal_levels(
        self,
        query: str,
        query_length: int,
        target_levels: Optional[List[TreeLevel]] = None
    ) -> List[TreeLevel]:
        """
        Select optimal tree levels for the query.

        Args:
            query: Search query
            query_length: Length of the query
            target_levels: Preferred target levels

        Returns:
            List[TreeLevel]: Selected levels
        """
        if target_levels:
            return target_levels

        # Adaptive level selection based on query characteristics
        selected_levels = []

        # Always include leaf level for detailed content
        if self.include_leaf_content:
            selected_levels.append(TreeLevel.LEAF)

        # Include higher levels based on query length
        if query_length > 100:  # Long query - need summaries
            selected_levels.extend([TreeLevel.LEVEL_3, TreeLevel.LEVEL_4, TreeLevel.ROOT])
        elif query_length > 50:  # Medium query
            selected_levels.extend([TreeLevel.LEVEL_2, TreeLevel.LEVEL_3])
        else:  # Short query - focus on specific content
            selected_levels.extend([TreeLevel.LEVEL_1, TreeLevel.LEVEL_2])

        return list(set(selected_levels))  # Remove duplicates

    def _apply_diversification(
        self,
        results: List[RAPTORResult],
        diversity_threshold: float
    ) -> List[RAPTORResult]:
        """
        Apply diversification to search results.

        Args:
            results: Initial search results
            diversity_threshold: Diversity threshold

        Returns:
            List[RAPTORResult]: Diversified results
        """
        if diversity_threshold <= 0.0 or len(results) <= 1:
            return results

        diversified = []
        remaining = results.copy()

        # Add the best result
        diversified.append(remaining.pop(0))

        # Add diverse results
        while remaining and len(diversified) < len(results):
            best_idx = 0
            best_score = -float('inf')

            for i, candidate in enumerate(remaining):
                # Compute diversity penalty
                diversity_penalty = 0.0
                for selected in diversified:
                    # Simple content-based diversity (could be more sophisticated)
                    if candidate.node.embedding and selected.node.embedding:
                        similarity = self._compute_node_similarity(
                            candidate.node.embedding,
                            selected.node.embedding
                        )
                        diversity_penalty = max(diversity_penalty, similarity)

                # Final score with diversity penalty
                final_score = candidate.score - diversity_threshold * diversity_penalty

                if final_score > best_score:
                    best_score = final_score
                    best_idx = i

            # Add the best diverse candidate
            diversified.append(remaining.pop(best_idx))

        return diversified

    async def get_tree_stats(self) -> RAPTORTreeStats:
        """
        Get tree statistics.

        Returns:
            RAPTORTreeStats: Tree statistics

        Raises:
            RAPTRetrieverException: If stats retrieval fails
        """
        if self._tree_stats is None:
            # Default implementation - override in subclasses
            return RAPTORTreeStats(
                tree_name=self.tree_name,
                total_nodes=0,
                tree_depth=0,
                nodes_per_level={},
                leaf_nodes=0,
                root_nodes=0,
            )
        return self._tree_stats

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the RAPTOR retriever.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            # Create a simple test query
            test_request = RAPTORSearchRequest(
                query="test query",
                top_k=5
            )

            start_time = time.time()
            response = await self.search(test_request)
            end_time = time.time()

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "tree_name": self.tree_name,
                "response_time_ms": (end_time - start_time) * 1000,
                "test_results_count": len(response.results),
                "tree_depth": self.max_tree_depth,
                "clustering_method": self.clustering_method,
                "summarization_method": self.summarization_method,
                "retrieval_strategy": self.retrieval_strategy,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "tree_name": self.tree_name,
                "error": str(e)
            }

    def supports_clustering_method(self, method: ClusteringMethod) -> bool:
        """
        Check if the retriever supports a specific clustering method.

        Args:
            method: Clustering method to check

        Returns:
            bool: True if method is supported
        """
        return method in self.capabilities.supported_clustering_methods

    def supports_summarization_method(self, method: SummarizationMethod) -> bool:
        """
        Check if the retriever supports a specific summarization method.

        Args:
            method: Summarization method to check

        Returns:
            bool: True if method is supported
        """
        return method in self.capabilities.supported_summarization_methods

    def supports_retrieval_strategy(self, strategy: RetrievalStrategy) -> bool:
        """
        Check if the retriever supports a specific retrieval strategy.

        Args:
            strategy: Retrieval strategy to check

        Returns:
            bool: True if strategy is supported
        """
        return strategy in self.capabilities.supported_retrieval_strategies

    def supports_traversal_mode(self, mode: TreeTraversalMode) -> bool:
        """
        Check if the retriever supports a specific traversal mode.

        Args:
            mode: Traversal mode to check

        Returns:
            bool: True if mode is supported
        """
        return mode in self.capabilities.supported_traversal_modes

    async def validate_request(
        self,
        request: RAPTORSearchRequest
    ) -> bool:
        """
        Validate a RAPTOR search request.

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

        # Check diversity threshold
        if request.diversity_threshold is not None and (request.diversity_threshold < 0.0 or request.diversity_threshold > 1.0):
            raise ValueError("diversity_threshold must be between 0.0 and 1.0")

        # Check max results per level
        if request.max_results_per_level is not None and (request.max_results_per_level < 1 or request.max_results_per_level > 50):
            raise ValueError("max_results_per_level must be between 1 and 50")

        return True

    def get_retriever_info(self) -> Dict[str, Any]:
        """
        Get information about the RAPTOR retriever.

        Returns:
            Dict[str, Any]: Retriever information
        """
        return {
            "provider": self.provider_name,
            "tree_name": self.tree_name,
            "max_tree_depth": self.max_tree_depth,
            "cluster_size": self.cluster_size,
            "clustering_method": self.clustering_method,
            "summarization_method": self.summarization_method,
            "retrieval_strategy": self.retrieval_strategy,
            "traversal_mode": self.traversal_mode,
            "capabilities": self.capabilities.dict(),
            "config": {
                "similarity_threshold": self.similarity_threshold,
                "diversity_threshold": self.diversity_threshold,
                "adaptive_threshold": self.adaptive_threshold,
                "include_leaf_content": self.include_leaf_content,
                "include_summaries": self.include_summaries,
                "max_results_per_level": self.max_results_per_level,
                "context_window_size": self.context_window_size,
                "enable_tree_pruning": self.enable_tree_pruning,
                "enable_dynamic_clustering": self.enable_dynamic_clustering,
                "enable_incremental_updates": self.enable_incremental_updates,
                "cache_enabled": self.cache_enabled,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
        }


class RAPTRetrieverException(Exception):
    """Exception raised by RAPTOR retrievers."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        tree_name: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.provider = provider
        self.tree_name = tree_name
        self.error_code = error_code


class TreeNotFoundException(RAPTRetrieverException):
    """Exception raised when tree is not found."""
    pass


class NodeNotFoundException(RAPTRetrieverException):
    """Exception raised when node is not found."""
    pass


class TreeConstructionException(RAPTRetrieverException):
    """Exception raised when tree construction fails."""
    pass


class ClusteringException(RAPTRetrieverException):
    """Exception raised when clustering fails."""
    pass


class SummarizationException(RAPTRetrieverException):
    """Exception raised when summarization fails."""
    pass


class TreeTraversalException(RAPTRetrieverException):
    """Exception raised when tree traversal fails."""
    pass