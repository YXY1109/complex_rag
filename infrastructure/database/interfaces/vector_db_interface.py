"""
Vector Database Interface Abstract Class

This module defines the abstract interface for vector database providers.
All vector database implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple, AsyncGenerator
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time


class VectorMetricType(str, Enum):
    """Vector similarity metric types."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    INNER_PRODUCT = "inner_product"
    MANHATTAN = "manhattan"


class IndexType(str, Enum):
    """Vector index types."""
    FLAT = "flat"
    IVF_FLAT = "ivf_flat"
    IVF_SQ8 = "ivf_sq8"
    IVF_PQ = "ivf_pq"
    HNSW = "hnsw"
    ANNOY = "annoy"
    SCANN = "scann"


class VectorData(BaseModel):
    """Vector data model."""
    id: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None
    created_at: Optional[float] = Field(default_factory=lambda: time.time())


class VectorSearchRequest(BaseModel):
    """Vector search request model."""
    vector: List[float]
    top_k: int = Field(default=10, ge=1, le=1000)
    metric_type: Optional[VectorMetricType] = None
    namespace: Optional[str] = None
    include_metadata: bool = Field(default=True)
    include_vectors: bool = Field(default=False)
    filter: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None


class VectorSearchResult(BaseModel):
    """Vector search result model."""
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None
    namespace: Optional[str] = None


class VectorSearchResponse(BaseModel):
    """Vector search response model."""
    results: List[VectorSearchResult]
    total_count: int
    search_time_ms: Optional[float] = None
    namespace: Optional[str] = None


class VectorCollectionInfo(BaseModel):
    """Vector collection information."""
    name: str
    dimension: int
    metric_type: VectorMetricType
    index_type: IndexType
    count: int
    size_mb: Optional[float] = None
    created_at: Optional[float] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorCollectionConfig(BaseModel):
    """Vector collection configuration."""
    name: str
    dimension: int = Field(ge=1, le=32768)
    metric_type: VectorMetricType = VectorMetricType.COSINE
    index_type: IndexType = IndexType.HNSW
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # Index parameters
    index_params: Optional[Dict[str, Any]] = None

    # Collection settings
    max_vectors: Optional[int] = Field(default=None, ge=1)
    shards_num: Optional[int] = Field(default=1, ge=1)
    replicas_num: Optional[int] = Field(default=1, ge=0)


class VectorDBCapabilities(BaseModel):
    """Vector database capabilities."""
    supported_index_types: List[IndexType]
    supported_metric_types: List[VectorMetricType]
    max_dimension: int
    max_collections_per_database: int
    max_vectors_per_collection: Optional[int] = None
    supports_filtering: bool
    supports_namespaces: bool
    supports_metadata: bool
    supports_dynamic_schema: bool
    supports_batch_operations: bool
    supports_upsert: bool
    supports_delete_by_filter: bool
    supports_backup_restore: bool
    supports_data_migration: bool
    supports_streaming_insert: bool
    supports_async_operations: bool
    supports_multivector: bool
    supports_sparse_vectors: bool
    supports_binary_vectors: bool


class VectorDBConfig(BaseModel):
    """Vector database configuration."""
    provider: str
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    database: str = "default"
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_after: int = Field(default=1, ge=0)
    pool_size: int = Field(default=10, ge=1)
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    connection_params: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorDBInterface(ABC):
    """
    Abstract interface for vector database providers.

    This class defines the contract that all vector database implementations must follow.
    It provides a unified interface for different vector databases while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: VectorDBConfig):
        """Initialize the vector database with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.host = config.host
        self.port = config.port
        self.database = config.database
        self._capabilities: Optional[VectorDBCapabilities] = None
        self._connected = False

    @property
    @abstractmethod
    def capabilities(self) -> VectorDBCapabilities:
        """
        Get the capabilities of this vector database provider.

        Returns:
            VectorDBCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the vector database.

        Returns:
            bool: True if connection successful

        Raises:
            VectorDBException: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the vector database.

        Returns:
            bool: True if disconnection successful
        """
        pass

    @abstractmethod
    async def create_collection(
        self,
        config: VectorCollectionConfig,
        **kwargs
    ) -> bool:
        """
        Create a new vector collection.

        Args:
            config: Collection configuration
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if collection created successfully

        Raises:
            VectorDBException: If collection creation fails
        """
        pass

    @abstractmethod
    async def drop_collection(
        self,
        collection_name: str,
        **kwargs
    ) -> bool:
        """
        Drop a vector collection.

        Args:
            collection_name: Name of collection to drop
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if collection dropped successfully

        Raises:
            VectorDBException: If collection drop fails
        """
        pass

    @abstractmethod
    async def list_collections(self, **kwargs) -> List[str]:
        """
        List all collections in the database.

        Args:
            **kwargs: Additional provider-specific parameters

        Returns:
            List[str]: List of collection names

        Raises:
            VectorDBException: If listing fails
        """
        pass

    @abstractmethod
    async def collection_exists(
        self,
        collection_name: str,
        **kwargs
    ) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of collection to check
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if collection exists

        Raises:
            VectorDBException: If check fails
        """
        pass

    @abstractmethod
    async def get_collection_info(
        self,
        collection_name: str,
        **kwargs
    ) -> VectorCollectionInfo:
        """
        Get information about a collection.

        Args:
            collection_name: Name of collection
            **kwargs: Additional provider-specific parameters

        Returns:
            VectorCollectionInfo: Collection information

        Raises:
            VectorDBException: If getting info fails
        """
        pass

    @abstractmethod
    async def insert(
        self,
        collection_name: str,
        vectors: Union[VectorData, List[VectorData]],
        **kwargs
    ) -> List[str]:
        """
        Insert vectors into a collection.

        Args:
            collection_name: Name of collection
            vectors: Single vector or list of vectors to insert
            **kwargs: Additional provider-specific parameters

        Returns:
            List[str]: List of inserted vector IDs

        Raises:
            VectorDBException: If insertion fails
        """
        pass

    @abstractmethod
    async def upsert(
        self,
        collection_name: str,
        vectors: Union[VectorData, List[VectorData]],
        **kwargs
    ) -> List[str]:
        """
        Upsert vectors into a collection (insert or update).

        Args:
            collection_name: Name of collection
            vectors: Single vector or list of vectors to upsert
            **kwargs: Additional provider-specific parameters

        Returns:
            List[str]: List of upserted vector IDs

        Raises:
            VectorDBException: If upsert fails
        """
        pass

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        request: VectorSearchRequest,
        **kwargs
    ) -> VectorSearchResponse:
        """
        Search for similar vectors.

        Args:
            collection_name: Name of collection to search
            request: Search request
            **kwargs: Additional provider-specific parameters

        Returns:
            VectorSearchResponse: Search results

        Raises:
            VectorDBException: If search fails
        """
        pass

    @abstractmethod
    async def delete(
        self,
        collection_name: str,
        vector_ids: Union[str, List[str]],
        **kwargs
    ) -> bool:
        """
        Delete vectors by ID.

        Args:
            collection_name: Name of collection
            vector_ids: Single vector ID or list of vector IDs to delete
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if deletion successful

        Raises:
            VectorDBException: If deletion fails
        """
        pass

    @abstractmethod
    async def delete_by_filter(
        self,
        collection_name: str,
        filter_dict: Dict[str, Any],
        **kwargs
    ) -> int:
        """
        Delete vectors by metadata filter.

        Args:
            collection_name: Name of collection
            filter_dict: Filter dictionary
            **kwargs: Additional provider-specific parameters

        Returns:
            int: Number of deleted vectors

        Raises:
            VectorDBException: If deletion fails
        """
        pass

    @abstractmethod
    async def update_metadata(
        self,
        collection_name: str,
        vector_id: str,
        metadata: Dict[str, Any],
        **kwargs
    ) -> bool:
        """
        Update vector metadata.

        Args:
            collection_name: Name of collection
            vector_id: Vector ID to update
            metadata: New metadata
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if update successful

        Raises:
            VectorDBException: If update fails
        """
        pass

    @abstractmethod
    async def get_vector(
        self,
        collection_name: str,
        vector_id: str,
        **kwargs
    ) -> Optional[VectorData]:
        """
        Get a vector by ID.

        Args:
            collection_name: Name of collection
            vector_id: Vector ID to retrieve
            **kwargs: Additional provider-specific parameters

        Returns:
            Optional[VectorData]: Vector data or None if not found

        Raises:
            VectorDBException: If retrieval fails
        """
        pass

    @abstractmethod
    async def count_vectors(
        self,
        collection_name: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> int:
        """
        Count vectors in a collection.

        Args:
            collection_name: Name of collection
            filter_dict: Optional filter to apply
            **kwargs: Additional provider-specific parameters

        Returns:
            int: Number of vectors

        Raises:
            VectorDBException: If count fails
        """
        pass

    async def simple_search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        metric_type: Optional[VectorMetricType] = None,
        namespace: Optional[str] = None,
        **kwargs
    ) -> VectorSearchResponse:
        """
        Simple search interface for common use cases.

        Args:
            collection_name: Name of collection to search
            query_vector: Query vector
            top_k: Number of results to return
            metric_type: Similarity metric type
            namespace: Namespace to search in
            **kwargs: Additional parameters

        Returns:
            VectorSearchResponse: Search results
        """
        request = VectorSearchRequest(
            vector=query_vector,
            top_k=top_k,
            metric_type=metric_type,
            namespace=namespace,
            include_metadata=True,
            include_vectors=False,
            **kwargs
        )

        return await self.search(collection_name, request)

    async def batch_search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int = 10,
        metric_type: Optional[VectorMetricType] = None,
        **kwargs
    ) -> List[VectorSearchResponse]:
        """
        Batch search multiple query vectors.

        Args:
            collection_name: Name of collection to search
            query_vectors: List of query vectors
            top_k: Number of results per query
            metric_type: Similarity metric type
            **kwargs: Additional parameters

        Returns:
            List[VectorSearchResponse]: Search results for each query
        """
        results = []
        for vector in query_vectors:
            result = await self.simple_search(
                collection_name=collection_name,
                query_vector=vector,
                top_k=top_k,
                metric_type=metric_type,
                **kwargs
            )
            results.append(result)

        return results

    async def validate_vector(
        self,
        vector: List[float],
        dimension: Optional[int] = None
    ) -> bool:
        """
        Validate a vector.

        Args:
            vector: Vector to validate
            dimension: Expected dimension (if None, uses provider capability)

        Returns:
            bool: True if vector is valid

        Raises:
            ValueError: If vector is invalid
        """
        if not vector:
            raise ValueError("Vector cannot be empty")

        expected_dim = dimension or self.capabilities.max_dimension

        if len(vector) > expected_dim:
            raise ValueError(f"Vector dimension {len(vector)} exceeds maximum {expected_dim}")

        return True

    def supports_feature(self, feature: str) -> bool:
        """
        Check if the provider supports a specific feature.

        Args:
            feature: Feature name (filtering, namespaces, metadata, etc.)

        Returns:
            bool: True if feature is supported
        """
        capability_map = {
            "filtering": "supports_filtering",
            "namespaces": "supports_namespaces",
            "metadata": "supports_metadata",
            "batch_operations": "supports_batch_operations",
            "upsert": "supports_upsert",
            "delete_by_filter": "supports_delete_by_filter",
            "backup_restore": "supports_backup_restore",
            "data_migration": "supports_data_migration",
            "streaming_insert": "supports_streaming_insert",
            "async_operations": "supports_async_operations",
            "multivector": "supports_multivector",
            "sparse_vectors": "supports_sparse_vectors",
            "binary_vectors": "supports_binary_vectors",
        }

        capability_attr = capability_map.get(feature)
        if capability_attr is None:
            return False

        return getattr(self.capabilities, capability_attr, False)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the vector database.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            if not self._connected:
                await self.connect()

            # List collections as a basic health check
            collections = await self.list_collections()

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "collections_count": len(collections),
                "response_time_ms": None,  # Could be measured in implementations
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "error": str(e)
            }

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information.

        Returns:
            Dict[str, Any]: Connection information
        """
        return {
            "provider": self.provider_name,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "connected": self._connected,
            "capabilities": self.capabilities.dict(),
            "config": {
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                "pool_size": self.config.pool_size,
                "ssl_enabled": self.config.ssl_enabled,
            }
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class VectorDBException(Exception):
    """Exception raised by vector database providers."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        database: str = None,
        collection: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.provider = provider
        self.database = database
        self.collection = collection
        self.error_code = error_code


class ConnectionException(VectorDBException):
    """Exception raised when connection fails."""
    pass


class CollectionException(VectorDBException):
    """Exception raised for collection-related errors."""
    pass


class IndexException(VectorDBException):
    """Exception raised for index-related errors."""
    pass


class SearchException(VectorDBException):
    """Exception raised when search fails."""
    pass


class ValidationException(VectorDBException):
    """Exception raised when validation fails."""
    pass