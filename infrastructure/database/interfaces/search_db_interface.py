"""
Search Database Interface Abstract Class

This module defines the abstract interface for search database providers.
All search database implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time


class SearchQueryType(str, Enum):
    """Search query types."""
    MATCH = "match"
    MATCH_PHRASE = "match_phrase"
    MATCH_ALL = "match_all"
    TERM = "term"
    TERMS = "terms"
    RANGE = "range"
    EXISTS = "exists"
    BOOL = "bool"
    WILDCARD = "wildcard"
    FUZZY = "fuzzy"
    NESTED = "nested"
    PREFIX = "prefix"
    REGEXP = "regexp"


class SortOrder(str, Enum):
    """Sort order types."""
    ASC = "asc"
    DESC = "desc"


class SearchQuery(BaseModel):
    """Search query model."""
    query_type: SearchQueryType
    field: Optional[str] = None
    value: Optional[Union[str, List[str], int, float, bool, Dict[str, Any]]] = None
    boost: Optional[float] = Field(default=None, ge=0.0)
    minimum_should_match: Optional[str] = None
    fuzziness: Optional[Union[str, int]] = None
    prefix_length: Optional[int] = Field(default=None, ge=0)
    max_expansions: Optional[int] = Field(default=None, ge=1)


class SearchSort(BaseModel):
    """Search sort model."""
    field: str
    order: SortOrder = SortOrder.ASC
    mode: Optional[str] = None  # min, max, sum, avg
    missing: Optional[Union[str, int, float]] = None
    unmapped_type: Optional[str] = None


class SearchHighlight(BaseModel):
    """Search highlight model."""
    fields: List[Dict[str, Any]]
    pre_tags: List[str] = Field(default=["<em>"])
    post_tags: List[str] = Field(default=["</em>"])
    fragment_size: Optional[int] = Field(default=150, ge=1)
    number_of_fragments: Optional[int] = Field(default=3, ge=0)
    require_field_match: Optional[bool] = None


class SearchAggregation(BaseModel):
    """Search aggregation model."""
    name: str
    type: str  # terms, avg, sum, min, max, stats, date_histogram, etc.
    field: Optional[str] = None
    script: Optional[str] = None
    size: Optional[int] = Field(default=10, ge=0)
    order: Optional[Dict[str, SortOrder]] = None
    format: Optional[str] = None
    interval: Optional[str] = None
    min_doc_count: Optional[int] = Field(default=1, ge=0)
    extended_bounds: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    """Search request model."""
    index: Union[str, List[str]]
    query: Optional[Union[SearchQuery, Dict[str, Any]]] = None
    filters: Optional[List[Union[SearchQuery, Dict[str, Any]]]] = None
    sort: Optional[List[Union[SearchSort, Dict[str, Any]]]] = None
    highlight: Optional[SearchHighlight] = None
    aggregations: Optional[List[SearchAggregation]] = None
    from_: Optional[int] = Field(default=0, ge=0, alias="from")
    size: Optional[int] = Field(default=10, ge=0, le=10000)
    track_total_hits: bool = True
    track_scores: bool = True
    explain: bool = False
    version: bool = False
    stored_fields: Optional[List[str]] = None
    docvalue_fields: Optional[List[str]] = None
    routing: Optional[str] = None
    preference: Optional[str] = None
    timeout: Optional[str] = None
    terminate_after: Optional[int] = Field(default=None, ge=0)


class SearchResult(BaseModel):
    """Search result model."""
    index: str
    id: str
    score: Optional[float] = None
    source: Optional[Dict[str, Any]] = None
    fields: Optional[Dict[str, Any]] = None
    highlight: Optional[Dict[str, List[str]]] = None
    sort: Optional[List[Union[str, int, float]]] = None
    matched_queries: Optional[List[str]] = None
    inner_hits: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    version: Optional[int] = None


class SearchResponse(BaseModel):
    """Search response model."""
    took: int  # milliseconds
    timed_out: bool
    hits: Dict[str, Any]  # total, hits, max_score
    aggregations: Optional[Dict[str, Any]] = None
    scrolls: Optional[str] = None
    suggestions: Optional[Dict[str, Any]] = None
    status: Optional[int] = None


class IndexMapping(BaseModel):
    """Index mapping model."""
    properties: Dict[str, Any]
    dynamic: Optional[str] = None  # true, false, strict
    date_detection: Optional[bool] = None
    numeric_detection: Optional[bool] = None


class IndexSettings(BaseModel):
    """Index settings model."""
    number_of_shards: int = Field(default=1, ge=1)
    number_of_replicas: int = Field(default=0, ge=0)
    refresh_interval: str = "1s"
    max_result_window: int = Field(default=10000, ge=0)
    analysis: Optional[Dict[str, Any]] = None
    similarity: Optional[Dict[str, Any]] = None


class IndexInfo(BaseModel):
    """Index information model."""
    name: str
    health: Optional[str] = None  # green, yellow, red
    status: Optional[str] = None  # open, close
    uuid: Optional[str] = None
    docs_count: Optional[int] = None
    docs_deleted: Optional[int] = None
    store_size: Optional[str] = None
    primary_store_size: Optional[str] = None
    mapping: Optional[IndexMapping] = None
    settings: Optional[IndexSettings] = None
    aliases: Optional[List[str]] = None


class DocumentOperation(BaseModel):
    """Document operation model."""
    operation: str  # index, update, delete
    index: str
    id: Optional[str] = None
    source: Optional[Dict[str, Any]] = None
    routing: Optional[str] = None
    version: Optional[int] = None
    version_type: Optional[str] = None
    if_seq_no: Optional[int] = None
    if_primary_term: Optional[int] = None


class BulkResponse(BaseModel):
    """Bulk operation response model."""
    took: int
    errors: bool
    items: List[Dict[str, Any]]


class SearchDBCapabilities(BaseModel):
    """Search database capabilities."""
    supported_query_types: List[SearchQueryType]
    supported_analyzers: List[str]
    supported_tokenizers: List[str]
    supported_filters: List[str]
    max_fields_per_document: int
    max_document_size_mb: int
    supports_aggregations: bool
    supports_highlights: bool
    supports_suggestions: bool
    supports_percolation: bool
    supports_geo_search: bool
    supports_nested_objects: bool
    supports_scripting: bool
    supports_templates: bool
    supports_snapshots: bool
    supports_ilm: bool  # Index Lifecycle Management
    supports_slm: bool  # Snapshot Lifecycle Management
    supports_cross_cluster_search: bool
    supports_machine_learning: bool
    max_indices: Optional[int] = None
    max_shards_per_node: Optional[int] = None


class SearchDBConfig(BaseModel):
    """Search database configuration."""
    provider: str
    hosts: List[str]
    username: Optional[str] = None
    password: Optional[str] = None
    scheme: str = "http"
    verify_certs: bool = True
    ca_certs: Optional[str] = None
    client_cert: Optional[str] = None
    client_key: Optional[str] = None
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_on_timeout: bool = True
    retry_on_status: List[int] = Field(default=[502, 503, 504, 429])
    http_compress: bool = True
    headers: Optional[Dict[str, str]] = None
    connection_params: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchDBInterface(ABC):
    """
    Abstract interface for search database providers.

    This class defines the contract that all search database implementations must follow.
    It provides a unified interface for different search databases while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: SearchDBConfig):
        """Initialize the search database with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.hosts = config.hosts
        self._capabilities: Optional[SearchDBCapabilities] = None
        self._connected = False

    @property
    @abstractmethod
    def capabilities(self) -> SearchDBCapabilities:
        """
        Get the capabilities of this search database provider.

        Returns:
            SearchDBCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the search database.

        Returns:
            bool: True if connection successful

        Raises:
            SearchDBException: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the search database.

        Returns:
            bool: True if disconnection successful
        """
        pass

    @abstractmethod
    async def create_index(
        self,
        index_name: str,
        mapping: IndexMapping,
        settings: Optional[IndexSettings] = None,
        **kwargs
    ) -> bool:
        """
        Create a new index.

        Args:
            index_name: Name of index to create
            mapping: Index mapping
            settings: Index settings
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if index created successfully

        Raises:
            SearchDBException: If index creation fails
        """
        pass

    @abstractmethod
    async def drop_index(
        self,
        index_name: str,
        **kwargs
    ) -> bool:
        """
        Drop an index.

        Args:
            index_name: Name of index to drop
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if index dropped successfully

        Raises:
            SearchDBException: If index drop fails
        """
        pass

    @abstractmethod
    async def list_indices(
        self,
        pattern: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        List all indices.

        Args:
            pattern: Pattern to match index names
            **kwargs: Additional provider-specific parameters

        Returns:
            List[str]: List of index names

        Raises:
            SearchDBException: If listing fails
        """
        pass

    @abstractmethod
    async def index_exists(
        self,
        index_name: str,
        **kwargs
    ) -> bool:
        """
        Check if an index exists.

        Args:
            index_name: Name of index to check
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if index exists

        Raises:
            SearchDBException: If check fails
        """
        pass

    @abstractmethod
    async def get_index_info(
        self,
        index_name: str,
        **kwargs
    ) -> IndexInfo:
        """
        Get information about an index.

        Args:
            index_name: Name of index
            **kwargs: Additional provider-specific parameters

        Returns:
            IndexInfo: Index information

        Raises:
            SearchDBException: If getting info fails
        """
        pass

    @abstractmethod
    async def index_document(
        self,
        index_name: str,
        document: Dict[str, Any],
        doc_id: Optional[str] = None,
        routing: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Index a document.

        Args:
            index_name: Name of index
            document: Document to index
            doc_id: Document ID (generated if not provided)
            routing: Routing value
            **kwargs: Additional provider-specific parameters

        Returns:
            str: Document ID

        Raises:
            SearchDBException: If indexing fails
        """
        pass

    @abstractmethod
    async def update_document(
        self,
        index_name: str,
        doc_id: str,
        document: Dict[str, Any],
        routing: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Update a document.

        Args:
            index_name: Name of index
            doc_id: Document ID
            document: Updated document data
            routing: Routing value
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if update successful

        Raises:
            SearchDBException: If update fails
        """
        pass

    @abstractmethod
    async def delete_document(
        self,
        index_name: str,
        doc_id: str,
        routing: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Delete a document.

        Args:
            index_name: Name of index
            doc_id: Document ID
            routing: Routing value
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if deletion successful

        Raises:
            SearchDBException: If deletion fails
        """
        pass

    @abstractmethod
    async def get_document(
        self,
        index_name: str,
        doc_id: str,
        routing: Optional[str] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.

        Args:
            index_name: Name of index
            doc_id: Document ID
            routing: Routing value
            **kwargs: Additional provider-specific parameters

        Returns:
            Optional[Dict[str, Any]]: Document or None if not found

        Raises:
            SearchDBException: If retrieval fails
        """
        pass

    @abstractmethod
    async def search(
        self,
        request: SearchRequest,
        **kwargs
    ) -> SearchResponse:
        """
        Search for documents.

        Args:
            request: Search request
            **kwargs: Additional provider-specific parameters

        Returns:
            SearchResponse: Search results

        Raises:
            SearchDBException: If search fails
        """
        pass

    @abstractmethod
    async def bulk(
        self,
        operations: List[DocumentOperation],
        **kwargs
    ) -> BulkResponse:
        """
        Execute bulk operations.

        Args:
            operations: List of document operations
            **kwargs: Additional provider-specific parameters

        Returns:
            BulkResponse: Bulk operation results

        Raises:
            SearchDBException: If bulk operation fails
        """
        pass

    async def simple_search(
        self,
        index: Union[str, List[str]],
        query: str,
        field: Optional[str] = None,
        size: int = 10,
        from_: int = 0,
        **kwargs
    ) -> SearchResponse:
        """
        Simple search interface for common use cases.

        Args:
            index: Index or list of indices to search
            query: Search query string
            field: Field to search in (if None, searches all fields)
            size: Number of results to return
            from_: Starting offset
            **kwargs: Additional parameters

        Returns:
            SearchResponse: Search results
        """
        if field:
            search_query = SearchQuery(
                query_type=SearchQueryType.MATCH,
                field=field,
                value=query
            )
        else:
            search_query = SearchQuery(
                query_type=SearchQueryType.MATCH_ALL,
                value={"query": query}
            )

        request = SearchRequest(
            index=index,
            query=search_query,
            size=size,
            from_=from_,
            **kwargs
        )

        return await self.search(request)

    async def term_search(
        self,
        index: Union[str, List[str]],
        field: str,
        value: Union[str, int, float, bool],
        size: int = 10,
        **kwargs
    ) -> SearchResponse:
        """
        Term search for exact matches.

        Args:
            index: Index or list of indices to search
            field: Field to search in
            value: Value to search for
            size: Number of results to return
            **kwargs: Additional parameters

        Returns:
            SearchResponse: Search results
        """
        search_query = SearchQuery(
            query_type=SearchQueryType.TERM,
            field=field,
            value=value
        )

        request = SearchRequest(
            index=index,
            query=search_query,
            size=size,
            **kwargs
        )

        return await self.search(request)

    async def range_search(
        self,
        index: Union[str, List[str]],
        field: str,
        gte: Optional[Union[int, float]] = None,
        lte: Optional[Union[int, float]] = None,
        gt: Optional[Union[int, float]] = None,
        lt: Optional[Union[int, float]] = None,
        size: int = 10,
        **kwargs
    ) -> SearchResponse:
        """
        Range search for numeric or date fields.

        Args:
            index: Index or list of indices to search
            field: Field to search in
            gte: Greater than or equal to
            lte: Less than or equal to
            gt: Greater than
            lt: Less than
            size: Number of results to return
            **kwargs: Additional parameters

        Returns:
            SearchResponse: Search results
        """
        range_dict = {}
        if gte is not None:
            range_dict["gte"] = gte
        if lte is not None:
            range_dict["lte"] = lte
        if gt is not None:
            range_dict["gt"] = gt
        if lt is not None:
            range_dict["lt"] = lt

        search_query = SearchQuery(
            query_type=SearchQueryType.RANGE,
            field=field,
            value=range_dict
        )

        request = SearchRequest(
            index=index,
            query=search_query,
            size=size,
            **kwargs
        )

        return await self.search(request)

    async def bool_search(
        self,
        index: Union[str, List[str]],
        must: Optional[List[Union[SearchQuery, Dict[str, Any]]]] = None,
        should: Optional[List[Union[SearchQuery, Dict[str, Any]]]] = None,
        must_not: Optional[List[Union[SearchQuery, Dict[str, Any]]]] = None,
        filter: Optional[List[Union[SearchQuery, Dict[str, Any]]]] = None,
        size: int = 10,
        **kwargs
    ) -> SearchResponse:
        """
        Boolean search combining multiple queries.

        Args:
            index: Index or list of indices to search
            must: Must match queries (AND)
            should: Should match queries (OR)
            must_not: Must not match queries (NOT)
            filter: Filter queries (no scoring)
            size: Number of results to return
            **kwargs: Additional parameters

        Returns:
            SearchResponse: Search results
        """
        bool_query = {}
        if must:
            bool_query["must"] = must
        if should:
            bool_query["should"] = should
        if must_not:
            bool_query["must_not"] = must_not
        if filter:
            bool_query["filter"] = filter

        search_query = SearchQuery(
            query_type=SearchQueryType.BOOL,
            value=bool_query
        )

        request = SearchRequest(
            index=index,
            query=search_query,
            size=size,
            **kwargs
        )

        return await self.search(request)

    async def count_documents(
        self,
        index: Union[str, List[str]],
        query: Optional[Union[SearchQuery, Dict[str, Any]]] = None,
        **kwargs
    ) -> int:
        """
        Count documents matching a query.

        Args:
            index: Index or list of indices to search
            query: Search query (if None, counts all documents)
            **kwargs: Additional parameters

        Returns:
            int: Document count
        """
        request = SearchRequest(
            index=index,
            query=query,
            size=0,  # Don't return documents
            track_total_hits=True,
            **kwargs
        )

        response = await self.search(request)
        return response.hits.get("total", {}).get("value", 0)

    def supports_feature(self, feature: str) -> bool:
        """
        Check if the provider supports a specific feature.

        Args:
            feature: Feature name (aggregations, highlights, suggestions, etc.)

        Returns:
            bool: True if feature is supported
        """
        capability_map = {
            "aggregations": "supports_aggregations",
            "highlights": "supports_highlights",
            "suggestions": "supports_suggestions",
            "percolation": "supports_percolation",
            "geo_search": "supports_geo_search",
            "nested_objects": "supports_nested_objects",
            "scripting": "supports_scripting",
            "templates": "supports_templates",
            "snapshots": "supports_snapshots",
            "ilm": "supports_ilm",
            "slm": "supports_slm",
            "cross_cluster_search": "supports_cross_cluster_search",
            "machine_learning": "supports_machine_learning",
        }

        capability_attr = capability_map.get(feature)
        if capability_attr is None:
            return False

        return getattr(self.capabilities, capability_attr, False)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the search database.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            if not self._connected:
                await self.connect()

            # List indices as a basic health check
            indices = await self.list_indices()

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "hosts": self.hosts,
                "indices_count": len(indices),
                "response_time_ms": None,  # Could be measured in implementations
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "hosts": self.hosts,
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
            "hosts": self.hosts,
            "connected": self._connected,
            "capabilities": self.capabilities.dict(),
            "config": {
                "scheme": self.config.scheme,
                "verify_certs": self.config.verify_certs,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                "retry_on_timeout": self.config.retry_on_timeout,
                "http_compress": self.config.http_compress,
            }
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class SearchDBException(Exception):
    """Exception raised by search database providers."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        index: str = None,
        doc_id: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.provider = provider
        self.index = index
        self.doc_id = doc_id
        self.error_code = error_code


class ConnectionException(SearchDBException):
    """Exception raised when connection fails."""
    pass


class IndexException(SearchDBException):
    """Exception raised for index-related errors."""
    pass


class QueryException(SearchDBException):
    """Exception raised when query execution fails."""
    pass


class DocumentException(SearchDBException):
    """Exception raised for document-related errors."""
    pass


class ValidationException(SearchDBException):
    """Exception raised when validation fails."""
    pass