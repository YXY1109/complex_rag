"""
Database Interfaces

This module contains abstract interfaces for all database providers.
All concrete implementations must inherit from these base classes.
"""

from .vector_db_interface import (
    # Vector database interfaces and models
    VectorDBInterface,
    VectorDBConfig,
    VectorDBCapabilities,
    VectorData,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorSearchResult,
    VectorCollectionInfo,
    VectorCollectionConfig,
    VectorMetricType,
    IndexType,

    # Vector database exceptions
    VectorDBException,
    ConnectionException as VectorConnectionException,
    CollectionException,
    IndexException,
    SearchException as VectorSearchException,
    ValidationException as VectorValidationException,
)

from .relational_db_interface import (
    # Relational database interfaces and models
    RelationalDBInterface,
    RelationalDBConfig,
    RelationalDBCapabilities,
    DatabaseConnection,
    QueryResult,
    TransactionConfig,
    TableInfo,
    IndexInfo as RelationalIndexInfo,
    IsolationLevel,
    QueryType,

    # Relational database exceptions
    RelationalDBException,
    ConnectionException as RelationalConnectionException,
    QueryException as RelationalQueryException,
    TransactionException,
    ValidationException as RelationalValidationException,
)

from .search_db_interface import (
    # Search database interfaces and models
    SearchDBInterface,
    SearchDBConfig,
    SearchDBCapabilities,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchQuery,
    SearchSort,
    SearchHighlight,
    SearchAggregation,
    IndexMapping,
    IndexSettings,
    IndexInfo as SearchIndexInfo,
    DocumentOperation,
    BulkResponse,
    SearchQueryType,
    SortOrder,

    # Search database exceptions
    SearchDBException,
    ConnectionException as SearchConnectionException,
    IndexException as SearchIndexException,
    QueryException as SearchQueryException,
    DocumentException,
    ValidationException as SearchValidationException,
)

__all__ = [
    # Vector database interfaces
    "VectorDBInterface",
    "VectorDBConfig",
    "VectorDBCapabilities",
    "VectorData",
    "VectorSearchRequest",
    "VectorSearchResponse",
    "VectorSearchResult",
    "VectorCollectionInfo",
    "VectorCollectionConfig",
    "VectorMetricType",
    "IndexType",
    "VectorDBException",
    "VectorConnectionException",
    "CollectionException",
    "VectorIndexException",
    "VectorSearchException",
    "VectorValidationException",

    # Relational database interfaces
    "RelationalDBInterface",
    "RelationalDBConfig",
    "RelationalDBCapabilities",
    "DatabaseConnection",
    "QueryResult",
    "TransactionConfig",
    "TableInfo",
    "RelationalIndexInfo",
    "IsolationLevel",
    "QueryType",
    "RelationalDBException",
    "RelationalConnectionException",
    "RelationalQueryException",
    "TransactionException",
    "RelationalValidationException",

    # Search database interfaces
    "SearchDBInterface",
    "SearchDBConfig",
    "SearchDBCapabilities",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SearchQuery",
    "SearchSort",
    "SearchHighlight",
    "SearchAggregation",
    "IndexMapping",
    "IndexSettings",
    "SearchIndexInfo",
    "DocumentOperation",
    "BulkResponse",
    "SearchQueryType",
    "SortOrder",
    "SearchDBException",
    "SearchConnectionException",
    "SearchIndexException",
    "SearchQueryException",
    "DocumentException",
    "SearchValidationException",
]