"""
Elasticsearch Search Database Client Implementation

This module implements the Elasticsearch client for search database operations.
Based on the search database interface abstract class.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import json

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import (
    ElasticsearchException,
    ConnectionError as ESConnectionError,
    NotFoundError,
    RequestError,
    AuthenticationException
)
from pydantic import BaseModel, Field

from ...interfaces.search_db_interface import (
    SearchDBInterface,
    SearchDBConfig,
    SearchDBCapabilities,
    SearchQuery,
    SearchResult,
    SearchAggregation,
    IndexInfo,
    SearchSchema,
    SearchDBException,
    ConnectionException as SearchConnectionException,
    IndexException,
    QueryException
)


class ElasticsearchConfig(SearchDBConfig):
    """Elasticsearch-specific configuration."""

    hosts: Union[str, List[str]] = Field(default=["localhost:9200"], description="Elasticsearch hosts")
    scheme: str = Field(default="http", description="Connection scheme (http/https)")
    username: Optional[str] = Field(default=None, description="Elasticsearch username")
    password: Optional[str] = Field(default=None, description="Elasticsearch password")
    api_key: Optional[str] = Field(default=None, description="Elasticsearch API key")

    # Connection settings
    timeout: int = Field(default=30, description="Connection timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    retry_on_status: List[int] = Field(default=[502, 503, 504], description="Status codes to retry on")

    # SSL settings
    verify_certs: bool = Field(default=True, description="Verify SSL certificates")
    ca_certs: Optional[str] = Field(default=None, description="CA certificates file path")
    client_cert: Optional[str] = Field(default=None, description="Client certificate file path")
    client_key: Optional[str] = Field(default=None, description="Client key file path")

    # Performance settings
    max_connections: int = Field(default=10, description="Maximum number of connections")
    max_connections_per_host: int = Field(default=10, description="Maximum connections per host")
    http_compress: bool = Field(default=True, description="Enable HTTP compression")


class ElasticsearchCapabilities(SearchDBCapabilities):
    """Elasticsearch-specific capabilities."""

    def __init__(self):
        super().__init__(
            provider="elasticsearch",
            supported_query_types=[
                "match", "match_phrase", "match_all", "multi_match",
                "term", "terms", "range", "wildcard", "regexp",
                "fuzzy", "bool", "nested", "geo", "script"
            ],
            supported_index_types=[
                "keyword", "text", "integer", "long", "float", "double",
                "boolean", "date", "geo_point", "geo_shape", "ip", "object", "nested"
            ],
            max_connections=1000,
            supports_async=True,
            supports_aggregations=True,
            supports_highlighting=True,
            supports_suggestions=True,
            supports_analyzers=True,
            supports_filters=True,
            supports_sorting=True,
            supports_pagination=True,
            supports_scroll=True,
            supports_bulk_operations=True,
            supports_index_templates=True,
            supports_index_lifecycle_management=True,
            supports_cluster_management=True
        )


class ElasticsearchQuery:
    """Elasticsearch query builder."""

    def __init__(self):
        self._query = {
            "query": {"match_all": {}},
            "sort": [],
            "size": 10,
            "from": 0,
            "highlight": {},
            "aggs": {},
            "_source": True
        }

    def match(self, field: str, query: str, boost: Optional[float] = None, operator: str = "or") -> 'ElasticsearchQuery':
        """Add match query."""
        match_query = {"match": {field: {"query": query, "operator": operator}}}
        if boost is not None:
            match_query["match"][field]["boost"] = boost

        self._query["query"] = match_query
        return self

    def match_phrase(self, field: str, query: str, boost: Optional[float] = None) -> 'ElasticsearchQuery':
        """Add match phrase query."""
        match_query = {"match_phrase": {field: query}}
        if boost is not None:
            match_query["match_phrase"][field]["boost"] = boost

        self._query["query"] = match_query
        return self

    def multi_match(self, query: str, fields: List[str], boost: Optional[float] = None,
                   type: str = "best_fields") -> 'ElasticsearchQuery':
        """Add multi-match query."""
        multi_query = {
            "multi_match": {
                "query": query,
                "fields": fields,
                "type": type
            }
        }
        if boost is not None:
            multi_query["multi_match"]["boost"] = boost

        self._query["query"] = multi_query
        return self

    def term(self, field: str, value: Any, boost: Optional[float] = None) -> 'ElasticsearchQuery':
        """Add term query."""
        term_query = {"term": {field: value}}
        if boost is not None:
            term_query["term"][field]["boost"] = boost

        self._query["query"] = term_query
        return self

    def terms(self, field: str, values: List[Any], boost: Optional[float] = None) -> 'ElasticsearchQuery':
        """Add terms query."""
        terms_query = {"terms": {field: values}}
        if boost is not None:
            terms_query["terms_boost"] = {field: boost}

        self._query["query"] = terms_query
        return self

    def range(self, field: str, gte: Optional[Any] = None, lte: Optional[Any] = None,
              gt: Optional[Any] = None, lt: Optional[Any] = None, boost: Optional[float] = None) -> 'ElasticsearchQuery':
        """Add range query."""
        range_query = {"range": {field: {}}}
        if gte is not None:
            range_query["range"][field]["gte"] = gte
        if lte is not None:
            range_query["range"][field]["lte"] = lte
        if gt is not None:
            range_query["range"][field]["gt"] = gt
        if lt is not None:
            range_query["range"][field]["lt"] = lt
        if boost is not None:
            range_query["range"][field]["boost"] = boost

        self._query["query"] = range_query
        return self

    def fuzzy(self, field: str, value: str, fuzziness: str = "AUTO",
              prefix_length: Optional[int] = None, boost: Optional[float] = None) -> 'ElasticsearchQuery':
        """Add fuzzy query."""
        fuzzy_query = {
            "fuzzy": {
                field: {
                    "value": value,
                    "fuzziness": fuzziness
                }
            }
        }
        if prefix_length is not None:
            fuzzy_query["fuzzy"][field]["prefix_length"] = prefix_length
        if boost is not None:
            fuzzy_query["fuzzy"][field]["boost"] = boost

        self._query["query"] = fuzzy_query
        return self

    def wildcard(self, field: str, value: str, boost: Optional[float] = None) -> 'ElasticsearchQuery':
        """Add wildcard query."""
        wildcard_query = {"wildcard": {field: value}}
        if boost is not None:
            wildcard_query["wildcard"][field]["boost"] = boost

        self._query["query"] = wildcard_query
        return self

    def bool(self, must: Optional[List[Dict]] = None, must_not: Optional[List[Dict]] = None,
             should: Optional[List[Dict]] = None, filter: Optional[List[Dict]] = None,
             minimum_should_match: Optional[int] = None) -> 'ElasticsearchQuery':
        """Add boolean query."""
        bool_query = {"bool": {}}

        if must:
            bool_query["bool"]["must"] = must
        if must_not:
            bool_query["bool"]["must_not"] = must_not
        if should:
            bool_query["bool"]["should"] = should
            if minimum_should_match is not None:
                bool_query["bool"]["minimum_should_match"] = minimum_should_match
        if filter:
            bool_query["bool"]["filter"] = filter

        self._query["query"] = bool_query
        return self

    def filter(self, *conditions: Dict) -> 'ElasticsearchQuery':
        """Add filter conditions."""
        if "bool" not in self._query["query"]:
            self._query["query"] = {"bool": {}}

        if "filter" not in self._query["query"]["bool"]:
            self._query["query"]["bool"]["filter"] = []

        self._query["query"]["bool"]["filter"].extend(conditions)
        return self

    def sort(self, field: str, order: str = "asc") -> 'ElasticsearchQuery':
        """Add sort condition."""
        self._query["sort"].append({field: {"order": order}})
        return self

    def size(self, size: int) -> 'ElasticsearchQuery':
        """Set result size."""
        self._query["size"] = size
        return self

    def from_offset(self, offset: int) -> 'ElasticsearchQuery':
        """Set offset."""
        self._query["from"] = offset
        return self

    def source(self, fields: Union[List[str], bool]) -> 'ElasticsearchQuery':
        """Set source fields."""
        self._query["_source"] = fields
        return self

    def highlight(self, fields: List[str], pre_tags: Optional[List[str]] = None,
                 post_tags: Optional[List[str]] = None, fragment_size: Optional[int] = None) -> 'ElasticsearchQuery':
        """Add highlighting."""
        highlight_config = {"fields": {field: {} for field in fields}}

        if pre_tags:
            highlight_config["pre_tags"] = pre_tags
        if post_tags:
            highlight_config["post_tags"] = post_tags
        if fragment_size:
            highlight_config["fragment_size"] = fragment_size

        self._query["highlight"] = highlight_config
        return self

    def aggregate(self, name: str, agg_type: str, field: Optional[str] = None,
                 script: Optional[str] = None, size: Optional[int] = None) -> 'ElasticsearchQuery':
        """Add aggregation."""
        agg_config = {agg_type: {}}

        if field:
            agg_config[agg_type]["field"] = field
        if script:
            agg_config[agg_type]["script"] = script
        if size:
            agg_config[agg_type]["size"] = size

        self._query["aggs"][name] = agg_config
        return self

    def build(self) -> Dict[str, Any]:
        """Build the final query."""
        return self._query


class ElasticsearchClient(SearchDBInterface):
    """
    Elasticsearch client implementation for search database operations.

    Provides async Elasticsearch operations with comprehensive error handling,
    connection management, and advanced search capabilities.
    """

    def __init__(self, config: ElasticsearchConfig):
        super().__init__(config)
        self.config: ElasticsearchConfig = config
        self._client: Optional[AsyncElasticsearch] = None
        self._connected = False
        self._capabilities = ElasticsearchCapabilities()

    @property
    def capabilities(self) -> SearchDBCapabilities:
        """Get Elasticsearch capabilities."""
        return self._capabilities

    async def connect(self) -> bool:
        """
        Connect to Elasticsearch cluster.

        Returns:
            bool: True if connection successful

        Raises:
            SearchConnectionException: If connection fails
        """
        try:
            # Build client configuration
            client_config = {
                "hosts": self.config.hosts,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                "retry_on_timeout": self.config.retry_on_timeout,
                "retry_on_status": self.config.retry_on_status,
                "http_compress": self.config.http_compress,
                "maxsize": self.config.max_connections_per_host
            }

            # Add authentication
            if self.config.api_key:
                client_config["api_key"] = self.config.api_key
            elif self.config.username and self.config.password:
                client_config["http_auth"] = (self.config.username, self.config.password)

            # Add SSL configuration
            if self.config.scheme == "https":
                client_config["verify_certs"] = self.config.verify_certs
                if self.config.ca_certs:
                    client_config["ca_certs"] = self.config.ca_certs
                if self.config.client_cert:
                    client_config["client_cert"] = self.config.client_cert
                if self.config.client_key:
                    client_config["client_key"] = self.config.client_key

            # Create async client
            self._client = AsyncElasticsearch(**client_config)

            # Test connection
            info = await self._client.info()

            self._connected = True
            logger.info(f"Connected to Elasticsearch cluster: {info['cluster_name']}")
            return True

        except ESConnectionError as e:
            error_msg = f"Failed to connect to Elasticsearch: {str(e)}"
            logger.error(error_msg)
            raise SearchConnectionException(error_msg, provider="elasticsearch") from e
        except AuthenticationException as e:
            error_msg = f"Authentication failed for Elasticsearch: {str(e)}"
            logger.error(error_msg)
            raise SearchConnectionException(error_msg, provider="elasticsearch") from e
        except Exception as e:
            error_msg = f"Unexpected error connecting to Elasticsearch: {str(e)}"
            logger.error(error_msg)
            raise SearchConnectionException(error_msg, provider="elasticsearch") from e

    async def disconnect(self) -> None:
        """Disconnect from Elasticsearch cluster."""
        try:
            if self._client:
                await self._client.close()
                self._client = None
                self._connected = False
                logger.info("Disconnected from Elasticsearch")
        except Exception as e:
            logger.error(f"Error disconnecting from Elasticsearch: {str(e)}")

    async def create_index(
        self,
        index_name: str,
        mapping: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new index.

        Args:
            index_name: Index name
            mapping: Index mapping
            settings: Index settings

        Returns:
            bool: True if creation successful

        Raises:
            IndexException: If index creation fails
        """
        try:
            if not self._connected:
                raise SearchConnectionException("Not connected to Elasticsearch", provider="elasticsearch")

            # Check if index already exists
            if await self._client.indices.exists(index=index_name):
                logger.warning(f"Index {index_name} already exists")
                return True

            # Create index body
            body = {}
            if settings:
                body["settings"] = settings
            if mapping:
                body["mappings"] = mapping

            # Create index
            await self._client.indices.create(index=index_name, body=body)

            logger.info(f"Created Elasticsearch index: {index_name}")
            return True

        except RequestError as e:
            error_msg = f"Failed to create index {index_name}: {str(e)}"
            logger.error(error_msg)
            raise IndexException(error_msg, provider="elasticsearch") from e
        except Exception as e:
            error_msg = f"Unexpected error creating index {index_name}: {str(e)}"
            logger.error(error_msg)
            raise IndexException(error_msg, provider="elasticsearch") from e

    async def drop_index(self, index_name: str) -> bool:
        """
        Drop an index.

        Args:
            index_name: Index name

        Returns:
            bool: True if deletion successful

        Raises:
            IndexException: If index deletion fails
        """
        try:
            if not self._connected:
                raise SearchConnectionException("Not connected to Elasticsearch", provider="elasticsearch")

            # Check if index exists
            if not await self._client.indices.exists(index=index_name):
                logger.warning(f"Index {index_name} does not exist")
                return True

            # Delete index
            await self._client.indices.delete(index=index_name)

            logger.info(f"Dropped Elasticsearch index: {index_name}")
            return True

        except NotFoundError:
            logger.warning(f"Index {index_name} does not exist")
            return True
        except RequestError as e:
            error_msg = f"Failed to drop index {index_name}: {str(e)}"
            logger.error(error_msg)
            raise IndexException(error_msg, provider="elasticsearch") from e
        except Exception as e:
            error_msg = f"Unexpected error dropping index {index_name}: {str(e)}"
            logger.error(error_msg)
            raise IndexException(error_msg, provider="elasticsearch") from e

    async def list_indices(self, pattern: str = "*") -> List[str]:
        """
        List indices.

        Args:
            pattern: Index name pattern

        Returns:
            List[str]: Index names

        Raises:
            IndexException: If listing fails
        """
        try:
            if not self._connected:
                raise SearchConnectionException("Not connected to Elasticsearch", provider="elasticsearch")

            # Get indices
            response = await self._client.indices.get(index=pattern)
            return list(response.keys())

        except NotFoundError:
            return []
        except Exception as e:
            error_msg = f"Failed to list indices: {str(e)}"
            logger.error(error_msg)
            raise IndexException(error_msg, provider="elasticsearch") from e

    async def get_index_info(self, index_name: str) -> IndexInfo:
        """
        Get index information.

        Args:
            index_name: Index name

        Returns:
            IndexInfo: Index information

        Raises:
            IndexException: If getting info fails
        """
        try:
            if not self._connected:
                raise SearchConnectionException("Not connected to Elasticsearch", provider="elasticsearch")

            # Get index stats
            stats = await self._client.indices.stats(index=index_name)

            # Get index mapping
            mapping = await self._client.indices.get_mapping(index=index_name)

            # Get index settings
            settings = await self._client.indices.get_settings(index=index_name)

            # Extract information
            index_stats = list(stats["indices"].values())[0] if stats["indices"] else {}
            index_mapping = list(mapping.values())[0]["mappings"] if mapping else {}
            index_settings = list(settings.values())[0]["settings"] if settings else {}

            return IndexInfo(
                name=index_name,
                health=index_stats.get("health", "unknown"),
                status=index_stats.get("status", "unknown"),
                docs_count=index_stats.get("total", {}).get("docs", {}).get("count", 0),
                store_size=index_stats.get("total", {}).get("store", {}).get("size_in_bytes", 0),
                mapping=index_mapping,
                settings=index_settings,
                created_time=None  # Not available in stats
            )

        except NotFoundError:
            raise IndexException(f"Index {index_name} does not exist", provider="elasticsearch")
        except Exception as e:
            error_msg = f"Failed to get index info for {index_name}: {str(e)}"
            logger.error(error_msg)
            raise IndexException(error_msg, provider="elasticsearch") from e

    async def index_document(
        self,
        index_name: str,
        document: Dict[str, Any],
        doc_id: Optional[str] = None,
        refresh: bool = False
    ) -> str:
        """
        Index a document.

        Args:
            index_name: Index name
            document: Document to index
            doc_id: Document ID
            refresh: Refresh index

        Returns:
            str: Document ID

        Raises:
            QueryException: If indexing fails
        """
        try:
            if not self._connected:
                raise SearchConnectionException("Not connected to Elasticsearch", provider="elasticsearch")

            response = await self._client.index(
                index=index_name,
                id=doc_id,
                body=document,
                refresh=refresh
            )

            doc_id = response["_id"]
            logger.debug(f"Indexed document {doc_id} in {index_name}")
            return doc_id

        except RequestError as e:
            error_msg = f"Failed to index document in {index_name}: {str(e)}"
            logger.error(error_msg)
            raise QueryException(error_msg, provider="elasticsearch") from e
        except Exception as e:
            error_msg = f"Unexpected error indexing document in {index_name}: {str(e)}"
            logger.error(error_msg)
            raise QueryException(error_msg, provider="elasticsearch") from e

    async def bulk_index(
        self,
        operations: List[Dict[str, Any]],
        refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Bulk index documents.

        Args:
            operations: Bulk operations
            refresh: Refresh index

        Returns:
            Dict[str, Any]: Bulk operation result

        Raises:
            QueryException: If bulk operation fails
        """
        try:
            if not self._connected:
                raise SearchConnectionException("Not connected to Elasticsearch", provider="elasticsearch")

            from elasticsearch.helpers import async_bulk

            # Perform bulk operation
            response = await async_bulk(self._client, operations, refresh=refresh)

            # Count errors
            errors = [item for item in response if item.get("error")]

            if errors:
                logger.warning(f"Bulk operation had {len(errors)} errors")
            else:
                logger.debug(f"Bulk operation completed successfully")

            return {
                "success": len(errors) == 0,
                "indexed": len([item for item in response if item.get("index")]),
                "errors": errors,
                "total": len(response)
            }

        except Exception as e:
            error_msg = f"Failed to perform bulk operation: {str(e)}"
            logger.error(error_msg)
            raise QueryException(error_msg, provider="elasticsearch") from e

    async def search(
        self,
        index_name: str,
        query: Dict[str, Any],
        size: Optional[int] = None,
        from_offset: Optional[int] = None,
        sort: Optional[List[Dict[str, Any]]] = None,
        highlight: Optional[Dict[str, Any]] = None,
        aggs: Optional[Dict[str, Any]] = None,
        source: Optional[Union[List[str], bool]] = None
    ) -> SearchResult:
        """
        Search documents.

        Args:
            index_name: Index name or pattern
            query: Search query
            size: Result size
            from_offset: Result offset
            sort: Sort configuration
            highlight: Highlight configuration
            aggs: Aggregations
            source: Source fields

        Returns:
            SearchResult: Search results

        Raises:
            QueryException: If search fails
        """
        try:
            if not self._connected:
                raise SearchConnectionException("Not connected to Elasticsearch", provider="elasticsearch")

            # Build search body
            body = {"query": query}

            if size is not None:
                body["size"] = size
            if from_offset is not None:
                body["from"] = from_offset
            if sort:
                body["sort"] = sort
            if highlight:
                body["highlight"] = highlight
            if aggs:
                body["aggs"] = aggs
            if source is not None:
                body["_source"] = source

            # Perform search
            response = await self._client.search(index=index_name, body=body)

            # Extract hits
            hits = response["hits"]["hits"]
            documents = []
            highlights = []

            for hit in hits:
                documents.append({
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "source": hit["_source"],
                    "index": hit["_index"],
                    "type": hit.get("_type")
                })

                if "highlight" in hit:
                    highlights.append({
                        "id": hit["_id"],
                        "highlight": hit["highlight"]
                    })

            # Extract aggregations
            aggregations = response.get("aggregations", {})

            # Build aggregation results
            agg_results = []
            for agg_name, agg_result in aggregations.items():
                agg_results.append(SearchAggregation(
                    name=agg_name,
                    type=list(agg_result.keys())[0] if agg_result else "unknown",
                    result=agg_result
                ))

            return SearchResult(
                success=True,
                hits=documents,
                total=response["hits"]["total"]["value"],
                max_score=response["hits"]["max_score"],
                aggregations=agg_results,
                highlights=highlights,
                took=response["took"],
                timed_out=response["timed_out"]
            )

        except RequestError as e:
            error_msg = f"Search failed in {index_name}: {str(e)}"
            logger.error(error_msg)
            raise QueryException(error_msg, provider="elasticsearch") from e
        except Exception as e:
            error_msg = f"Unexpected error searching in {index_name}: {str(e)}"
            logger.error(error_msg)
            raise QueryException(error_msg, provider="elasticsearch") from e

    async def get_document(
        self,
        index_name: str,
        doc_id: str,
        source: Optional[Union[List[str], bool]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.

        Args:
            index_name: Index name
            doc_id: Document ID
            source: Source fields

        Returns:
            Optional[Dict[str, Any]]: Document or None

        Raises:
            QueryException: If get fails
        """
        try:
            if not self._connected:
                raise SearchConnectionException("Not connected to Elasticsearch", provider="elasticsearch")

            body = {}
            if source is not None:
                body["_source"] = source

            response = await self._client.get(index=index_name, id=doc_id, body=body)
            return response["_source"]

        except NotFoundError:
            return None
        except RequestError as e:
            error_msg = f"Failed to get document {doc_id} from {index_name}: {str(e)}"
            logger.error(error_msg)
            raise QueryException(error_msg, provider="elasticsearch") from e
        except Exception as e:
            error_msg = f"Unexpected error getting document {doc_id} from {index_name}: {str(e)}"
            logger.error(error_msg)
            raise QueryException(error_msg, provider="elasticsearch") from e

    async def update_document(
        self,
        index_name: str,
        doc_id: str,
        document: Dict[str, Any],
        refresh: bool = False
    ) -> bool:
        """
        Update a document.

        Args:
            index_name: Index name
            doc_id: Document ID
            document: Document updates
            refresh: Refresh index

        Returns:
            bool: True if update successful

        Raises:
            QueryException: If update fails
        """
        try:
            if not self._connected:
                raise SearchConnectionException("Not connected to Elasticsearch", provider="elasticsearch")

            await self._client.update(
                index=index_name,
                id=doc_id,
                body={"doc": document},
                refresh=refresh
            )

            logger.debug(f"Updated document {doc_id} in {index_name}")
            return True

        except NotFoundError:
            logger.warning(f"Document {doc_id} not found in {index_name}")
            return False
        except RequestError as e:
            error_msg = f"Failed to update document {doc_id} in {index_name}: {str(e)}"
            logger.error(error_msg)
            raise QueryException(error_msg, provider="elasticsearch") from e
        except Exception as e:
            error_msg = f"Unexpected error updating document {doc_id} in {index_name}: {str(e)}"
            logger.error(error_msg)
            raise QueryException(error_msg, provider="elasticsearch") from e

    async def delete_document(
        self,
        index_name: str,
        doc_id: str,
        refresh: bool = False
    ) -> bool:
        """
        Delete a document.

        Args:
            index_name: Index name
            doc_id: Document ID
            refresh: Refresh index

        Returns:
            bool: True if deletion successful

        Raises:
            QueryException: If delete fails
        """
        try:
            if not self._connected:
                raise SearchConnectionException("Not connected to Elasticsearch", provider="elasticsearch")

            await self._client.delete(
                index=index_name,
                id=doc_id,
                refresh=refresh
            )

            logger.debug(f"Deleted document {doc_id} from {index_name}")
            return True

        except NotFoundError:
            logger.warning(f"Document {doc_id} not found in {index_name}")
            return False
        except RequestError as e:
            error_msg = f"Failed to delete document {doc_id} from {index_name}: {str(e)}"
            logger.error(error_msg)
            raise QueryException(error_msg, provider="elasticsearch") from e
        except Exception as e:
            error_msg = f"Unexpected error deleting document {doc_id} from {index_name}: {str(e)}"
            logger.error(error_msg)
            raise QueryException(error_msg, provider="elasticsearch") from e

    def create_query_builder(self) -> ElasticsearchQuery:
        """
        Create a query builder.

        Returns:
            ElasticsearchQuery: Query builder
        """
        return ElasticsearchQuery()

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Elasticsearch cluster.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            if not self._connected:
                return {
                    "status": "unhealthy",
                    "provider": "elasticsearch",
                    "error": "Not connected to Elasticsearch"
                }

            start_time = asyncio.get_event_loop().time()

            # Get cluster health
            health = await self._client.cluster.health()

            # Get cluster info
            info = await self._client.info()

            end_time = asyncio.get_event_loop().time()
            response_time_ms = (end_time - start_time) * 1000

            return {
                "status": health["status"],
                "provider": "elasticsearch",
                "cluster_name": info["cluster_name"],
                "cluster_status": health["status"],
                "number_of_nodes": health["number_of_nodes"],
                "number_of_data_nodes": health["number_of_data_nodes"],
                "active_primary_shards": health["active_primary_shards"],
                "active_shards": health["active_shards"],
                "response_time_ms": response_time_ms,
                "timed_out": health["timed_out"]
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "elasticsearch",
                "error": str(e)
            }