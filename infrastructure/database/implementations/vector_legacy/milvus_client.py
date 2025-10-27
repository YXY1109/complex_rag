"""
Milvus Vector Database Client Implementation

This module implements the Milvus client for vector database operations.
Based on the vector database interface abstract class.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import json
import numpy as np

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException,
    MilvusUnavailableException
)
from pydantic import BaseModel, Field

from ...interfaces.vector_db_interface import (
    VectorDBInterface,
    VectorDBConfig,
    VectorDBCapabilities,
    VectorIndexConfig,
    VectorIndexType,
    DistanceMetric,
    VectorSearchResult,
    VectorUpsertResult,
    VectorQueryResult,
    CollectionInfo,
    VectorDBException,
    ConnectionException as VectorConnectionException,
    CollectionException,
    IndexException,
    SearchException
)


class MilvusConfig(VectorDBConfig):
    """Milvus-specific configuration."""

    host: str = Field(default="localhost", description="Milvus host address")
    port: int = Field(default=19530, description="Milvus port")
    alias: str = Field(default="default", description="Connection alias")
    user: Optional[str] = Field(default=None, description="Milvus username")
    password: Optional[str] = Field(default=None, description="Milvus password")
    database: str = Field(default="default", description="Database name")

    # Connection settings
    timeout: int = Field(default=10, description="Connection timeout in seconds")
    secure: bool = Field(default=False, description="Use secure connection")
    server_pem_path: Optional[str] = Field(default=None, description="Server PEM path for secure connection")
    server_name: Optional[str] = Field(default=None, description="Server name for secure connection")
    client_pem_path: Optional[str] = Field(default=None, description="Client PEM path for secure connection")
    client_key_path: Optional[str] = Field(default=None, description="Client key path for secure connection")

    # Pool settings
    pool_size: int = Field(default=10, description="Connection pool size")
    pool_timeout: float = Field(default=10.0, description="Pool timeout in seconds")


class MilvusIndexConfig(VectorIndexConfig):
    """Milvus-specific index configuration."""

    def __init__(self, index_type: VectorIndexType, metric_type: DistanceMetric, **kwargs):
        super().__init__(index_type, metric_type, **kwargs)

        # Milvus-specific index parameters
        self.milvus_params = self._get_milvus_index_params()

    def _get_milvus_index_params(self) -> Dict[str, Any]:
        """Get Milvus-specific index parameters."""
        params = {}

        if self.index_type == VectorIndexType.FLAT:
            params = {"nlist": 128} if self.dimension > 128 else {}

        elif self.index_type == VectorIndexType.IVF_FLAT:
            params = {
                "nlist": min(self.dimension * 4, 16384),
                "metric_type": self._get_milvus_metric_type()
            }

        elif self.index_type == VectorIndexType.IVF_SQ8:
            params = {
                "nlist": min(self.dimension * 4, 16384),
                "metric_type": self._get_milvus_metric_type()
            }

        elif self.index_type == VectorIndexType.IVF_PQ:
            params = {
                "nlist": min(self.dimension * 4, 16384),
                "m": min(16, self.dimension // 4),
                "metric_type": self._get_milvus_metric_type()
            }

        elif self.index_type == VectorIndexType.HNSW:
            params = {
                "M": min(16, self.dimension // 2),
                "efConstruction": min(500, self.dimension * 10),
                "metric_type": self._get_milvus_metric_type()
            }

        elif self.index_type == VectorIndexType.ANNOY:
            params = {
                "n_trees": min(32, self.dimension * 2),
                "metric_type": self._get_milvus_metric_type()
            }

        elif self.index_type == VectorIndexType.RHNSW_FLAT:
            params = {
                "M": min(16, self.dimension // 2),
                "efConstruction": min(500, self.dimension * 10),
                "metric_type": self._get_milvus_metric_type()
            }

        elif self.index_type == VectorIndexType.RHNSW_SQ:
            params = {
                "M": min(16, self.dimension // 2),
                "efConstruction": min(500, self.dimension * 10),
                "metric_type": self._get_milvus_metric_type()
            }

        elif self.index_type == VectorIndexType.RHNSW_PQ:
            params = {
                "M": min(16, self.dimension // 2),
                "efConstruction": min(500, self.dimension * 10),
                "PQM": min(64, self.dimension // 4),
                "metric_type": self._get_milvus_metric_type()
            }

        # Add custom parameters
        params.update(self.custom_params)

        return params

    def _get_milvus_metric_type(self) -> str:
        """Convert distance metric to Milvus metric type."""
        metric_mapping = {
            DistanceMetric.COSINE: "COSINE",
            DistanceMetric.EUCLIDEAN: "L2",
            DistanceMetric.DOT_PRODUCT: "IP"  # Inner Product
        }
        return metric_mapping.get(self.metric_type, "L2")


class MilvusCapabilities(VectorDBCapabilities):
    """Milvus-specific capabilities."""

    def __init__(self):
        super().__init__(
            provider="milvus",
            supported_index_types=[
                VectorIndexType.FLAT,
                VectorIndexType.IVF_FLAT,
                VectorIndexType.IVF_SQ8,
                VectorIndexType.IVF_PQ,
                VectorIndexType.HNSW,
                VectorIndexType.ANNOY,
                VectorIndexType.RHNSW_FLAT,
                VectorIndexType.RHNSW_SQ,
                VectorIndexType.RHNSW_PQ
            ],
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.DOT_PRODUCT
            ],
            max_dimension=32768,
            supports_async=True,
            supports_batch_operations=True,
            supports_metadata=True,
            supports_filtering=True,
            supports_hybrid_search=True,
            supports_multi_vector=False,
            supports_streaming=False,
            supports_backup=True,
            supports_partitioning=True,
            supports_sharding=True,
            supports_mmap=True
        )


class MilvusClient(VectorDBInterface):
    """
    Milvus client implementation for vector database operations.

    Provides async Milvus operations with comprehensive error handling,
    connection management, and advanced search capabilities.
    """

    def __init__(self, config: MilvusConfig):
        super().__init__(config)
        self.config: MilvusConfig = config
        self._connected = False
        self._capabilities = MilvusCapabilities()
        self._collections: Dict[str, Collection] = {}

    @property
    def capabilities(self) -> VectorDBCapabilities:
        """Get Milvus capabilities."""
        return self._capabilities

    async def connect(self) -> bool:
        """
        Connect to Milvus server.

        Returns:
            bool: True if connection successful

        Raises:
            VectorConnectionException: If connection fails
        """
        try:
            # Build connection parameters
            connect_params = {
                "host": self.config.host,
                "port": self.config.port,
                "alias": self.config.alias,
                "timeout": self.config.timeout
            }

            if self.config.user:
                connect_params["user"] = self.config.user
            if self.config.password:
                connect_params["password"] = self.config.password
            if self.config.database:
                connect_params["db_name"] = self.config.database
            if self.config.secure:
                connect_params["secure"] = True
                if self.config.server_pem_path:
                    connect_params["server_pem_path"] = self.config.server_pem_path
                if self.config.server_name:
                    connect_params["server_name"] = self.config.server_name
                if self.config.client_pem_path:
                    connect_params["client_pem_path"] = self.config.client_pem_path
                if self.config.client_key_path:
                    connect_params["client_key_path"] = self.config.client_key_path

            # Connect to Milvus
            connections.connect(**connect_params)

            # Test connection by checking server version
            server_version = utility.get_server_version(using=self.config.alias)

            self._connected = True
            logger.info(f"Connected to Milvus server: {self.config.host}:{self.config.port}, version: {server_version}")
            return True

        except MilvusUnavailableException as e:
            error_msg = f"Milvus server unavailable: {str(e)}"
            logger.error(error_msg)
            raise VectorConnectionException(error_msg, provider="milvus") from e
        except MilvusException as e:
            error_msg = f"Failed to connect to Milvus: {str(e)}"
            logger.error(error_msg)
            raise VectorConnectionException(error_msg, provider="milvus") from e
        except Exception as e:
            error_msg = f"Unexpected error connecting to Milvus: {str(e)}"
            logger.error(error_msg)
            raise VectorConnectionException(error_msg, provider="milvus") from e

    async def disconnect(self) -> None:
        """Disconnect from Milvus server."""
        try:
            if self._connected:
                connections.disconnect(alias=self.config.alias)
                self._connected = False
                self._collections.clear()
                logger.info("Disconnected from Milvus server")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {str(e)}")

    async def create_collection(
        self,
        name: str,
        dimension: int,
        index_config: Optional[VectorIndexConfig] = None,
        metadata_fields: Optional[List[Dict[str, Any]]] = None,
        enable_dynamic_fields: bool = True
    ) -> bool:
        """
        Create a new collection.

        Args:
            name: Collection name
            dimension: Vector dimension
            index_config: Index configuration
            metadata_fields: Additional metadata fields
            enable_dynamic_fields: Enable dynamic JSON field

        Returns:
            bool: True if creation successful

        Raises:
            CollectionException: If collection creation fails
        """
        try:
            if not self._connected:
                raise VectorConnectionException("Not connected to Milvus", provider="milvus")

            # Check if collection already exists
            if utility.has_collection(name, using=self.config.alias):
                logger.warning(f"Collection {name} already exists")
                return True

            # Create field schemas
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=65535, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]

            # Add metadata fields
            if metadata_fields:
                for field in metadata_fields:
                    field_type = self._get_field_type(field["type"])
                    field_schema = FieldSchema(
                        name=field["name"],
                        dtype=field_type,
                        **{k: v for k, v in field.items() if k not in ["name", "type"]}
                    )
                    fields.append(field_schema)

            # Add dynamic field if enabled
            if enable_dynamic_fields:
                fields.append(FieldSchema(name="metadata", dtype=DataType.JSON))

            # Create collection schema
            schema = CollectionSchema(fields, f"Collection {name}")

            # Create collection
            collection = Collection(name, schema, using=self.config.alias)
            self._collections[name] = collection

            logger.info(f"Created Milvus collection: {name} with dimension {dimension}")
            return True

        except MilvusException as e:
            error_msg = f"Failed to create collection {name}: {str(e)}"
            logger.error(error_msg)
            raise CollectionException(error_msg, provider="milvus") from e
        except Exception as e:
            error_msg = f"Unexpected error creating collection {name}: {str(e)}"
            logger.error(error_msg)
            raise CollectionException(error_msg, provider="milvus") from e

    async def drop_collection(self, name: str) -> bool:
        """
        Drop a collection.

        Args:
            name: Collection name

        Returns:
            bool: True if deletion successful

        Raises:
            CollectionException: If collection deletion fails
        """
        try:
            if not self._connected:
                raise VectorConnectionException("Not connected to Milvus", provider="milvus")

            if not utility.has_collection(name, using=self.config.alias):
                logger.warning(f"Collection {name} does not exist")
                return True

            utility.drop_collection(name, using=self.config.alias)

            if name in self._collections:
                del self._collections[name]

            logger.info(f"Dropped Milvus collection: {name}")
            return True

        except MilvusException as e:
            error_msg = f"Failed to drop collection {name}: {str(e)}"
            logger.error(error_msg)
            raise CollectionException(error_msg, provider="milvus") from e
        except Exception as e:
            error_msg = f"Unexpected error dropping collection {name}: {str(e)}"
            logger.error(error_msg)
            raise CollectionException(error_msg, provider="milvus") from e

    async def list_collections(self) -> List[str]:
        """
        List all collections.

        Returns:
            List[str]: Collection names

        Raises:
            CollectionException: If listing fails
        """
        try:
            if not self._connected:
                raise VectorConnectionException("Not connected to Milvus", provider="milvus")

            return utility.list_collections(using=self.config.alias)

        except MilvusException as e:
            error_msg = f"Failed to list collections: {str(e)}"
            logger.error(error_msg)
            raise CollectionException(error_msg, provider="milvus") from e
        except Exception as e:
            error_msg = f"Unexpected error listing collections: {str(e)}"
            logger.error(error_msg)
            raise CollectionException(error_msg, provider="milvus") from e

    async def get_collection_info(self, name: str) -> CollectionInfo:
        """
        Get collection information.

        Args:
            name: Collection name

        Returns:
            CollectionInfo: Collection information

        Raises:
            CollectionException: If getting info fails
        """
        try:
            if not self._connected:
                raise VectorConnectionException("Not connected to Milvus", provider="milvus")

            if not utility.has_collection(name, using=self.config.alias):
                raise CollectionException(f"Collection {name} does not exist", provider="milvus")

            collection = self._get_collection(name)

            # Get statistics
            stats = collection.describe()

            # Get row count
            collection.load()
            num_entities = collection.num_entities

            return CollectionInfo(
                name=name,
                dimension=next((f["params"]["dim"] for f in stats["fields"] if f["name"] == "vector"), 0),
                row_count=num_entities,
                index_info=stats.get("indexes", []),
                fields=stats.get("fields", []),
                created_time=datetime.fromtimestamp(stats.get("created_utc_timestamp", 0) / 1000),
                properties=stats
            )

        except MilvusException as e:
            error_msg = f"Failed to get collection info for {name}: {str(e)}"
            logger.error(error_msg)
            raise CollectionException(error_msg, provider="milvus") from e
        except Exception as e:
            error_msg = f"Unexpected error getting collection info for {name}: {str(e)}"
            logger.error(error_msg)
            raise CollectionException(error_msg, provider="milvus") from e

    async def create_index(
        self,
        collection_name: str,
        index_config: VectorIndexConfig,
        field_name: str = "vector"
    ) -> bool:
        """
        Create index on collection.

        Args:
            collection_name: Collection name
            index_config: Index configuration
            field_name: Field name to index

        Returns:
            bool: True if index creation successful

        Raises:
            IndexException: If index creation fails
        """
        try:
            if not self._connected:
                raise VectorConnectionException("Not connected to Milvus", provider="milvus")

            collection = self._get_collection(collection_name)

            # Convert to Milvus index config
            milvus_config = MilvusIndexConfig(
                index_type=index_config.index_type,
                metric_type=index_config.metric_type,
                dimension=index_config.dimension,
                custom_params=index_config.custom_params
            )

            # Create index
            collection.create_index(
                field_name=field_name,
                index_params=milvus_config.milvus_params,
                index_name=index_config.name or f"{field_name}_index"
            )

            logger.info(f"Created index on {collection_name}.{field_name}")
            return True

        except MilvusException as e:
            error_msg = f"Failed to create index on {collection_name}: {str(e)}"
            logger.error(error_msg)
            raise IndexException(error_msg, provider="milvus") from e
        except Exception as e:
            error_msg = f"Unexpected error creating index on {collection_name}: {str(e)}"
            logger.error(error_msg)
            raise IndexException(error_msg, provider="milvus") from e

    async def insert(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        partition_name: Optional[str] = None
    ) -> VectorUpsertResult:
        """
        Insert vectors into collection.

        Args:
            collection_name: Collection name
            ids: Vector IDs
            vectors: Vector data
            metadata: Metadata for each vector
            partition_name: Partition name

        Returns:
            VectorUpsertResult: Insert result

        Raises:
            VectorDBException: If insert fails
        """
        return await self._upsert_vectors(
            collection_name=collection_name,
            ids=ids,
            vectors=vectors,
            metadata=metadata,
            partition_name=partition_name,
            is_insert=True
        )

    async def upsert(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        partition_name: Optional[str] = None
    ) -> VectorUpsertResult:
        """
        Upsert vectors into collection.

        Args:
            collection_name: Collection name
            ids: Vector IDs
            vectors: Vector data
            metadata: Metadata for each vector
            partition_name: Partition name

        Returns:
            VectorUpsertResult: Upsert result

        Raises:
            VectorDBException: If upsert fails
        """
        return await self._upsert_vectors(
            collection_name=collection_name,
            ids=ids,
            vectors=vectors,
            metadata=metadata,
            partition_name=partition_name,
            is_insert=False
        )

    async def _upsert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        partition_name: Optional[str] = None,
        is_insert: bool = False
    ) -> VectorUpsertResult:
        """Internal method for insert/upsert operations."""
        try:
            if not self._connected:
                raise VectorConnectionException("Not connected to Milvus", provider="milvus")

            collection = self._get_collection(collection_name)

            # Validate input
            if len(ids) != len(vectors):
                raise VectorDBException("Length of ids and vectors must match", provider="milvus")

            if metadata and len(metadata) != len(vectors):
                raise VectorDBException("Length of metadata and vectors must match", provider="milvus")

            # Prepare data
            data = {
                "id": ids,
                "vector": vectors
            }

            if metadata:
                data["metadata"] = metadata

            # Load collection
            collection.load()

            # Perform insert/upsert
            if is_insert:
                result = collection.insert(data, partition_name=partition_name)
            else:
                result = collection.upsert(data, partition_name=partition_name)

            # Get insert count
            insert_count = result.insert_count if hasattr(result, 'insert_count') else len(ids)

            return VectorUpsertResult(
                success=True,
                insert_count=insert_count,
                ids=ids[:insert_count]
            )

        except MilvusException as e:
            error_msg = f"Failed to {'insert' if is_insert else 'upsert'} vectors in {collection_name}: {str(e)}"
            logger.error(error_msg)
            raise VectorDBException(error_msg, provider="milvus") from e
        except Exception as e:
            operation = "insert" if is_insert else "upsert"
            error_msg = f"Unexpected error {operation}ing vectors in {collection_name}: {str(e)}"
            logger.error(error_msg)
            raise VectorDBException(error_msg, provider="milvus") from e

    async def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
        partition_names: Optional[List[str]] = None,
        consistency_level: str = "Strong"
    ) -> VectorSearchResult:
        """
        Search vectors in collection.

        Args:
            collection_name: Collection name
            query_vectors: Query vectors
            top_k: Number of results to return
            filter_expr: Filter expression
            output_fields: Output fields
            search_params: Search parameters
            partition_names: Partition names
            consistency_level: Consistency level

        Returns:
            VectorSearchResult: Search results

        Raises:
            SearchException: If search fails
        """
        try:
            if not self._connected:
                raise VectorConnectionException("Not connected to Milvus", provider="milvus")

            collection = self._get_collection(collection_name)

            # Load collection
            collection.load()

            # Default search params
            default_search_params = {
                "metric_type": "L2",  # Will be overridden by index
                "params": {"nprobe": 10}
            }

            if search_params:
                default_search_params.update(search_params)

            # Prepare search parameters for each query vector
            search_params_list = [default_search_params] * len(query_vectors)

            # Perform search
            results = collection.search(
                data=query_vectors,
                anns_field="vector",
                param=search_params_list,
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields or ["id"],
                partition_names=partition_names,
                consistency_level=consistency_level
            )

            # Convert results
            search_results = []
            for i, query_result in enumerate(results):
                query_results = []
                for hit in query_result:
                    query_results.append({
                        "id": hit.id,
                        "score": hit.score,
                        "distance": hit.distance if hasattr(hit, 'distance') else None,
                        "metadata": hit.entity if hasattr(hit, 'entity') else {}
                    })
                search_results.append(query_results)

            return VectorSearchResult(
                success=True,
                results=search_results,
                total_queries=len(query_vectors)
            )

        except MilvusException as e:
            error_msg = f"Failed to search in collection {collection_name}: {str(e)}"
            logger.error(error_msg)
            raise SearchException(error_msg, provider="milvus") from e
        except Exception as e:
            error_msg = f"Unexpected error searching in collection {collection_name}: {str(e)}"
            logger.error(error_msg)
            raise SearchException(error_msg, provider="milvus") from e

    async def query(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        consistency_level: str = "Strong"
    ) -> VectorQueryResult:
        """
        Query vectors by filter expression.

        Args:
            collection_name: Collection name
            filter_expr: Filter expression
            output_fields: Output fields
            partition_names: Partition names
            limit: Result limit
            offset: Result offset
            consistency_level: Consistency level

        Returns:
            VectorQueryResult: Query results

        Raises:
            VectorDBException: If query fails
        """
        try:
            if not self._connected:
                raise VectorConnectionException("Not connected to Milvus", provider="milvus")

            collection = self._get_collection(collection_name)

            # Load collection
            collection.load()

            # Build query parameters
            query_params = {
                "expr": filter_expr,
                "output_fields": output_fields or ["id"],
                "consistency_level": consistency_level
            }

            if partition_names:
                query_params["partition_names"] = partition_names

            # Perform query
            results = collection.query(**query_params)

            # Apply limit and offset if specified
            if limit is not None:
                start_idx = offset or 0
                end_idx = start_idx + limit
                results = results[start_idx:end_idx]

            # Convert results
            query_results = []
            for result in results:
                query_results.append(dict(result))

            return VectorQueryResult(
                success=True,
                results=query_results,
                total_count=len(query_results)
            )

        except MilvusException as e:
            error_msg = f"Failed to query collection {collection_name}: {str(e)}"
            logger.error(error_msg)
            raise VectorDBException(error_msg, provider="milvus") from e
        except Exception as e:
            error_msg = f"Unexpected error querying collection {collection_name}: {str(e)}"
            logger.error(error_msg)
            raise VectorDBException(error_msg, provider="milvus") from e

    async def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        partition_name: Optional[str] = None
    ) -> int:
        """
        Delete vectors from collection.

        Args:
            collection_name: Collection name
            ids: Vector IDs to delete
            filter_expr: Filter expression
            partition_name: Partition name

        Returns:
            int: Number of deleted vectors

        Raises:
            VectorDBException: If delete fails
        """
        try:
            if not self._connected:
                raise VectorConnectionException("Not connected to Milvus", provider="milvus")

            collection = self._get_collection(collection_name)

            # Prepare delete expression
            if ids:
                # Build ID filter
                id_strs = [f'"{id_}"' for id_ in ids]
                delete_expr = f'id in [{", ".join(id_strs)}]'
            elif filter_expr:
                delete_expr = filter_expr
            else:
                raise VectorDBException("Either ids or filter_expr must be provided", provider="milvus")

            # Perform delete
            result = collection.delete(
                expr=delete_expr,
                partition_name=partition_name
            )

            delete_count = result.delete_count if hasattr(result, 'delete_count') else 0
            logger.info(f"Deleted {delete_count} vectors from {collection_name}")
            return delete_count

        except MilvusException as e:
            error_msg = f"Failed to delete from collection {collection_name}: {str(e)}"
            logger.error(error_msg)
            raise VectorDBException(error_msg, provider="milvus") from e
        except Exception as e:
            error_msg = f"Unexpected error deleting from collection {collection_name}: {str(e)}"
            logger.error(error_msg)
            raise VectorDBException(error_msg, provider="milvus") from e

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Milvus connection.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            if not self._connected:
                return {
                    "status": "unhealthy",
                    "provider": "milvus",
                    "error": "Not connected to Milvus"
                }

            start_time = asyncio.get_event_loop().time()

            # Get server version
            server_version = utility.get_server_version(using=self.config.alias)

            # List collections to test connectivity
            collections = utility.list_collections(using=self.config.alias)

            end_time = asyncio.get_event_loop().time()
            response_time_ms = (end_time - start_time) * 1000

            return {
                "status": "healthy",
                "provider": "milvus",
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "server_version": server_version,
                "collection_count": len(collections),
                "response_time_ms": response_time_ms
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "milvus",
                "host": self.config.host,
                "port": self.config.port,
                "error": str(e)
            }

    def _get_collection(self, name: str) -> Collection:
        """Get collection instance."""
        if name in self._collections:
            return self._collections[name]

        if not utility.has_collection(name, using=self.config.alias):
            raise CollectionException(f"Collection {name} does not exist", provider="milvus")

        collection = Collection(name, using=self.config.alias)
        self._collections[name] = collection
        return collection

    def _get_field_type(self, field_type: str) -> DataType:
        """Convert string field type to Milvus DataType."""
        type_mapping = {
            "int64": DataType.INT64,
            "int": DataType.INT64,
            "float": DataType.FLOAT,
            "double": DataType.DOUBLE,
            "bool": DataType.BOOL,
            "varchar": DataType.VARCHAR,
            "text": DataType.VARCHAR,
            "json": DataType.JSON,
            "array": DataType.ARRAY
        }
        return type_mapping.get(field_type.lower(), DataType.VARCHAR)