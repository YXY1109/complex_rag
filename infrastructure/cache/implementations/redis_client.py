"""
Redis Cache Client Implementation

This module provides a Redis cache client that implements
the CacheInterface abstract base class.
"""

import asyncio
import json
import pickle
import time
from typing import Any, Dict, List, Optional, Union
import uuid
import zlib
from datetime import datetime, timezone

try:
    import aioredis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
except ImportError:
    aioredis = None
    RedisError = Exception
    RedisConnectionError = Exception

from ..interfaces.cache_interface import (
    CacheInterface,
    CacheConfig,
    CacheCapabilities,
    CacheType,
    CachePolicy,
    CacheEntry,
    CacheResult,
    CacheStats,
    CacheException,
    ConnectionException,
    SerializationException,
    KeyException,
    ValueException,
    PolicyException
)


class RedisCacheClient(CacheInterface):
    """
    Redis cache client implementation.

    Provides Redis-based caching with comprehensive error handling,
    performance optimization, and Redis-specific features.
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize Redis cache client with configuration.

        Args:
            config: Cache configuration with Redis-specific settings
        """
        super().__init__(config)

        # Validate aioredis availability
        if aioredis is None:
            raise ImportError("aioredis is required for Redis cache client")

        # Redis-specific configuration
        self.host = config.host or 'localhost'
        self.port = config.port or 6379
        self.password = config.password
        self.database = config.database or 0

        # Redis client
        self._redis: Optional[aioredis.Redis] = None
        self._connection_pool: Optional[aioredis.ConnectionPool] = None
        self._connected = False

        # Serialization settings
        self._serialization_method = config.serialization.lower()
        self._compression = config.compression

    @property
    def capabilities(self) -> CacheCapabilities:
        """Get Redis cache capabilities."""
        return CacheCapabilities(
            provider="redis",
            cache_type=CacheType.REDIS,
            supported_data_types=[
                "string", "hash", "list", "set", "sorted_set",
                "bytes", "int", "float", "json", "pickle"
            ],
            max_key_length=512,  # Redis default
            max_value_size=512 * 1024 * 1024,  # 512MB
            max_memory=None,  # Configurable in Redis
            supports_ttl=True,
            supports_persistence=True,
            supports_replication=True,
            supports_sharding=True,
            supports_compression=self._compression,
            supports_encryption=False,  # Can be added with TLS
            supports_transactions=True,
            supports_pub_sub=True,
            supports_lua_scripts=True,
            supports_async_operations=True
        )

    async def connect(self) -> bool:
        """
        Connect to Redis service.

        Returns:
            bool: True if connection successful

        Raises:
            ConnectionException: If connection fails
        """
        try:
            # Create connection pool
            self._connection_pool = aioredis.ConnectionPool.from_url(
                f"redis://{self.host}:{self.port}/{self.database}",
                password=self.password,
                encoding='utf-8',
                decode_responses=False,  # Handle encoding manually
                max_connections=self.max_connections,
                socket_timeout=self.timeout,
                socket_connect_timeout=self.timeout
            )

            # Create Redis client
            self._redis = aioredis.Redis(connection_pool=self._connection_pool)

            # Test connection
            await self._redis.ping()

            self._connected = True
            return True

        except RedisConnectionError as e:
            raise ConnectionException(
                f"Redis connection failed: {str(e)}",
                provider=self.provider_name,
                error_code="CONNECTION_ERROR"
            )
        except Exception as e:
            raise ConnectionException(
                f"Unexpected Redis connection error: {str(e)}",
                provider=self.provider_name,
                error_code="UNKNOWN_ERROR"
            )

    async def disconnect(self) -> None:
        """Disconnect from Redis service."""
        if self._redis:
            await self._redis.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()
        self._redis = None
        self._connection_pool = None
        self._connected = False

    async def ping(self) -> bool:
        """
        Ping Redis service to check connectivity.

        Returns:
            bool: True if service is responsive

        Raises:
            ConnectionException: If ping fails
        """
        try:
            if not self._redis:
                raise ConnectionException(
                    "Not connected to Redis",
                    provider=self.provider_name,
                    error_code="NOT_CONNECTED"
                )

            result = await self._redis.ping()
            return result

        except RedisError as e:
            raise ConnectionException(
                f"Redis ping failed: {str(e)}",
                provider=self.provider_name,
                error_code="PING_ERROR"
            )
        except Exception as e:
            raise ConnectionException(
                f"Unexpected Redis ping error: {str(e)}",
                provider=self.provider_name,
                error_code="UNKNOWN_ERROR"
            )

    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value for storage.

        Args:
            value: Value to serialize

        Returns:
            bytes: Serialized data

        Raises:
            SerializationException: If serialization fails
        """
        try:
            if self._serialization_method == 'json':
                data = json.dumps(value, ensure_ascii=False).encode('utf-8')
            elif self._serialization_method == 'pickle':
                data = pickle.dumps(value)
            elif self._serialization_method == 'str':
                data = str(value).encode('utf-8')
            else:
                data = pickle.dumps(value)

            # Apply compression if enabled
            if self._compression:
                data = zlib.compress(data)

            return data

        except Exception as e:
            raise SerializationException(
                f"Failed to serialize value: {str(e)}",
                provider=self.provider_name,
                error_code="SERIALIZATION_ERROR"
            )

    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize value from storage.

        Args:
            data: Serialized data

        Returns:
            Any: Deserialized value

        Raises:
            SerializationException: If deserialization fails
        """
        try:
            # Apply decompression if needed
            if self._compression and isinstance(data, bytes):
                data = zlib.decompress(data)

            if self._serialization_method == 'json':
                return json.loads(data.decode('utf-8'))
            elif self._serialization_method == 'pickle':
                return pickle.loads(data)
            elif self._serialization_method == 'str':
                return data.decode('utf-8')
            else:
                return pickle.loads(data)

        except Exception as e:
            raise SerializationException(
                f"Failed to deserialize value: {str(e)}",
                provider=self.provider_name,
                error_code="DESERIALIZATION_ERROR"
            )

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CacheResult:
        """
        Set a value in Redis cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            metadata: Additional metadata (optional)

        Returns:
            CacheResult: Operation result

        Raises:
            CacheException: If set operation fails
        """
        if not key:
            raise KeyException(
                "Cache key cannot be empty",
                provider=self.provider_name,
                error_code="EMPTY_KEY"
            )

        if not self._redis:
            raise ConnectionException(
                "Not connected to Redis",
                provider=self.provider_name,
                error_code="NOT_CONNECTED"
            )

        try:
            start_time = time.time()

            # Format key
            formatted_key = self._format_key(key)

            # Serialize value
            serialized_value = self._serialize(value)

            # Set value with TTL
            if ttl is not None:
                success = await self._redis.setex(formatted_key, ttl, serialized_value)
            else:
                success = await self._redis.set(formatted_key, serialized_value)

            # Store metadata if provided
            if metadata and success:
                metadata_key = f"{formatted_key}:metadata"
                metadata_value = self._serialize(metadata)
                metadata_ttl = ttl or self.default_ttl
                await self._redis.setex(metadata_key, metadata_ttl, metadata_value)

            operation_time = (time.time() - start_time) * 1000

            self._update_stats(False, operation_time)

            return CacheResult(
                success=bool(success),
                key=key,
                hit=False,
                ttl=ttl or self.default_ttl,
                operation_time_ms=operation_time
            )

        except RedisError as e:
            raise CacheException(
                f"Redis set operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="SET_ERROR"
            )
        except SerializationException:
            raise
        except Exception as e:
            raise CacheException(
                f"Unexpected Redis set error: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="UNKNOWN_ERROR"
            )

    async def get(self, key: str) -> CacheResult:
        """
        Get a value from Redis cache.

        Args:
            key: Cache key

        Returns:
            CacheResult: Operation result

        Raises:
            CacheException: If get operation fails
        """
        if not key:
            raise KeyException(
                "Cache key cannot be empty",
                provider=self.provider_name,
                error_code="EMPTY_KEY"
            )

        if not self._redis:
            raise ConnectionException(
                "Not connected to Redis",
                provider=self.provider_name,
                error_code="NOT_CONNECTED"
            )

        try:
            start_time = time.time()

            # Format key
            formatted_key = self._format_key(key)

            # Get value
            data = await self._redis.get(formatted_key)

            if data is None:
                operation_time = (time.time() - start_time) * 1000
                self._update_stats(False, operation_time)

                return CacheResult(
                    success=True,
                    key=key,
                    hit=False,
                    operation_time_ms=operation_time
                )

            # Deserialize value
            value = self._deserialize(data)

            # Get metadata if exists
            metadata = None
            metadata_key = f"{formatted_key}:metadata"
            metadata_data = await self._redis.get(metadata_key)
            if metadata_data:
                try:
                    metadata = self._deserialize(metadata_data)
                except:
                    pass  # Ignore metadata errors

            # Get remaining TTL
            remaining_ttl = await self._redis.ttl(formatted_key)
            if remaining_ttl == -1:
                remaining_ttl = None  # No expiration

            operation_time = (time.time() - start_time) * 1000
            self._update_stats(True, operation_time)

            return CacheResult(
                success=True,
                key=key,
                value=value,
                hit=True,
                ttl=remaining_ttl,
                metadata=metadata,
                operation_time_ms=operation_time
            )

        except RedisError as e:
            raise CacheException(
                f"Redis get operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="GET_ERROR"
            )
        except SerializationException:
            raise
        except Exception as e:
            raise CacheException(
                f"Unexpected Redis get error: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="UNKNOWN_ERROR"
            )

    async def delete(self, key: str) -> CacheResult:
        """
        Delete a value from Redis cache.

        Args:
            key: Cache key

        Returns:
            CacheResult: Operation result

        Raises:
            CacheException: If delete operation fails
        """
        if not key:
            raise KeyException(
                "Cache key cannot be empty",
                provider=self.provider_name,
                error_code="EMPTY_KEY"
            )

        if not self._redis:
            raise ConnectionException(
                "Not connected to Redis",
                provider=self.provider_name,
                error_code="NOT_CONNECTED"
            )

        try:
            start_time = time.time()

            # Format key
            formatted_key = self._format_key(key)

            # Delete key and metadata
            pipe = self._redis.pipeline()
            pipe.delete(formatted_key)
            pipe.delete(f"{formatted_key}:metadata")
            results = await pipe.execute()

            operation_time = (time.time() - start_time) * 1000

            return CacheResult(
                success=bool(results[0]),  # Key deletion result
                key=key,
                operation_time_ms=operation_time
            )

        except RedisError as e:
            raise CacheException(
                f"Redis delete operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="DELETE_ERROR"
            )
        except Exception as e:
            raise CacheException(
                f"Unexpected Redis delete error: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="UNKNOWN_ERROR"
            )

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis cache.

        Args:
            key: Cache key

        Returns:
            bool: True if key exists

        Raises:
            CacheException: If check fails
        """
        if not key:
            raise KeyException(
                "Cache key cannot be empty",
                provider=self.provider_name,
                error_code="EMPTY_KEY"
            )

        if not self._redis:
            raise ConnectionException(
                "Not connected to Redis",
                provider=self.provider_name,
                error_code="NOT_CONNECTED"
            )

        try:
            formatted_key = self._format_key(key)
            return bool(await self._redis.exists(formatted_key))

        except RedisError as e:
            raise CacheException(
                f"Redis exists operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="EXISTS_ERROR"
            )
        except Exception as e:
            raise CacheException(
                f"Unexpected Redis exists error: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="UNKNOWN_ERROR"
            )

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set TTL for an existing key in Redis.

        Args:
            key: Cache key
            ttl: Time to live in seconds

        Returns:
            bool: True if TTL set successfully

        Raises:
            CacheException: If TTL operation fails
        """
        if not key:
            raise KeyException(
                "Cache key cannot be empty",
                provider=self.provider_name,
                error_code="EMPTY_KEY"
            )

        if not self._redis:
            raise ConnectionException(
                "Not connected to Redis",
                provider=self.provider_name,
                error_code="NOT_CONNECTED"
            )

        try:
            formatted_key = self._format_key(key)
            return bool(await self._redis.expire(formatted_key, ttl))

        except RedisError as e:
            raise CacheException(
                f"Redis expire operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="EXPIRE_ERROR"
            )
        except Exception as e:
            raise CacheException(
                f"Unexpected Redis expire error: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="UNKNOWN_ERROR"
            )

    async def ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key in Redis.

        Args:
            key: Cache key

        Returns:
            Optional[int]: Remaining TTL in seconds, None if no TTL or key doesn't exist

        Raises:
            CacheException: If TTL operation fails
        """
        if not key:
            raise KeyException(
                "Cache key cannot be empty",
                provider=self.provider_name,
                error_code="EMPTY_KEY"
            )

        if not self._redis:
            raise ConnectionException(
                "Not connected to Redis",
                provider=self.provider_name,
                error_code="NOT_CONNECTED"
            )

        try:
            formatted_key = self._format_key(key)
            ttl = await self._redis.ttl(formatted_key)

            if ttl == -2:  # Key doesn't exist
                return None
            elif ttl == -1:  # No expiration
                return None
            else:
                return ttl

        except RedisError as e:
            raise CacheException(
                f"Redis TTL operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="TTL_ERROR"
            )
        except Exception as e:
            raise CacheException(
                f"Unexpected Redis TTL error: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="UNKNOWN_ERROR"
            )

    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get all keys matching a pattern in Redis.

        Args:
            pattern: Key pattern (optional, supports wildcards)

        Returns:
            List[str]: Matching keys

        Raises:
            CacheException: If keys operation fails
        """
        if not self._redis:
            raise ConnectionException(
                "Not connected to Redis",
                provider=self.provider_name,
                error_code="NOT_CONNECTED"
            )

        try:
            search_pattern = pattern or "*"
            formatted_pattern = self._format_key(search_pattern)

            # Remove prefix from returned keys
            keys = await self._redis.keys(formatted_pattern)
            if self.key_prefix:
                prefix_len = len(self.key_prefix) + 1  # +1 for ':'
                return [key.decode('utf-8')[prefix_len:] for key in keys if key]
            else:
                return [key.decode('utf-8') for key in keys if key]

        except RedisError as e:
            raise CacheException(
                f"Redis keys operation failed: {str(e)}",
                provider=self.provider_name,
                error_code="KEYS_ERROR"
            )
        except Exception as e:
            raise CacheException(
                f"Unexpected Redis keys error: {str(e)}",
                provider=self.provider_name,
                error_code="UNKNOWN_ERROR"
            )

    async def clear(self) -> CacheResult:
        """
        Clear all cache entries in Redis.

        Returns:
            CacheResult: Operation result

        Raises:
            CacheException: If clear operation fails
        """
        if not self._redis:
            raise ConnectionException(
                "Not connected to Redis",
                provider=self.provider_name,
                error_code="NOT_CONNECTED"
            )

        try:
            start_time = time.time()

            # If key prefix is set, only delete keys with that prefix
            if self.key_prefix:
                pattern = f"{self.key_prefix}:*"
                keys = await self._redis.keys(pattern)
                if keys:
                    await self._redis.delete(*keys)
                    cleared_count = len(keys)
                else:
                    cleared_count = 0
            else:
                # Clear entire database
                cleared_count = await self._redis.flushdb()

            operation_time = (time.time() - start_time) * 1000

            return CacheResult(
                success=True,
                key="*",
                operation_time_ms=operation_time,
                metadata={"cleared_count": cleared_count}
            )

        except RedisError as e:
            raise CacheException(
                f"Redis clear operation failed: {str(e)}",
                provider=self.provider_name,
                error_code="CLEAR_ERROR"
            )
        except Exception as e:
            raise CacheException(
                f"Unexpected Redis clear error: {str(e)}",
                provider=self.provider_name,
                error_code="UNKNOWN_ERROR"
            )

    async def size(self) -> int:
        """
        Get the number of entries in Redis cache.

        Returns:
            int: Number of entries

        Raises:
            CacheException: If size operation fails
        """
        if not self._redis:
            raise ConnectionException(
                "Not connected to Redis",
                provider=self.provider_name,
                error_code="NOT_CONNECTED"
            )

        try:
            if self.key_prefix:
                # Count keys with prefix
                pattern = f"{self.key_prefix}:*"
                keys = await self._redis.keys(pattern)
                return len(keys)
            else:
                # Get database size
                return await self._redis.dbsize()

        except RedisError as e:
            raise CacheException(
                f"Redis size operation failed: {str(e)}",
                provider=self.provider_name,
                error_code="SIZE_ERROR"
            )
        except Exception as e:
            raise CacheException(
                f"Unexpected Redis size error: {str(e)}",
                provider=self.provider_name,
                error_code="UNKNOWN_ERROR"
            )

    async def mget(self, keys: List[str]) -> Dict[str, CacheResult]:
        """
        Get multiple values from Redis cache efficiently.

        Args:
            keys: List of cache keys

        Returns:
            Dict[str, CacheResult]: Results for each key

        Raises:
            CacheException: If mget operation fails
        """
        if not keys:
            return {}

        if not self._redis:
            raise ConnectionException(
                "Not connected to Redis",
                provider=self.provider_name,
                error_code="NOT_CONNECTED"
            )

        try:
            start_time = time.time()

            # Format keys
            formatted_keys = [self._format_key(key) for key in keys]

            # Get values in batch
            values = await self._redis.mget(formatted_keys)

            # Process results
            results = {}
            for i, key in enumerate(keys):
                try:
                    if values[i] is not None:
                        value = self._deserialize(values[i])
                        results[key] = CacheResult(
                            success=True,
                            key=key,
                            value=value,
                            hit=True
                        )
                    else:
                        results[key] = CacheResult(
                            success=True,
                            key=key,
                            hit=False
                        )
                except Exception as e:
                    results[key] = CacheResult(
                        success=False,
                        key=key,
                        error=str(e)
                    )

            operation_time = (time.time() - start_time) * 1000

            # Update stats
            hits = sum(1 for r in results.values() if r.hit)
            self._update_stats(hits > 0, operation_time)

            return results

        except RedisError as e:
            raise CacheException(
                f"Redis mget operation failed: {str(e)}",
                provider=self.provider_name,
                error_code="MGET_ERROR"
            )
        except Exception as e:
            raise CacheException(
                f"Unexpected Redis mget error: {str(e)}",
                provider=self.provider_name,
                error_code="UNKNOWN_ERROR"
            )

    async def mset(self, items: Dict[str, Any], ttl: Optional[int] = None) -> Dict[str, CacheResult]:
        """
        Set multiple values in Redis cache efficiently.

        Args:
            items: Dictionary of key-value pairs
            ttl: Default TTL for all items (optional)

        Returns:
            Dict[str, CacheResult]: Results for each key

        Raises:
            CacheException: If mset operation fails
        """
        if not items:
            return {}

        if not self._redis:
            raise ConnectionException(
                "Not connected to Redis",
                provider=self.provider_name,
                error_code="NOT_CONNECTED"
            )

        try:
            start_time = time.time()

            # Prepare items
            formatted_items = {}
            serialized_items = {}

            for key, value in items.items():
                formatted_key = self._format_key(key)
                serialized_value = self._serialize(value)
                formatted_items[formatted_key] = serialized_value

            # Set values in batch
            success = await self._redis.mset(formatted_items)

            # Set TTL if provided
            if ttl is not None and success:
                pipe = self._redis.pipeline()
                for formatted_key in formatted_items.keys():
                    pipe.expire(formatted_key, ttl)
                await pipe.execute()

            operation_time = (time.time() - start_time) * 1000

            # Process results
            results = {}
            for key in items.keys():
                results[key] = CacheResult(
                    success=bool(success),
                    key=key,
                    hit=False,
                    ttl=ttl,
                    operation_time_ms=operation_time
                )

            self._update_stats(False, operation_time)

            return results

        except RedisError as e:
            raise CacheException(
                f"Redis mset operation failed: {str(e)}",
                provider=self.provider_name,
                error_code="MSET_ERROR"
            )
        except SerializationException:
            raise
        except Exception as e:
            raise CacheException(
                f"Unexpected Redis mset error: {str(e)}",
                provider=self.provider_name,
                error_code="UNKNOWN_ERROR"
            )

    async def increment(self, key: str, amount: int = 1) -> CacheResult:
        """
        Increment a numeric value in Redis.

        Args:
            key: Cache key
            amount: Increment amount

        Returns:
            CacheResult: Operation result with new value

        Raises:
            CacheException: If increment operation fails
        """
        if not key:
            raise KeyException(
                "Cache key cannot be empty",
                provider=self.provider_name,
                error_code="EMPTY_KEY"
            )

        if not self._redis:
            raise ConnectionException(
                "Not connected to Redis",
                provider=self.provider_name,
                error_code="NOT_CONNECTED"
            )

        try:
            start_time = time.time()

            formatted_key = self._format_key(key)
            new_value = await self._redis.incrby(formatted_key, amount)

            operation_time = (time.time() - start_time) * 1000

            return CacheResult(
                success=True,
                key=key,
                value=new_value,
                hit=True,
                operation_time_ms=operation_time
            )

        except RedisError as e:
            raise CacheException(
                f"Redis increment operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="INCREMENT_ERROR"
            )
        except Exception as e:
            raise CacheException(
                f"Unexpected Redis increment error: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="UNKNOWN_ERROR"
            )


# Export Redis cache client
__all__ = ['RedisCacheClient']