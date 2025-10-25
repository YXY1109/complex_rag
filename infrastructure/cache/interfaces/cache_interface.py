"""
Cache Interface Abstract Class

This module defines the abstract interface for caching services.
All cache implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Set
from pydantic import BaseModel, Field
from enum import Enum
import time
from dataclasses import dataclass


class CacheType(str, Enum):
    """Cache types."""
    REDIS = "redis"
    MEMORY = "memory"
    MEMCACHED = "memcached"
    FILESYSTEM = "filesystem"
    CUSTOM = "custom"


class CachePolicy(str, Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    LIFO = "lifo"  # Last In First Out
    TTL = "ttl"  # Time To Live
    NO_EVICTION = "no_eviction"


@dataclass
class CacheEntry:
    """Cache entry representation."""
    key: str
    value: Any
    ttl: Optional[int] = None
    created_at: Optional[float] = None
    accessed_at: Optional[float] = None
    access_count: int = 0
    size: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CacheResult:
    """Cache operation result."""
    success: bool
    key: str
    value: Optional[Any] = None
    hit: bool = False
    ttl: Optional[int] = None
    operation_time_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class CacheStats:
    """Cache statistics."""
    cache_type: str
    total_keys: int
    hits: int
    misses: int
    hit_rate: float
    memory_usage: Optional[int] = None
    memory_limit: Optional[int] = None
    eviction_count: int = 0
    operations_count: int = 0
    avg_operation_time_ms: float = 0.0
    uptime_seconds: float = 0.0


class CacheConfig(BaseModel):
    """Cache configuration model."""
    provider: str
    cache_type: CacheType
    host: Optional[str] = Field(default=None, description="Cache server host")
    port: Optional[int] = Field(default=None, description="Cache server port")
    password: Optional[str] = Field(default=None, description="Cache server password")
    database: Optional[int] = Field(default=0, description="Cache database number")

    # Connection settings
    timeout: int = Field(default=5, description="Connection timeout in seconds")
    max_connections: int = Field(default=10, description="Maximum connections")
    connection_pool_size: int = Field(default=5, description="Connection pool size")

    # Cache settings
    default_ttl: int = Field(default=3600, description="Default TTL in seconds")
    max_memory: Optional[str] = Field(default=None, description="Maximum memory limit")
    eviction_policy: CachePolicy = Field(default=CachePolicy.LRU, description="Eviction policy")

    # Performance settings
    batch_size: int = Field(default=100, description="Batch operation size")
    compression: bool = Field(default=False, description="Enable compression")
    serialization: str = Field(default="pickle", description="Serialization method")

    # Advanced settings
    key_prefix: Optional[str] = Field(default=None, description="Key prefix")
    key_encoding: str = Field(default="utf-8", description="Key encoding")
    value_encoding: str = Field(default="utf-8", description="Value encoding")

    # Custom options
    custom_options: Optional[Dict[str, Any]] = None


class CacheCapabilities(BaseModel):
    """Cache capabilities model."""
    provider: str
    cache_type: CacheType
    supported_data_types: List[str]
    max_key_length: Optional[int] = None
    max_value_size: Optional[int] = None
    max_memory: Optional[int] = None
    supports_ttl: bool = False
    supports_persistence: bool = False
    supports_replication: bool = False
    supports_sharding: bool = False
    supports_compression: bool = False
    supports_encryption: bool = False
    supports_transactions: bool = False
    supports_pub_sub: bool = False
    supports_lua_scripts: bool = False
    supports_async_operations: bool = False


class CacheInterface(ABC):
    """
    Abstract interface for caching services.

    This class defines the contract that all cache implementations must follow.
    It provides a unified interface for different caching systems while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: CacheConfig):
        """Initialize the cache client with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.cache_type = config.cache_type
        self.host = config.host
        self.port = config.port
        self.database = config.database
        self.timeout = config.timeout
        self.max_connections = config.max_connections
        self.default_ttl = config.default_ttl
        self.eviction_policy = config.eviction_policy
        self.key_prefix = config.key_prefix
        self._capabilities: Optional[CacheCapabilities] = None
        self._stats = {
            'hits': 0,
            'misses': 0,
            'operations': 0,
            'evictions': 0,
            'total_time': 0.0,
            'start_time': time.time()
        }

    @property
    @abstractmethod
    def capabilities(self) -> CacheCapabilities:
        """
        Get the capabilities of this cache provider.

        Returns:
            CacheCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the cache service.

        Returns:
            bool: True if connection successful

        Raises:
            ConnectionException: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the cache service.
        """
        pass

    @abstractmethod
    async def ping(self) -> bool:
        """
        Ping the cache service to check connectivity.

        Returns:
            bool: True if service is responsive

        Raises:
            ConnectionException: If ping fails
        """
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CacheResult:
        """
        Set a value in the cache.

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
        pass

    @abstractmethod
    async def get(self, key: str) -> CacheResult:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            CacheResult: Operation result

        Raises:
            CacheException: If get operation fails
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> CacheResult:
        """
        Delete a value from the cache.

        Args:
            key: Cache key

        Returns:
            CacheResult: Operation result

        Raises:
            CacheException: If delete operation fails
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: Cache key

        Returns:
            bool: True if key exists

        Raises:
            CacheException: If check fails
        """
        pass

    @abstractmethod
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set TTL for an existing key.

        Args:
            key: Cache key
            ttl: Time to live in seconds

        Returns:
            bool: True if TTL set successfully

        Raises:
            CacheException: If TTL operation fails
        """
        pass

    @abstractmethod
    async def ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            Optional[int]: Remaining TTL in seconds, None if no TTL or key doesn't exist

        Raises:
            CacheException: If TTL operation fails
        """
        pass

    @abstractmethod
    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get all keys matching a pattern.

        Args:
            pattern: Key pattern (optional, supports wildcards)

        Returns:
            List[str]: Matching keys

        Raises:
            CacheException: If keys operation fails
        """
        pass

    @abstractmethod
    async def clear(self) -> CacheResult:
        """
        Clear all cache entries.

        Returns:
            CacheResult: Operation result

        Raises:
            CacheException: If clear operation fails
        """
        pass

    @abstractmethod
    async def size(self) -> int:
        """
        Get the number of entries in the cache.

        Returns:
            int: Number of entries

        Raises:
            CacheException: If size operation fails
        """
        pass

    async def mget(self, keys: List[str]) -> Dict[str, CacheResult]:
        """
        Get multiple values from the cache.

        Args:
            keys: List of cache keys

        Returns:
            Dict[str, CacheResult]: Results for each key

        Raises:
            CacheException: If mget operation fails
        """
        results = {}
        for key in keys:
            try:
                result = await self.get(key)
                results[key] = result
            except Exception as e:
                results[key] = CacheResult(
                    success=False,
                    key=key,
                    error=str(e)
                )
        return results

    async def mset(self, items: Dict[str, Any], ttl: Optional[int] = None) -> Dict[str, CacheResult]:
        """
        Set multiple values in the cache.

        Args:
            items: Dictionary of key-value pairs
            ttl: Default TTL for all items (optional)

        Returns:
            Dict[str, CacheResult]: Results for each key

        Raises:
            CacheException: If mset operation fails
        """
        results = {}
        for key, value in items.items():
            try:
                result = await self.set(key, value, ttl)
                results[key] = result
            except Exception as e:
                results[key] = CacheResult(
                    success=False,
                    key=key,
                    error=str(e)
                )
        return results

    async def mdelete(self, keys: List[str]) -> Dict[str, CacheResult]:
        """
        Delete multiple keys from the cache.

        Args:
            keys: List of cache keys

        Returns:
            Dict[str, CacheResult]: Results for each key

        Raises:
            CacheException: If mdelete operation fails
        """
        results = {}
        for key in keys:
            try:
                result = await self.delete(key)
                results[key] = result
            except Exception as e:
                results[key] = CacheResult(
                    success=False,
                    key=key,
                    error=str(e)
                )
        return results

    async def increment(self, key: str, amount: int = 1) -> CacheResult:
        """
        Increment a numeric value in the cache.

        Args:
            key: Cache key
            amount: Increment amount

        Returns:
            CacheResult: Operation result with new value

        Raises:
            CacheException: If increment operation fails
        """
        # Default implementation - get, increment, set
        result = await self.get(key)
        if result.hit and isinstance(result.value, (int, float)):
            new_value = result.value + amount
            set_result = await self.set(key, new_value)
            return CacheResult(
                success=set_result.success,
                key=key,
                value=new_value,
                hit=True,
                operation_time_ms=set_result.operation_time_ms,
                error=set_result.error
            )
        else:
            # Key doesn't exist or value is not numeric
            set_result = await self.set(key, amount)
            return CacheResult(
                success=set_result.success,
                key=key,
                value=amount,
                hit=False,
                operation_time_ms=set_result.operation_time_ms,
                error=set_result.error
            )

    async def decrement(self, key: str, amount: int = 1) -> CacheResult:
        """
        Decrement a numeric value in the cache.

        Args:
            key: Cache key
            amount: Decrement amount

        Returns:
            CacheResult: Operation result with new value

        Raises:
            CacheException: If decrement operation fails
        """
        return await self.increment(key, -amount)

    async def get_stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats: Cache statistics
        """
        current_time = time.time()
        uptime = current_time - self._stats['start_time']
        total_ops = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total_ops) if total_ops > 0 else 0.0
        avg_time = (self._stats['total_time'] / self._stats['operations']) if self._stats['operations'] > 0 else 0.0

        try:
            current_size = await self.size()
        except:
            current_size = 0

        return CacheStats(
            cache_type=self.provider_name,
            total_keys=current_size,
            hits=self._stats['hits'],
            misses=self._stats['misses'],
            hit_rate=hit_rate,
            eviction_count=self._stats['evictions'],
            operations_count=self._stats['operations'],
            avg_operation_time_ms=avg_time,
            uptime_seconds=uptime
        )

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = {
            'hits': 0,
            'misses': 0,
            'operations': 0,
            'evictions': 0,
            'total_time': 0.0,
            'start_time': time.time()
        }

    def _format_key(self, key: str) -> str:
        """Format key with prefix."""
        if self.key_prefix:
            return f"{self.key_prefix}:{key}"
        return key

    def _update_stats(self, hit: bool, operation_time: float) -> None:
        """Update internal statistics."""
        self._stats['operations'] += 1
        self._stats['total_time'] += operation_time
        if hit:
            self._stats['hits'] += 1
        else:
            self._stats['misses'] += 1

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the cache provider.

        Returns:
            Dict[str, Any]: Provider information
        """
        return {
            "provider": self.provider_name,
            "cache_type": self.cache_type,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "capabilities": self.capabilities.dict(),
            "config": {
                "timeout": self.timeout,
                "max_connections": self.max_connections,
                "default_ttl": self.default_ttl,
                "eviction_policy": self.eviction_policy,
                "key_prefix": self.key_prefix
            }
        }


class CacheException(Exception):
    """Exception raised by cache services."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        key: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.provider = provider
        self.key = key
        self.error_code = error_code


class ConnectionException(CacheException):
    """Exception raised when connection fails."""
    pass


class SerializationException(CacheException):
    """Exception raised when serialization fails."""
    pass


class KeyException(CacheException):
    """Exception raised for key-related errors."""
    pass


class ValueException(CacheException):
    """Exception raised for value-related errors."""
    pass


class PolicyException(CacheException):
    """Exception raised for policy-related errors."""
    pass