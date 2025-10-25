"""
Cache Module

This module contains cache interfaces and implementations for
multi-level caching with Redis, memory, and custom strategies.
"""

from .interfaces import (
    # Cache interfaces
    CacheInterface,
    CacheConfig,
    CacheCapabilities,
    CacheType,
    CachePolicy,
    CacheEntry,
    CacheResult,
    CacheStats,

    # Cache exceptions
    CacheException,
    ConnectionException,
    SerializationException,
    KeyException,
    ValueException,
    PolicyException,
)

from .implementations import (
    # Cache clients
    RedisCacheClient,
    MemoryCacheClient,
    MultiLevelCacheClient,
    CacheLevel,
)

__all__ = [
    # Cache interfaces
    "CacheInterface",
    "CacheConfig",
    "CacheCapabilities",
    "CacheType",
    "CachePolicy",
    "CacheEntry",
    "CacheResult",
    "CacheStats",

    # Cache exceptions
    "CacheException",
    "ConnectionException",
    "SerializationException",
    "KeyException",
    "ValueException",
    "PolicyException",

    # Cache clients
    "RedisCacheClient",
    "MemoryCacheClient",
    "MultiLevelCacheClient",
    "CacheLevel",
]
