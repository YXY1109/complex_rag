"""
Cache Interfaces

This module contains abstract interfaces for caching services.
"""

from .cache_interface import (
    # Cache interfaces and models
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
]
