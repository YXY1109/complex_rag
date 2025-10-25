"""
Memory Cache Client Implementation

This module provides an in-memory cache client that implements
the CacheInterface abstract base class.
"""

import asyncio
import json
import pickle
import time
import threading
from typing import Any, Dict, List, Optional, Union, OrderedDict
from collections import OrderedDict as collections_OrderedDict
from datetime import datetime, timezone
import weakref
import gc

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


class MemoryCacheClient(CacheInterface):
    """
    In-memory cache client implementation.

    Provides thread-safe in-memory caching with configurable eviction policies,
    size limits, and comprehensive statistics.
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize memory cache client with configuration.

        Args:
            config: Cache configuration with memory-specific settings
        """
        super().__init__(config)

        # Memory-specific configuration
        self.max_items = config.custom_options.get('max_items', 1000) if config.custom_options else 1000
        self.max_memory_mb = config.custom_options.get('max_memory_mb', 100) if config.custom_options else 100
        self.cleanup_interval = config.custom_options.get('cleanup_interval', 60) if config.custom_options else 60

        # Internal storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: collections_OrderedDict = collections_OrderedDict()  # For LRU
        self._access_count: Dict[str, int] = {}  # For LFU
        self._creation_time: Dict[str, float] = {}  # For FIFO/LIFO

        # Thread safety
        self._lock = threading.RLock()

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_cleanup = False

        # Statistics
        self._memory_usage = 0

    @property
    def capabilities(self) -> CacheCapabilities:
        """Get memory cache capabilities."""
        return CacheCapabilities(
            provider="memory",
            cache_type=CacheType.MEMORY,
            supported_data_types=[
                "string", "dict", "list", "tuple", "set",
                "bytes", "int", "float", "bool", "None",
                "json", "pickle", "custom_objects"
            ],
            max_key_length=None,  # No practical limit
            max_value_size=None,  # Limited by available memory
            max_memory=self.max_memory_mb * 1024 * 1024,  # Convert to bytes
            supports_ttl=True,
            supports_persistence=False,
            supports_replication=False,
            supports_sharding=False,
            supports_compression=False,  # Could be added
            supports_encryption=False,   # Could be added
            supports_transactions=False,
            supports_pub_sub=False,
            supports_lua_scripts=False,
            supports_async_operations=True
        )

    async def connect(self) -> bool:
        """
        Connect to memory cache service.

        Returns:
            bool: True if connection successful
        """
        try:
            # Initialize cleanup task
            if not self._cleanup_task or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_expired())

            return True

        except Exception as e:
            raise ConnectionException(
                f"Failed to initialize memory cache: {str(e)}",
                provider=self.provider_name,
                error_code="INITIALIZATION_ERROR"
            )

    async def disconnect(self) -> None:
        """Disconnect from memory cache service."""
        with self._lock:
            self._stop_cleanup = True
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Clear cache
            self._cache.clear()
            self._access_order.clear()
            self._access_count.clear()
            self._creation_time.clear()
            self._memory_usage = 0

    async def ping(self) -> bool:
        """
        Ping memory cache service.

        Returns:
            bool: True if service is responsive
        """
        return True  # Memory cache is always responsive

    def _estimate_size(self, obj: Any) -> int:
        """
        Estimate memory usage of an object.

        Args:
            obj: Object to estimate size for

        Returns:
            int: Estimated size in bytes
        """
        try:
            if isinstance(obj, (str, bytes)):
                return len(obj)
            elif isinstance(obj, (int, float, bool)):
                return 8  # Rough estimate
            elif isinstance(obj, (list, tuple, set)):
                return sum(self._estimate_size(item) for item in obj) + 64
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items()) + 64
            else:
                # For custom objects, use pickle size as estimate
                return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate for complex objects

    def _should_evict(self, key: str, new_size: int) -> bool:
        """
        Check if eviction is needed for new item.

        Args:
            key: Cache key
            new_size: Size of new item in bytes

        Returns:
            bool: True if eviction is needed
        """
        # Check item count limit
        if self.max_items and len(self._cache) >= self.max_items:
            return True

        # Check memory limit
        if self.max_memory_mb:
            total_memory = self._memory_usage + new_size
            if total_memory > self.max_memory_mb * 1024 * 1024:
                return True

        return False

    def _evict_item(self) -> Optional[str]:
        """
        Evict an item based on configured policy.

        Returns:
            Optional[str]: Evicted key, None if no eviction
        """
        if not self._cache:
            return None

        if self.eviction_policy == CachePolicy.LRU:
            # Least Recently Used
            if self._access_order:
                evicted_key, _ = self._access_order.popitem(last=False)
                return evicted_key

        elif self.eviction_policy == CachePolicy.LFU:
            # Least Frequently Used
            if self._access_count:
                evicted_key = min(self._access_count.keys(), key=lambda k: self._access_count[k])
                return evicted_key

        elif self.eviction_policy == CachePolicy.FIFO:
            # First In First Out
            if self._creation_time:
                evicted_key = min(self._creation_time.keys(), key=lambda k: self._creation_time[k])
                return evicted_key

        elif self.eviction_policy == CachePolicy.LIFO:
            # Last In First Out
            if self._creation_time:
                evicted_key = max(self._creation_time.keys(), key=lambda k: self._creation_time[k])
                return evicted_key

        return None

    def _cleanup_entry(self, key: str) -> None:
        """
        Clean up all tracking data for a key.

        Args:
            key: Cache key to clean up
        """
        if key in self._cache:
            self._memory_usage -= self._cache[key].size or 0
            del self._cache[key]

        if key in self._access_order:
            del self._access_order[key]

        if key in self._access_count:
            del self._access_count[key]

        if key in self._creation_time:
            del self._creation_time[key]

        self._stats['evictions'] += 1

    async def _cleanup_expired(self) -> None:
        """Background task to clean up expired entries."""
        while not self._stop_cleanup:
            try:
                await asyncio.sleep(self.cleanup_interval)

                with self._lock:
                    current_time = time.time()
                    expired_keys = []

                    for key, entry in self._cache.items():
                        if entry.ttl and (current_time - (entry.created_at or current_time)) > entry.ttl:
                            expired_keys.append(key)

                    for key in expired_keys:
                        self._cleanup_entry(key)

            except asyncio.CancelledError:
                break
            except Exception:
                # Ignore cleanup errors and continue
                pass

    def _is_expired(self, entry: CacheEntry) -> bool:
        """
        Check if a cache entry is expired.

        Args:
            entry: Cache entry to check

        Returns:
            bool: True if entry is expired
        """
        if not entry.ttl or not entry.created_at:
            return False

        return (time.time() - entry.created_at) > entry.ttl

    def _update_access_tracking(self, key: str) -> None:
        """
        Update access tracking for a key.

        Args:
            key: Cache key
        """
        # Update LRU tracking
        self._access_order.pop(key, None)  # Remove if exists
        self._access_order[key] = time.time()  # Add to end

        # Update LFU tracking
        self._access_count[key] = self._access_count.get(key, 0) + 1

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CacheResult:
        """
        Set a value in memory cache.

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

        try:
            start_time = time.time()

            with self._lock:
                # Estimate size
                estimated_size = self._estimate_size(value)

                # Check if eviction is needed
                evicted_keys = []
                while self._should_evict(key, estimated_size):
                    evicted_key = self._evict_item()
                    if evicted_key:
                        evicted_keys.append(evicted_key)
                        self._cleanup_entry(evicted_key)
                    else:
                        break

                # Clean up existing entry if it exists
                if key in self._cache:
                    self._cleanup_entry(key)

                # Create cache entry
                current_time = time.time()
                entry = CacheEntry(
                    key=key,
                    value=value,
                    ttl=ttl or self.default_ttl,
                    created_at=current_time,
                    accessed_at=current_time,
                    access_count=1,
                    size=estimated_size,
                    metadata=metadata
                )

                # Store entry
                self._cache[key] = entry
                self._memory_usage += estimated_size

                # Update tracking
                self._access_order[key] = current_time
                self._access_count[key] = 1
                self._creation_time[key] = current_time

                operation_time = (time.time() - start_time) * 1000
                self._update_stats(False, operation_time)

                return CacheResult(
                    success=True,
                    key=key,
                    hit=False,
                    ttl=entry.ttl,
                    operation_time_ms=operation_time,
                    metadata={"evicted_keys": evicted_keys} if evicted_keys else None
                )

        except Exception as e:
            raise CacheException(
                f"Memory cache set operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="SET_ERROR"
            )

    async def get(self, key: str) -> CacheResult:
        """
        Get a value from memory cache.

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

        try:
            start_time = time.time()

            with self._lock:
                if key not in self._cache:
                    operation_time = (time.time() - start_time) * 1000
                    self._update_stats(False, operation_time)

                    return CacheResult(
                        success=True,
                        key=key,
                        hit=False,
                        operation_time_ms=operation_time
                    )

                entry = self._cache[key]

                # Check if expired
                if self._is_expired(entry):
                    self._cleanup_entry(key)
                    operation_time = (time.time() - start_time) * 1000
                    self._update_stats(False, operation_time)

                    return CacheResult(
                        success=True,
                        key=key,
                        hit=False,
                        operation_time_ms=operation_time
                    )

                # Update access tracking
                current_time = time.time()
                entry.accessed_at = current_time
                entry.access_count += 1
                self._update_access_tracking(key)

                # Calculate remaining TTL
                remaining_ttl = None
                if entry.ttl and entry.created_at:
                    elapsed = current_time - entry.created_at
                    remaining_ttl = max(0, entry.ttl - elapsed)
                    if remaining_ttl == 0:
                        remaining_ttl = None

                operation_time = (time.time() - start_time) * 1000
                self._update_stats(True, operation_time)

                return CacheResult(
                    success=True,
                    key=key,
                    value=entry.value,
                    hit=True,
                    ttl=remaining_ttl,
                    metadata=entry.metadata,
                    operation_time_ms=operation_time
                )

        except Exception as e:
            raise CacheException(
                f"Memory cache get operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="GET_ERROR"
            )

    async def delete(self, key: str) -> CacheResult:
        """
        Delete a value from memory cache.

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

        try:
            start_time = time.time()

            with self._lock:
                existed = key in self._cache
                if existed:
                    self._cleanup_entry(key)

                operation_time = (time.time() - start_time) * 1000

                return CacheResult(
                    success=existed,
                    key=key,
                    hit=existed,
                    operation_time_ms=operation_time
                )

        except Exception as e:
            raise CacheException(
                f"Memory cache delete operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="DELETE_ERROR"
            )

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in memory cache.

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

        try:
            with self._lock:
                if key not in self._cache:
                    return False

                # Check if expired
                if self._is_expired(self._cache[key]):
                    self._cleanup_entry(key)
                    return False

                return True

        except Exception as e:
            raise CacheException(
                f"Memory cache exists operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="EXISTS_ERROR"
            )

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set TTL for an existing key in memory cache.

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

        try:
            with self._lock:
                if key in self._cache:
                    self._cache[key].ttl = ttl
                    return True
                return False

        except Exception as e:
            raise CacheException(
                f"Memory cache expire operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="EXPIRE_ERROR"
            )

    async def ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key in memory cache.

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

        try:
            with self._lock:
                if key not in self._cache:
                    return None

                entry = self._cache[key]

                # Check if expired
                if self._is_expired(entry):
                    self._cleanup_entry(key)
                    return None

                # Calculate remaining TTL
                if not entry.ttl or not entry.created_at:
                    return None

                current_time = time.time()
                elapsed = current_time - entry.created_at
                remaining_ttl = max(0, entry.ttl - elapsed)

                return remaining_ttl if remaining_ttl > 0 else None

        except Exception as e:
            raise CacheException(
                f"Memory cache TTL operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="TTL_ERROR"
            )

    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get all keys matching a pattern in memory cache.

        Args:
            pattern: Key pattern (optional, supports simple wildcards)

        Returns:
            List[str]: Matching keys

        Raises:
            CacheException: If keys operation fails
        """
        try:
            with self._lock:
                all_keys = list(self._cache.keys())

                if not pattern:
                    return all_keys

                # Simple pattern matching (supports * wildcard)
                import fnmatch
                return [key for key in all_keys if fnmatch.fnmatch(key, pattern)]

        except Exception as e:
            raise CacheException(
                f"Memory cache keys operation failed: {str(e)}",
                provider=self.provider_name,
                error_code="KEYS_ERROR"
            )

    async def clear(self) -> CacheResult:
        """
        Clear all cache entries in memory cache.

        Returns:
            CacheResult: Operation result

        Raises:
            CacheException: If clear operation fails
        """
        try:
            start_time = time.time()

            with self._lock:
                cleared_count = len(self._cache)

                self._cache.clear()
                self._access_order.clear()
                self._access_count.clear()
                self._creation_time.clear()
                self._memory_usage = 0

                # Reset stats
                self.reset_stats()

                operation_time = (time.time() - start_time) * 1000

                return CacheResult(
                    success=True,
                    key="*",
                    operation_time_ms=operation_time,
                    metadata={"cleared_count": cleared_count}
                )

        except Exception as e:
            raise CacheException(
                f"Memory cache clear operation failed: {str(e)}",
                provider=self.provider_name,
                error_code="CLEAR_ERROR"
            )

    async def size(self) -> int:
        """
        Get the number of entries in memory cache.

        Returns:
            int: Number of entries

        Raises:
            CacheException: If size operation fails
        """
        try:
            with self._lock:
                return len(self._cache)
        except Exception as e:
            raise CacheException(
                f"Memory cache size operation failed: {str(e)}",
                provider=self.provider_name,
                error_code="SIZE_ERROR"
            )

    async def get_stats(self) -> CacheStats:
        """
        Get memory cache statistics.

        Returns:
            CacheStats: Cache statistics
        """
        try:
            base_stats = await super().get_stats()

            with self._lock:
                # Add memory-specific stats
                base_stats.memory_usage = self._memory_usage
                base_stats.memory_limit = self.max_memory_mb * 1024 * 1024 if self.max_memory_mb else None

            return base_stats

        except Exception:
            # Return basic stats if error occurs
            return CacheStats(
                cache_type=self.provider_name,
                total_keys=0,
                hits=0,
                misses=0,
                hit_rate=0.0
            )

    async def increment(self, key: str, amount: int = 1) -> CacheResult:
        """
        Increment a numeric value in memory cache.

        Args:
            key: Cache key
            amount: Increment amount

        Returns:
            CacheResult: Operation result with new value

        Raises:
            CacheException: If increment operation fails
        """
        try:
            # Get current value
            get_result = await self.get(key)

            if get_result.hit and isinstance(get_result.value, (int, float)):
                new_value = get_result.value + amount
                await self.set(key, new_value)
                return CacheResult(
                    success=True,
                    key=key,
                    value=new_value,
                    hit=True
                )
            else:
                # Key doesn't exist or value is not numeric
                await self.set(key, amount)
                return CacheResult(
                    success=True,
                    key=key,
                    value=amount,
                    hit=False
                )

        except Exception as e:
            raise CacheException(
                f"Memory cache increment operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="INCREMENT_ERROR"
            )


# Export memory cache client
__all__ = ['MemoryCacheClient']