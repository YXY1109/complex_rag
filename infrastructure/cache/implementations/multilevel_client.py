"""
Multi-Level Cache Client Implementation

This module provides a multi-level cache client that implements
the CacheInterface abstract base class with L1 (memory) and L2 (Redis) caching.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass

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

# Import cache implementations
try:
    from .redis_client import RedisCacheClient
    from .memory_client import MemoryCacheClient
except ImportError:
    RedisCacheClient = None
    MemoryCacheClient = None


@dataclass
class CacheLevel:
    """Cache level configuration."""
    name: str
    client: CacheInterface
    ttl_multiplier: float = 1.0  # TTL multiplier for this level
    write_through: bool = True  # Whether to write through to this level
    read_through: bool = True   # Whether to read through from this level


class MultiLevelCacheClient(CacheInterface):
    """
    Multi-level cache client implementation.

    Provides L1 (memory) and L2 (Redis) caching with intelligent
    promotion/demotion, write-through, and read-through strategies.
    """

    def __init__(self, config: CacheConfig, levels: List[CacheLevel]):
        """
        Initialize multi-level cache client with configuration.

        Args:
            config: Cache configuration
            levels: List of cache levels (L1, L2, etc.)
        """
        super().__init__(config)

        # Cache levels
        self.levels = levels
        self.l1_level = levels[0] if len(levels) > 0 else None
        self.l2_level = levels[1] if len(levels) > 1 else None

        # Multi-level cache configuration
        self.promotion_enabled = config.custom_options.get('promotion_enabled', True) if config.custom_options else True
        self.demotion_enabled = config.custom_options.get('demotion_enabled', True) if config.custom_options else True
        self.background_sync = config.custom_options.get('background_sync', True) if config.custom_options else True
        self.sync_interval = config.custom_options.get('sync_interval', 300) if config.custom_options else 300  # 5 minutes

        # Background synchronization
        self._sync_task: Optional[asyncio.Task] = None
        self._stop_sync = False

        # Statistics for each level
        self._level_stats = {}

    @property
    def capabilities(self) -> CacheCapabilities:
        """Get multi-level cache capabilities."""
        # Combine capabilities from all levels
        all_data_types = set()
        max_memory = None
        supports_ttl = True
        supports_persistence = False
        supports_replication = False
        supports_compression = False
        supports_encryption = False
        supports_transactions = False
        supports_pub_sub = False
        supports_lua_scripts = False

        for level in self.levels:
            caps = level.client.capabilities
            all_data_types.update(caps.supported_data_types)
            if caps.max_memory and (max_memory is None or caps.max_memory > max_memory):
                max_memory = caps.max_memory
            supports_ttl = supports_ttl or caps.supports_ttl
            supports_persistence = supports_persistence or caps.supports_persistence
            supports_replication = supports_replication or caps.supports_replication
            supports_compression = supports_compression or caps.supports_compression
            supports_encryption = supports_encryption or caps.supports_encryption
            supports_transactions = supports_transactions or caps.supports_transactions
            supports_pub_sub = supports_pub_sub or caps.supports_pub_sub
            supports_lua_scripts = supports_lua_scripts or caps.supports_lua_scripts

        return CacheCapabilities(
            provider="multilevel",
            cache_type=CacheType.CUSTOM,
            supported_data_types=list(all_data_types),
            max_key_length=512,  # Conservative limit
            max_value_size=None,  # Limited by L2 cache
            max_memory=max_memory,
            supports_ttl=supports_ttl,
            supports_persistence=supports_persistence,
            supports_replication=supports_replication,
            supports_sharding=False,  # Not directly supported in multi-level
            supports_compression=supports_compression,
            supports_encryption=supports_encryption,
            supports_transactions=supports_transactions,
            supports_pub_sub=supports_pub_sub,
            supports_lua_scripts=supports_lua_scripts,
            supports_async_operations=True
        )

    async def connect(self) -> bool:
        """
        Connect to all cache levels.

        Returns:
            bool: True if all connections successful

        Raises:
            ConnectionException: If any connection fails
        """
        try:
            # Connect all levels
            connection_results = []
            for level in self.levels:
                try:
                    result = await level.client.connect()
                    connection_results.append(result)
                except Exception as e:
                    connection_results.append(False)

            # Start background sync if enabled
            if self.background_sync and not self._sync_task:
                self._sync_task = asyncio.create_task(self._background_sync())

            # Return True if at least L1 is connected
            return connection_results[0] if connection_results else False

        except Exception as e:
            raise ConnectionException(
                f"Multi-level cache connection failed: {str(e)}",
                provider=self.provider_name,
                error_code="CONNECTION_ERROR"
            )

    async def disconnect(self) -> None:
        """Disconnect from all cache levels."""
        self._stop_sync = True

        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        # Disconnect all levels
        for level in self.levels:
            try:
                await level.client.disconnect()
            except:
                pass  # Ignore disconnect errors

    async def ping(self) -> bool:
        """
        Ping all cache levels.

        Returns:
            bool: True if at least L1 is responsive

        Raises:
            ConnectionException: If all levels fail
        """
        # Check L1 first
        if self.l1_level:
            try:
                if await self.l1_level.client.ping():
                    return True
            except:
                pass

        # Check other levels
        for level in self.levels[1:]:
            try:
                if await level.client.client.ping():
                    return True
            except:
                pass

        raise ConnectionException(
            "All cache levels are unresponsive",
            provider=self.provider_name,
            error_code="ALL_LEVELS_DOWN"
        )

    async def _background_sync(self) -> None:
        """Background task to sync cache levels."""
        while not self._stop_sync:
            try:
                await asyncio.sleep(self.sync_interval)
                await self._sync_levels()
            except asyncio.CancelledError:
                break
            except Exception:
                # Ignore sync errors and continue
                pass

    async def _sync_levels(self) -> None:
        """Synchronize cache levels."""
        if not self.l1_level or not self.l2_level:
            return

        try:
            # Get L1 keys
            l1_keys = await self.l1_level.client.keys()

            # For each L1 key, check if it exists in L2
            for key in l1_keys:
                try:
                    l1_result = await self.l1_level.client.get(key)
                    if l1_result.hit:
                        l2_exists = await self.l2_level.client.exists(key)
                        if not l2_exists and self.demotion_enabled:
                            # Demote to L2
                            ttl = l1_result.ttl or self.default_ttl
                            await self.l2_level.client.set(
                                key,
                                l1_result.value,
                                int(ttl * self.l2_level.ttl_multiplier)
                            )
                except:
                    continue  # Skip problematic keys

        except:
            pass  # Ignore sync errors

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CacheResult:
        """
        Set a value in multi-level cache.

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
            results = []

            # Write to L1 if enabled
            if self.l1_level and self.l1_level.write_through:
                l1_ttl = int((ttl or self.default_ttl) * self.l1_level.ttl_multiplier)
                l1_result = await self.l1_level.client.set(key, value, l1_ttl, metadata)
                results.append(l1_result)

            # Write to L2 if enabled
            if self.l2_level and self.l2_level.write_through:
                l2_ttl = int((ttl or self.default_ttl) * self.l2_level.ttl_multiplier)
                l2_result = await self.l2_level.client.set(key, value, l2_ttl, metadata)
                results.append(l2_result)

            # Write to other levels
            for level in self.levels[2:]:
                if level.write_through:
                    level_ttl = int((ttl or self.default_ttl) * level.ttl_multiplier)
                    level_result = await level.client.set(key, value, level_ttl, metadata)
                    results.append(level_result)

            operation_time = (time.time() - start_time) * 1000

            # Return result from L1 if available, otherwise from first successful write
            primary_result = results[0] if results else CacheResult(success=False, key=key)

            self._update_stats(primary_result.hit, operation_time)

            return CacheResult(
                success=any(r.success for r in results),
                key=key,
                hit=False,
                ttl=ttl or self.default_ttl,
                operation_time_ms=operation_time,
                metadata={"level_results": len(results)}
            )

        except Exception as e:
            raise CacheException(
                f"Multi-level cache set operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="SET_ERROR"
            )

    async def get(self, key: str) -> CacheResult:
        """
        Get a value from multi-level cache.

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

            # Try L1 first
            if self.l1_level and self.l1_level.read_through:
                l1_result = await self.l1_level.client.get(key)
                if l1_result.hit and not self._is_expired(l1_result):
                    # L1 hit
                    operation_time = (time.time() - start_time) * 1000
                    self._update_stats(True, operation_time)

                    # Update access time and promote if needed
                    if self.promotion_enabled and self.l2_level:
                        await self._promote_to_l2(key, l1_result.value, l1_result.ttl)

                    return CacheResult(
                        success=True,
                        key=key,
                        value=l1_result.value,
                        hit=True,
                        ttl=l1_result.ttl,
                        metadata=l1_result.metadata,
                        operation_time_ms=operation_time,
                        metadata={"level": "L1"}
                    )

            # Try L2
            if self.l2_level and self.l2_level.read_through:
                l2_result = await self.l2_level.client.get(key)
                if l2_result.hit and not self._is_expired(l2_result):
                    # L2 hit - promote to L1
                    operation_time = (time.time() - start_time) * 1000
                    self._update_stats(True, operation_time)

                    if self.promotion_enabled and self.l1_level:
                        await self._promote_to_l1(key, l2_result.value, l2_result.ttl)

                    return CacheResult(
                        success=True,
                        key=key,
                        value=l2_result.value,
                        hit=True,
                        ttl=l2_result.ttl,
                        metadata=l2_result.metadata,
                        operation_time_ms=operation_time,
                        metadata={"level": "L2"}
                    )

            # Try other levels
            for i, level in enumerate(self.levels[2:], start=3):
                if level.read_through:
                    level_result = await level.client.get(key)
                    if level_result.hit and not self._is_expired(level_result):
                        operation_time = (time.time() - start_time) * 1000
                        self._update_stats(True, operation_time)

                        # Promote to higher levels
                        if self.promotion_enabled:
                            await self._promote_upwards(key, level_result.value, level_result.ttl, i)

                        return CacheResult(
                            success=True,
                            key=key,
                            value=level_result.value,
                            hit=True,
                            ttl=level_result.ttl,
                            metadata=level_result.metadata,
                            operation_time_ms=operation_time,
                            metadata={"level": f"L{i}"}
                        )

            # Cache miss
            operation_time = (time.time() - start_time) * 1000
            self._update_stats(False, operation_time)

            return CacheResult(
                success=True,
                key=key,
                hit=False,
                operation_time_ms=operation_time,
                metadata={"level": "MISS"}
            )

        except Exception as e:
            raise CacheException(
                f"Multi-level cache get operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="GET_ERROR"
            )

    def _is_expired(self, result: CacheResult) -> bool:
        """Check if a cache result is expired."""
        return result.ttl is not None and result.ttl <= 0

    async def _promote_to_l1(self, key: str, value: Any, ttl: Optional[int]) -> None:
        """Promote a value to L1 cache."""
        if not self.l1_level:
            return

        try:
            l1_ttl = int((ttl or self.default_ttl) * self.l1_level.ttl_multiplier)
            await self.l1_level.client.set(key, value, l1_ttl)
        except:
            pass  # Ignore promotion errors

    async def _promote_to_l2(self, key: str, value: Any, ttl: Optional[int]) -> None:
        """Promote a value to L2 cache."""
        if not self.l2_level:
            return

        try:
            l2_ttl = int((ttl or self.default_ttl) * self.l2_level.ttl_multiplier)
            await self.l2_level.client.set(key, value, l2_ttl)
        except:
            pass  # Ignore promotion errors

    async def _promote_upwards(self, key: str, value: Any, ttl: Optional[int], from_level: int) -> None:
        """Promote a value to higher cache levels."""
        # Promote to all higher levels
        for i in range(min(from_level - 1, len(self.levels))):
            level = self.levels[i]
            try:
                level_ttl = int((ttl or self.default_ttl) * level.ttl_multiplier)
                await level.client.set(key, value, level_ttl)
            except:
                pass  # Ignore promotion errors

    async def delete(self, key: str) -> CacheResult:
        """
        Delete a value from all cache levels.

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
            results = []

            # Delete from all levels
            for level in self.levels:
                try:
                    level_result = await level.client.delete(key)
                    results.append(level_result)
                except:
                    results.append(CacheResult(success=False, key=key))

            operation_time = (time.time() - start_time) * 1000

            return CacheResult(
                success=any(r.success for r in results),
                key=key,
                hit=any(r.hit for r in results),
                operation_time_ms=operation_time,
                metadata={"deleted_from_levels": sum(1 for r in results if r.success)}
            )

        except Exception as e:
            raise CacheException(
                f"Multi-level cache delete operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="DELETE_ERROR"
            )

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in any cache level.

        Args:
            key: Cache key

        Returns:
            bool: True if key exists in any level

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
            # Check all levels
            for level in self.levels:
                try:
                    if await level.client.exists(key):
                        return True
                except:
                    continue  # Skip failed levels

            return False

        except Exception as e:
            raise CacheException(
                f"Multi-level cache exists operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="EXISTS_ERROR"
            )

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set TTL for a key in all cache levels.

        Args:
            key: Cache key
            ttl: Time to live in seconds

        Returns:
            bool: True if TTL set in at least one level

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
            results = []

            # Set TTL in all levels where key exists
            for level in self.levels:
                try:
                    if await level.client.exists(key):
                        level_ttl = int(ttl * level.ttl_multiplier)
                        result = await level.client.expire(key, level_ttl)
                        results.append(result)
                except:
                    results.append(False)

            return any(results)

        except Exception as e:
            raise CacheException(
                f"Multi-level cache expire operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="EXPIRE_ERROR"
            )

    async def ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key from the highest level where it exists.

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
            # Check L1 first
            if self.l1_level:
                l1_ttl = await self.l1_level.client.ttl(key)
                if l1_ttl is not None:
                    return l1_ttl

            # Check L2
            if self.l2_level:
                l2_ttl = await self.l2_level.client.ttl(key)
                if l2_ttl is not None:
                    return l2_ttl

            # Check other levels
            for level in self.levels[2:]:
                level_ttl = await level.client.ttl(key)
                if level_ttl is not None:
                    return level_ttl

            return None

        except Exception as e:
            raise CacheException(
                f"Multi-level cache TTL operation failed: {str(e)}",
                provider=self.provider_name,
                key=key,
                error_code="TTL_ERROR"
            )

    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get all keys matching a pattern from all cache levels.

        Args:
            pattern: Key pattern (optional)

        Returns:
            List[str]: Union of matching keys from all levels

        Raises:
            CacheException: If keys operation fails
        """
        try:
            all_keys = set()

            # Get keys from all levels
            for level in self.levels:
                try:
                    level_keys = await level.client.keys(pattern)
                    all_keys.update(level_keys)
                except:
                    continue  # Skip failed levels

            return list(all_keys)

        except Exception as e:
            raise CacheException(
                f"Multi-level cache keys operation failed: {str(e)}",
                provider=self.provider_name,
                error_code="KEYS_ERROR"
            )

    async def clear(self) -> CacheResult:
        """
        Clear all cache entries from all levels.

        Returns:
            CacheResult: Operation result

        Raises:
            CacheException: If clear operation fails
        """
        try:
            start_time = time.time()
            results = []

            # Clear all levels
            for level in self.levels:
                try:
                    level_result = await level.client.clear()
                    results.append(level_result)
                except:
                    results.append(CacheResult(success=False, key="*"))

            operation_time = (time.time() - start_time) * 1000

            return CacheResult(
                success=any(r.success for r in results),
                key="*",
                operation_time_ms=operation_time,
                metadata={"cleared_levels": sum(1 for r in results if r.success)}
            )

        except Exception as e:
            raise CacheException(
                f"Multi-level cache clear operation failed: {str(e)}",
                provider=self.provider_name,
                error_code="CLEAR_ERROR"
            )

    async def size(self) -> int:
        """
        Get the total number of entries across all cache levels.

        Returns:
            int: Total number of entries

        Raises:
            CacheException: If size operation fails
        """
        try:
            total_size = 0

            for level in self.levels:
                try:
                    level_size = await level.client.size()
                    total_size += level_size
                except:
                    continue  # Skip failed levels

            return total_size

        except Exception as e:
            raise CacheException(
                f"Multi-level cache size operation failed: {str(e)}",
                provider=self.provider_name,
                error_code="SIZE_ERROR"
            )

    async def get_stats(self) -> CacheStats:
        """
        Get comprehensive statistics from all cache levels.

        Returns:
            CacheStats: Combined cache statistics
        """
        try:
            base_stats = await super().get_stats()

            # Collect stats from all levels
            level_stats = {}
            total_memory = 0
            max_memory = None

            for i, level in enumerate(self.levels, start=1):
                try:
                    level_stat = await level.client.get_stats()
                    level_stats[f"L{i}"] = level_stat

                    if level_stat.memory_usage:
                        total_memory += level_stat.memory_usage
                    if level_stat.memory_limit and (max_memory is None or level_stat.memory_limit > max_memory):
                        max_memory = level_stat.memory_limit
                except:
                    continue

            # Update base stats with multi-level information
            base_stats.memory_usage = total_memory
            base_stats.memory_limit = max_memory

            # Store level stats for detailed reporting
            self._level_stats = level_stats

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

    def get_level_stats(self) -> Dict[str, CacheStats]:
        """
        Get statistics for individual cache levels.

        Returns:
            Dict[str, CacheStats]: Statistics for each level
        """
        return self._level_stats.copy()


# Export multi-level cache client
__all__ = ['MultiLevelCacheClient', 'CacheLevel']