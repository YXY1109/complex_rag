"""
Cache Configuration

This module contains configuration for cache services.
"""

from pydantic import Field

from ..settings import BaseConfig


class CacheConfig(BaseConfig):
    """Cache configuration."""

    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: str = Field(default="", env="REDIS_PASSWORD")
    redis_database: int = Field(default=0, env="REDIS_DATABASE")
    redis_ssl: bool = Field(default=False, env="REDIS_SSL")
    redis_ssl_cert_reqs: str = Field(default="required", env="REDIS_SSL_CERT_REQS")
    redis_socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    redis_socket_connect_timeout: int = Field(default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT")

    # Redis Connection Pool
    redis_max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    redis_retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")
    redis_health_check_interval: int = Field(default=30, env="REDIS_HEALTH_CHECK_INTERVAL")

    # Cache Settings
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    default_ttl: int = Field(default=3600, env="DEFAULT_TTL")  # 1 hour
    default_prefix: str = Field(default="complex_rag:", env="DEFAULT_PREFIX")

    # Memory Cache Settings
    memory_cache_enabled: bool = Field(default=True, env="MEMORY_CACHE_ENABLED")
    memory_cache_size: int = Field(default=1000, env="MEMORY_CACHE_SIZE")
    memory_cache_ttl: int = Field(default=300, env="MEMORY_CACHE_TTL")  # 5 minutes

    # Cache Categories
    llm_cache_ttl: int = Field(default=1800, env="LLM_CACHE_TTL")  # 30 minutes
    embedding_cache_ttl: int = Field(default=3600, env="EMBEDDING_CACHE_TTL")  # 1 hour
    retrieval_cache_ttl: int = Field(default=600, env="RETRIEVAL_CACHE_TTL")  # 10 minutes
    user_cache_ttl: int = Field(default=7200, env="USER_CACHE_TTL")  # 2 hours
    config_cache_ttl: int = Field(default=86400, env="CONFIG_CACHE_TTL")  # 24 hours

    # Cache Keys
    cache_key_prefixes: dict[str, str] = Field(
        default={
            "llm": "llm:",
            "embedding": "embedding:",
            "retrieval": "retrieval:",
            "user": "user:",
            "knowledge_base": "kb:",
            "document": "doc:",
            "chat": "chat:",
            "config": "config:",
            "health": "health:",
        },
        env="CACHE_KEY_PREFIXES"
    )

    # Cache Serialization
    cache_serializer: str = Field(default="json", env="CACHE_SERIALIZER")  # json, pickle, msgpack
    cache_compression: bool = Field(default=False, env="CACHE_COMPRESSION")
    cache_compression_level: int = Field(default=6, env="CACHE_COMPRESSION_LEVEL")

    # Cache Eviction
    max_memory_policy: str = Field(default="allkeys-lru", env="MAX_MEMORY_POLICY")
    cache_cleanup_interval: int = Field(default=3600, env="CACHE_CLEANUP_INTERVAL")  # 1 hour

    # Cache Statistics
    cache_stats_enabled: bool = Field(default=True, env="CACHE_STATS_ENABLED")
    cache_stats_interval: int = Field(default=300, env="CACHE_STATS_INTERVAL")  # 5 minutes

    # Distributed Cache
    distributed_cache: bool = Field(default=False, env="DISTRIBUTED_CACHE")
    cache_replication_count: int = Field(default=1, env="CACHE_REPLICATION_COUNT")

    def get_redis_url(self) -> str:
        """Get Redis URL."""
        scheme = "rediss" if self.redis_ssl else "redis"
        auth_part = f":{self.redis_password}@" if self.redis_password else ""
        return f"{scheme}://{auth_part}{self.redis_host}:{self.redis_port}/{self.redis_database}"

    def get_cache_key(self, category: str, key: str) -> str:
        """Get cache key with prefix."""
        prefix = self.cache_key_prefixes.get(category, category + ":")
        return f"{self.default_prefix}{prefix}{key}"

    def get_ttl_for_category(self, category: str) -> int:
        """Get TTL for cache category."""
        ttl_map = {
            "llm": self.llm_cache_ttl,
            "embedding": self.embedding_cache_ttl,
            "retrieval": self.retrieval_cache_ttl,
            "user": self.user_cache_ttl,
            "config": self.config_cache_ttl,
        }
        return ttl_map.get(category, self.default_ttl)


# Global cache configuration instance
cache_config = CacheConfig()