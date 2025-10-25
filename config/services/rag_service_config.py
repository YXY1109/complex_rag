"""
RAG Service Configuration

This module contains configuration specific to the Sanic RAG service.
"""

from typing import Optional
from pydantic import Field

from ..settings import BaseConfig
from ...rag_service.interfaces.llm_interface import LLMConfig
from ...rag_service.interfaces.embedding_interface import EmbeddingConfig
from ...rag_service.interfaces.rerank_interface import RerankConfig


class RAGServiceConfig(BaseConfig):
    """RAG service specific configuration."""

    # Sanic Server Settings
    host: str = Field(default="0.0.0.0", env="RAG_HOST")
    port: int = Field(default=8001, env="RAG_PORT")
    workers: int = Field(default=1, env="RAG_WORKERS")  # Single process for best performance
    debug: bool = Field(default=False, env="RAG_DEBUG")
    access_log: bool = Field(default=True, env="RAG_ACCESS_LOG")
    keep_alive: bool = Field(default=True, env="RAG_KEEP_ALIVE")
    keep_alive_timeout: int = Field(default=65, env="RAG_KEEP_ALIVE_TIMEOUT")

    # Request Settings
    request_timeout: int = Field(default=300, env="RAG_REQUEST_TIMEOUT")
    request_max_size: int = Field(default=16 * 1024 * 1024, env="RAG_REQUEST_MAX_SIZE")  # 16MB
    response_timeout: int = Field(default=300, env="RAG_RESPONSE_TIMEOUT")

    # AI Model Settings
    default_llm_provider: str = Field(default="openai", env="DEFAULT_LLM_PROVIDER")
    default_llm_model: str = Field(default="gpt-3.5-turbo", env="DEFAULT_LLM_MODEL")
    default_embedding_provider: str = Field(default="openai", env="DEFAULT_EMBEDDING_PROVIDER")
    default_embedding_model: str = Field(default="text-embedding-ada-002", env="DEFAULT_EMBEDDING_MODEL")
    default_rerank_provider: str = Field(default="bge", env="DEFAULT_RERANK_PROVIDER")
    default_rerank_model: str = Field(default="bge-reranker-base", env="DEFAULT_RERANK_MODEL")

    # OpenAI Compatibility
    openai_compatible_endpoints: bool = Field(default=True, env="OPENAI_COMPATIBLE_ENDPOINTS")
    openai_route_prefix: str = Field(default="/v1", env="OPENAI_ROUTE_PREFIX")

    # LLM Settings
    max_tokens: int = Field(default=4096, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    top_p: float = Field(default=1.0, env="TOP_P")
    frequency_penalty: float = Field(default=0.0, env="FREQUENCY_PENALTY")
    presence_penalty: float = Field(default=0.0, env="PRESENCE_PENALTY")
    stream_response: bool = Field(default=True, env="STREAM_RESPONSE")

    # Embedding Settings
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(default=100, env="EMBEDDING_BATCH_SIZE")
    embedding_timeout: int = Field(default=30, env="EMBEDDING_TIMEOUT")

    # Rerank Settings
    rerank_top_k: int = Field(default=10, env="RERANK_TOP_K")
    rerank_timeout: int = Field(default=30, env="RERANK_TIMEOUT")

    # RAG Settings
    max_retrieved_docs: int = Field(default=10, env="MAX_RETRIEVED_DOCS")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    context_max_length: int = Field(default=4000, env="CONTEXT_MAX_LENGTH")
    overlap_size: int = Field(default=200, env="OVERLAP_SIZE")

    # Memory Settings
    memory_enabled: bool = Field(default=True, env="MEMORY_ENABLED")
    memory_provider: str = Field(default="mem0", env="MEMORY_PROVIDER")
    memory_max_tokens: int = Field(default=2000, env="MEMORY_MAX_TOKENS")
    memory_ttl: int = Field(default=3600, env="MEMORY_TTL")  # 1 hour

    # Retrieval Settings
    retrieval_strategies: list[str] = Field(
        default=["vector", "keyword", "entity"],
        env="RETRIEVAL_STRATEGIES"
    )
    vector_weight: float = Field(default=0.6, env="VECTOR_WEIGHT")
    keyword_weight: float = Field(default=0.3, env="KEYWORD_WEIGHT")
    entity_weight: float = Field(default=0.1, env="ENTITY_WEIGHT")

    # GraphRAG Settings
    graphrag_enabled: bool = Field(default=True, env="GRAPHRAG_ENABLED")
    graphrag_mode: str = Field(default="light", env="GRAPHRAG_MODE")  # "light" or "general"
    graphrag_entity_types: list[str] = Field(
        default=["PERSON", "ORG", "GPE", "EVENT", "DATE"],
        env="GRAPHRAG_ENTITY_TYPES"
    )

    # Cache Settings
    cache_enabled: bool = Field(default=True, env="RAG_CACHE_ENABLED")
    cache_ttl: int = Field(default=1800, env="RAG_CACHE_TTL")  # 30 minutes
    cache_max_size: int = Field(default=1000, env="RAG_CACHE_MAX_SIZE")

    # Queue Settings
    queue_enabled: bool = Field(default=True, env="RAG_QUEUE_ENABLED")
    queue_provider: str = Field(default="trio", env="QUEUE_PROVIDER")
    queue_max_workers: int = Field(default=10, env="QUEUE_MAX_WORKERS")
    queue_timeout: int = Field(default=300, env="QUEUE_TIMEOUT")

    # Monitoring Settings
    metrics_enabled: bool = Field(default=True, env="RAG_METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="RAG_METRICS_PORT")
    tracing_enabled: bool = Field(default=False, env="RAG_TRACING_ENABLED")
    tracing_endpoint: str = Field(default="", env="RAG_TRACING_ENDPOINT")

    # Health Check Settings
    health_check_enabled: bool = Field(default=True, env="RAG_HEALTH_CHECK_ENABLED")
    health_check_interval: int = Field(default=30, env="RAG_HEALTH_CHECK_INTERVAL")
    health_check_timeout: int = Field(default=5, env="RAG_HEALTH_CHECK_TIMEOUT")

    # Service Configurations (for compatibility with service interfaces)
    llm: Optional[LLMConfig] = Field(default=None, env="RAG_LLM_CONFIG")
    embedding: Optional[EmbeddingConfig] = Field(default=None, env="RAG_EMBEDDING_CONFIG")
    rerank: Optional[RerankConfig] = Field(default=None, env="RAG_RERANK_CONFIG")
    memory: Optional[dict] = Field(default=None, env="RAG_MEMORY_CONFIG")

    @property
    def server_url(self) -> str:
        """Get server URL."""
        return f"http://{self.host}:{self.port}"

    @property
    def openai_base_url(self) -> str:
        """Get OpenAI-compatible base URL."""
        return f"{self.server_url}{self.openai_route_prefix}"


# Global RAG service configuration instance
rag_service_config = RAGServiceConfig()