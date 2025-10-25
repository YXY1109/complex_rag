"""
Main Settings Configuration

This module provides the main configuration settings for the Complex RAG system.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings as PydanticBaseSettings


class BaseConfig(PydanticBaseSettings):
    """Base configuration class."""

    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # Project
    project_name: str = Field(default="Complex RAG", env="PROJECT_NAME")
    project_version: str = Field(default="0.1.0", env="PROJECT_VERSION")
    project_root: Path = Field(default=Path(__file__).parent.parent)

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}")

    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class APIConfig(BaseConfig):
    """API service configuration."""

    # FastAPI Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")

    # API Limits
    max_request_size: int = Field(default=16 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 16MB
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")

    # Pagination
    default_page_size: int = Field(default=20, env="DEFAULT_PAGE_SIZE")
    max_page_size: int = Field(default=1000, env="MAX_PAGE_SIZE")


class RAGServiceConfig(BaseConfig):
    """RAG service configuration."""

    # Sanic Settings
    rag_host: str = Field(default="0.0.0.0", env="RAG_HOST")
    rag_port: int = Field(default=8001, env="RAG_PORT")
    rag_workers: int = Field(default=1, env="RAG_WORKERS")  # Single process for best performance

    # AI Model Settings
    default_llm_model: str = Field(default="gpt-3.5-turbo", env="DEFAULT_LLM_MODEL")
    default_embedding_model: str = Field(default="text-embedding-ada-002", env="DEFAULT_EMBEDDING_MODEL")
    default_rerank_model: str = Field(default="bge-reranker-base", env="DEFAULT_RERANK_MODEL")

    # RAG Settings
    max_context_length: int = Field(default=4000, env="MAX_CONTEXT_LENGTH")
    max_retrieved_docs: int = Field(default=10, env="MAX_RETRIEVED_DOCS")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")

    # Memory Settings
    memory_enabled: bool = Field(default=True, env="MEMORY_ENABLED")
    memory_max_tokens: int = Field(default=2000, env="MEMORY_MAX_TOKENS")


class DatabaseConfig(BaseConfig):
    """Database configuration."""

    # MySQL Configuration
    mysql_host: str = Field(default="localhost", env="MYSQL_HOST")
    mysql_port: int = Field(default=3306, env="MYSQL_PORT")
    mysql_user: str = Field(default="root", env="MYSQL_USER")
    mysql_password: str = Field(default="", env="MYSQL_PASSWORD")
    mysql_database: str = Field(default="complex_rag", env="MYSQL_DATABASE")

    # SQLAlchemy Settings
    sqlalchemy_echo: bool = Field(default=False, env="SQLALCHEMY_ECHO")
    sqlalchemy_pool_size: int = Field(default=10, env="SQLALCHEMY_POOL_SIZE")
    sqlalchemy_max_overflow: int = Field(default=20, env="SQLALCHEMY_MAX_OVERFLOW")
    sqlalchemy_pool_timeout: int = Field(default=30, env="SQLALCHEMY_POOL_TIMEOUT")
    sqlalchemy_pool_recycle: int = Field(default=3600, env="SQLALCHEMY_POOL_RECYCLE")

    # Milvus Configuration
    milvus_host: str = Field(default="localhost", env="MILVUS_HOST")
    milvus_port: int = Field(default=19530, env="MILVUS_PORT")
    milvus_user: str = Field(default="", env="MILVUS_USER")
    milvus_password: str = Field(default="", env="MILVUS_PASSWORD")
    milvus_database: str = Field(default="default", env="MILVUS_DATABASE")

    # Elasticsearch Configuration
    elasticsearch_host: str = Field(default="localhost", env="ELASTICSEARCH_HOST")
    elasticsearch_port: int = Field(default=9200, env="ELASTICSEARCH_PORT")
    elasticsearch_username: str = Field(default="", env="ELASTICSEARCH_USERNAME")
    elasticsearch_password: str = Field(default="", env="ELASTICSEARCH_PASSWORD")
    elasticsearch_index_prefix: str = Field(default="complex_rag", env="ELASTICSEARCH_INDEX_PREFIX")


class CacheConfig(BaseConfig):
    """Cache configuration."""

    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: str = Field(default="", env="REDIS_PASSWORD")
    redis_database: int = Field(default=0, env="REDIS_DATABASE")
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")

    # Cache Settings
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    memory_cache_size: int = Field(default=1000, env="MEMORY_CACHE_SIZE")


class StorageConfig(BaseConfig):
    """Storage configuration."""

    # MinIO Configuration
    minio_endpoint: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="", env="MINIO_SECRET_KEY")
    minio_secure: bool = Field(default=False, env="MINIO_SECURE")
    minio_region: str = Field(default="us-east-1", env="MINIO_REGION")

    # Storage Settings
    default_bucket: str = Field(default="complex-rag", env="DEFAULT_BUCKET")
    upload_max_size: int = Field(default=100 * 1024 * 1024, env="UPLOAD_MAX_SIZE")  # 100MB
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".md", ".json", ".csv", ".xlsx", ".pptx"],
        env="ALLOWED_EXTENSIONS"
    )


class AIModelsConfig(BaseConfig):
    """AI models configuration."""

    # OpenAI Configuration
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", env="OPENAI_BASE_URL")
    openai_organization: str = Field(default="", env="OPENAI_ORGANIZATION")
    openai_timeout: int = Field(default=60, env="OPENAI_TIMEOUT")

    # Anthropic Configuration
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    anthropic_base_url: str = Field(default="https://api.anthropic.com", env="ANTHROPIC_BASE_URL")
    anthropic_timeout: int = Field(default=60, env="ANTHROPIC_TIMEOUT")

    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_timeout: int = Field(default=300, env="OLLAMA_TIMEOUT")

    # Qwen Configuration
    qwen_api_key: str = Field(default="", env="QWEN_API_KEY")
    qwen_base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", env="QWEN_BASE_URL")
    qwen_timeout: int = Field(default=60, env="QWEN_TIMEOUT")

    # Baidu Configuration
    bce_api_key: str = Field(default="", env="BCE_API_KEY")
    bce_secret_key: str = Field(default="", env="BCE_SECRET_KEY")
    bce_base_url: str = Field(default="https://aip.baidubce.com", env="BCE_BASE_URL")
    bce_timeout: int = Field(default=60, env="BCE_TIMEOUT")


class Settings(
    APIConfig,
    RAGServiceConfig,
    DatabaseConfig,
    CacheConfig,
    StorageConfig,
    AIModelsConfig,
):
    """Main settings class that combines all configurations."""

    def get_database_url(self) -> str:
        """Get MySQL database URL."""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    def get_milvus_uri(self) -> str:
        """Get Milvus URI."""
        if self.milvus_password:
            return f"https://{self.milvus_user}:{self.milvus_password}@{self.milvus_host}:{self.milvus_port}"
        return f"http://{self.milvus_host}:{self.milvus_port}"

    def get_redis_url(self) -> str:
        """Get Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_database}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_database}"

    def get_elasticsearch_url(self) -> str:
        """Get Elasticsearch URL."""
        if self.elasticsearch_password:
            return f"https://{self.elasticsearch_username}:{self.elasticsearch_password}@{self.elasticsearch_host}:{self.elasticsearch_port}"
        return f"http://{self.elasticsearch_host}:{self.elasticsearch_port}"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()