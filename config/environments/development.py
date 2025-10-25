"""
Development Environment Configuration

This module contains development-specific configuration settings.
"""

from ..settings import BaseConfig


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""

    # Environment
    environment: str = "development"
    debug: bool = True

    # Database
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "complex_rag_dev"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_database: str = "development"

    # Elasticsearch
    elasticsearch_host: str = "localhost"
    elasticsearch_port: int = 9200
    elasticsearch_index_prefix: str = "complex_rag_dev"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_database: int = 0

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    default_bucket: str = "complex-rag-dev"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_reload: bool = True
    cors_origins: list[str] = ["*"]
    rate_limit_enabled: bool = False

    # RAG Service Settings
    rag_host: str = "0.0.0.0"
    rag_port: int = 8001
    rag_workers: int = 1
    rag_debug: bool = True

    # AI Models (Development - use local models when possible)
    default_llm_provider: str = "ollama"
    default_embedding_provider: str = "local"
    openai_api_key: str = ""  # Add your key for testing
    anthropic_api_key: str = ""  # Add your key for testing
    qwen_api_key: str = ""  # Add your key for testing
    bce_api_key: str = ""  # Add your key for testing

    # Ollama (Local LLM)
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 300
    ollama_keep_alive: str = "1h"

    # Logging
    log_level: str = "DEBUG"
    access_log: bool = True
    rag_access_log: bool = True
    api_access_log: bool = True

    # Security
    secret_key: str = "dev-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours for development

    # Performance
    sqlalchemy_echo: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5 minutes
    memory_cache_size: int = 100

    # File Upload
    upload_max_size: int = 50 * 1024 * 1024  # 50MB for development
    upload_dir: str = "uploads"
    allowed_file_types: list[str] = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/plain",
        "text/markdown",
        "application/json",
        "text/csv",
        "image/jpeg",
        "image/png",
        "image/gif",
    ]

    # Testing
    testing_enabled: bool = True
    test_database: str = "complex_rag_test"
    test_redis_database: int = 1
    test_milvus_database: str = "test"

    # Debug Settings
    debug_toolbar: bool = True
    debug_sql: bool = True
    debug_requests: bool = True
    debug_templates: bool = True

    # Monitoring
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    health_check_enabled: bool = True

    # Development Tools
    profiler_enabled: bool = True
    swagger_ui_enabled: bool = True
    redoc_enabled: bool = True
    api_docs_enabled: bool = True

    # Hot Reload
    auto_reload: bool = True
    reload_dirs: list[str] = ["api", "rag_service", "document_parser", "core_rag", "infrastructure", "config"]

    # Development Features
    mock_external_services: bool = True
    enable_admin_interface: bool = True
    enable_debug_endpoints: bool = True

    # Error Handling
    send_errors_to_sentry: bool = False
    detailed_error_messages: bool = True
    include_stack_traces: bool = True

    # Performance Monitoring
    enable_slow_query_log: bool = True
    slow_query_threshold: float = 0.5  # seconds
    enable_request_profiling: bool = True

    # CORS
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
    cors_allow_headers: list[str] = ["*"]

    # Rate Limiting
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 1000
    rate_limit_window: int = 60

    # Cache
    cache_backend: str = "memory"  # Use memory cache for development
    redis_cache_enabled: bool = False

    # File Storage
    storage_provider: str = "local"
    local_storage_path: str = "local_storage"
    minio_enabled: bool = False

    # Database Settings
    auto_migrate: bool = True
    create_sample_data: bool = True
    seed_database: bool = True

    # Email (for development testing)
    smtp_server: str = "localhost"
    smtp_port: int = 1025
    smtp_use_tls: bool = False
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = "dev@example.com"
    email_backend: str = "console"  # Print emails to console

    # External Services
    external_service_timeout: int = 30
    external_service_retries: int = 3
    mock_external_services: bool = True

    # Feature Flags
    enable_experimental_features: bool = True
    enable_beta_features: bool = True
    feature_flag_provider: str = "local"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return True

    def get_database_url(self) -> str:
        """Get database URL for development."""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset=utf8mb4"
        )

    def get_test_database_url(self) -> str:
        """Get test database URL."""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.test_database}"
            f"?charset=utf8mb4"
        )


# Global development configuration instance
development_config = DevelopmentConfig()