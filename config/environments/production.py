"""
Production Environment Configuration

This module contains production-specific configuration settings.
"""

from ..settings import BaseConfig


class ProductionConfig(BaseConfig):
    """Production environment configuration."""

    # Environment
    environment: str = "production"
    debug: bool = False

    # Database
    mysql_host: str = "mysql-cluster.internal"
    mysql_port: int = 3306
    mysql_user: str = "rag_app"
    mysql_password: str = ""  # Set via environment variable
    mysql_database: str = "complex_rag"

    # Milvus
    milvus_host: str = "milvus-cluster.internal"
    milvus_port: int = 19530
    milvus_user: str = "rag_app"
    milvus_password: str = ""  # Set via environment variable
    milvus_secure: bool = True

    # Elasticsearch
    elasticsearch_host: str = "elasticsearch-cluster.internal"
    elasticsearch_port: int = 9200
    elasticsearch_username: str = "rag_app"
    elasticsearch_password: str = ""  # Set via environment variable
    elasticsearch_scheme: str = "https"
    elasticsearch_index_prefix: str = "complex_rag"

    # Redis
    redis_host: str = "redis-cluster.internal"
    redis_port: int = 6379
    redis_password: str = ""  # Set via environment variable
    redis_database: int = 0
    redis_ssl: bool = True

    # MinIO/S3
    minio_endpoint: str = "s3.amazonaws.com"
    minio_access_key: str = ""  # Set via environment variable
    minio_secret_key: str = ""  # Set via environment variable
    minio_secure: bool = True
    minio_region: str = "us-west-2"
    default_bucket: str = "complex-rag-production"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = False
    cors_origins: list[str] = [
        "https://app.example.com",
        "https://admin.example.com",
    ]
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # RAG Service Settings
    rag_host: str = "0.0.0.0"
    rag_port: int = 8001
    rag_workers: int = 4
    rag_debug: bool = False

    # AI Models (Production - use cloud services)
    default_llm_provider: str = "openai"
    default_embedding_provider: str = "openai"
    openai_api_key: str = ""  # Set via environment variable
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""  # Set via environment variable
    qwen_api_key: str = ""  # Set via environment variable
    bce_api_key: str = ""  # Set via environment variable

    # Logging
    log_level: str = "INFO"
    access_log: bool = True
    rag_access_log: bool = True
    api_access_log: bool = True
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"

    # Security
    secret_key: str = ""  # Set via environment variable - must be strong
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    api_key_header: str = "X-API-Key"
    api_keys: list[str] = []  # Set via environment variable

    # Performance
    sqlalchemy_echo: bool = False
    sqlalchemy_pool_size: int = 20
    sqlalchemy_max_overflow: int = 40
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    memory_cache_size: int = 10000

    # File Upload
    upload_max_size: int = 100 * 1024 * 1024  # 100MB
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

    # Error Handling
    send_errors_to_sentry: bool = True
    sentry_dsn: str = ""  # Set via environment variable
    detailed_error_messages: bool = False
    include_stack_traces: bool = False

    # Monitoring
    metrics_enabled: bool = True
    metrics_port: int = 9090
    tracing_enabled: bool = True
    tracing_endpoint: str = ""  # Set via environment variable
    health_check_enabled: bool = True
    health_check_interval: int = 30

    # Performance Monitoring
    enable_slow_query_log: bool = True
    slow_query_threshold: float = 0.1  # seconds
    enable_request_profiling: bool = True

    # SSL/TLS
    ssl_enabled: bool = True
    ssl_cert_path: str = "/etc/ssl/certs/app.crt"
    ssl_key_path: str = "/etc/ssl/private/app.key"

    # CORS
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["GET", "POST", "PUT", "DELETE"]
    cors_allow_headers: list[str] = [
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Request-ID",
    ]

    # Cache
    cache_backend: str = "redis"
    redis_cache_enabled: bool = True
    distributed_cache: bool = True

    # File Storage
    storage_provider: str = "s3"
    s3_endpoint: str = "s3.amazonaws.com"
    s3_region: str = "us-west-2"
    s3_bucket: str = "complex-rag-production"
    s3_secure: bool = True

    # Database Settings
    auto_migrate: bool = False
    backup_enabled: bool = True
    backup_retention_days: int = 30
    connection_pool_size: int = 20
    connection_max_overflow: int = 40

    # Email
    smtp_server: str = "smtp.example.com"
    smtp_port: int = 587
    smtp_use_tls: bool = True
    smtp_username: str = "noreply@example.com"
    smtp_password: str = ""  # Set via environment variable
    email_from: str = "noreply@example.com"
    email_backend: str = "smtp"

    # External Services
    external_service_timeout: int = 30
    external_service_retries: int = 3
    mock_external_services: bool = False

    # Security Headers
    security_headers_enabled: bool = True
    content_security_policy: str = "default-src 'self'"
    x_frame_options: str = "DENY"
    x_content_type_options: str = "nosniff"
    x_xss_protection: str = "1; mode=block"

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    ddos_protection: bool = True

    # Content Security
    input_validation_enabled: bool = True
    output_sanitization_enabled: bool = True
    xss_protection_enabled: bool = True
    sql_injection_protection: bool = True

    # Data Protection
    encryption_enabled: bool = True
    encryption_key: str = ""  # Set via environment variable
    data_retention_days: int = 365
    gdpr_compliance: bool = True

    # API Documentation
    swagger_ui_enabled: bool = False
    redoc_enabled: bool = False
    api_docs_enabled: bool = False

    # Feature Flags
    enable_experimental_features: bool = False
    enable_beta_features: bool = False
    feature_flag_provider: str = "launchdarkly"

    # Load Balancing
    load_balancer_enabled: bool = True
    session_affinity: bool = False
    health_check_path: str = "/health"

    # Autoscaling
    autoscaling_enabled: bool = True
    min_replicas: int = 2
    max_replicas: int = 20
    target_cpu_utilization: int = 70

    # Backup and Disaster Recovery
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    backup_retention_days: int = 30
    disaster_recovery_enabled: bool = True

    # Compliance and Auditing
    audit_logging_enabled: bool = True
    audit_retention_days: int = 2555  # 7 years
    compliance_checks_enabled: bool = True

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return True

    def get_database_url(self) -> str:
        """Get database URL for production."""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset=utf8mb4&ssl=true"
        )

    def get_redis_url(self) -> str:
        """Get Redis URL for production."""
        return (
            f"rediss://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_database}"
        )

    def get_elasticsearch_url(self) -> str:
        """Get Elasticsearch URL for production."""
        return (
            f"https://{self.elasticsearch_username}:{self.elasticsearch_password}"
            f"@{self.elasticsearch_host}:{self.elasticsearch_port}"
        )


# Global production configuration instance
production_config = ProductionConfig()