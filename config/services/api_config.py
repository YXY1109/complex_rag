"""
API Service Configuration

This module contains configuration specific to the FastAPI API service.
"""

from pydantic import Field

from ..settings import BaseConfig


class APIServiceConfig(BaseConfig):
    """API service specific configuration."""

    # FastAPI Server Settings
    title: str = Field(default="Complex RAG API", env="API_TITLE")
    description: str = Field(
        default="Modern, high-performance, enterprise-grade RAG system API",
        env="API_DESCRIPTION"
    )
    version: str = Field(default="0.1.0", env="API_VERSION")
    docs_url: str = Field(default="/docs", env="API_DOCS_URL")
    redoc_url: str = Field(default="/redoc", env="API_REDOC_URL")
    openapi_url: str = Field(default="/openapi.json", env="API_OPENAPI_URL")

    # Server Settings
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")
    reload: bool = Field(default=False, env="API_RELOAD")
    log_level: str = Field(default="info", env="API_LOG_LEVEL")

    # CORS Settings
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: list[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: list[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")

    # Request Settings
    max_request_size: int = Field(default=16 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 16MB
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")
    keep_alive_timeout: int = Field(default=65, env="KEEP_ALIVE_TIMEOUT")

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds

    # API Keys (if authentication is enabled)
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    api_keys: list[str] = Field(default=[], env="API_KEYS")

    # File Upload Settings
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    allowed_file_types: list[str] = Field(
        default=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "text/plain",
            "text/markdown",
            "application/json",
            "text/csv"
        ],
        env="ALLOWED_FILE_TYPES"
    )

    # Pagination Settings
    default_page_size: int = Field(default=20, env="DEFAULT_PAGE_SIZE")
    max_page_size: int = Field(default=1000, env="MAX_PAGE_SIZE")

    # Logging Settings
    access_log: bool = Field(default=True, env="API_ACCESS_LOG")
    log_format: str = Field(
        default='{time:YYYY-MM-DD HH:mm:ss} | {level} | {client} | {method} {path} | {status} | {duration}',
        env="API_LOG_FORMAT"
    )

    # Cache Settings
    cache_enabled: bool = Field(default=True, env="API_CACHE_ENABLED")
    cache_ttl: int = Field(default=300, env="API_CACHE_TTL")  # 5 minutes

    # Health Check Settings
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    health_check_timeout: int = Field(default=5, env="HEALTH_CHECK_TIMEOUT")

    @property
    def server_url(self) -> str:
        """Get server URL."""
        return f"http://{self.host}:{self.port}"

    @property
    def docs_url_full(self) -> str:
        """Get full documentation URL."""
        return f"{self.server_url}{self.docs_url}"


# Global API configuration instance
api_config = APIServiceConfig()