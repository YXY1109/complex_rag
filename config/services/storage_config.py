"""
Storage Configuration

This module contains configuration for object storage services.
"""

from pydantic import Field

from ..settings import BaseConfig


class StorageConfig(BaseConfig):
    """Storage configuration."""

    # MinIO Configuration
    minio_endpoint: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="", env="MINIO_SECRET_KEY")
    minio_secure: bool = Field(default=False, env="MINIO_SECURE")
    minio_region: str = Field(default="us-east-1", env="MINIO_REGION")
    minio_http_client: str = Field(default="urllib3", env="MINIO_HTTP_CLIENT")
    minio_timeout: int = Field(default=60, env="MINIO_TIMEOUT")

    # S3 Configuration (for AWS S3 compatibility)
    s3_endpoint: str = Field(default="", env="S3_ENDPOINT")
    s3_access_key: str = Field(default="", env="S3_ACCESS_KEY")
    s3_secret_key: str = Field(default="", env="S3_SECRET_KEY")
    s3_region: str = Field(default="us-east-1", env="S3_REGION")
    s3_bucket: str = Field(default="", env="S3_BUCKET")
    s3_secure: bool = Field(default=True, env="S3_SECURE")

    # Local Storage Configuration
    local_storage_path: str = Field(default="local_storage", env="LOCAL_STORAGE_PATH")
    local_storage_create_dirs: bool = Field(default=True, env="LOCAL_STORAGE_CREATE_DIRS")

    # Storage Provider Priority
    storage_provider: str = Field(default="minio", env="STORAGE_PROVIDER")  # minio, s3, local

    # Bucket Settings
    default_bucket: str = Field(default="complex-rag", env="DEFAULT_BUCKET")
    buckets: dict[str, str] = Field(
        default={
            "documents": "complex-rag-documents",
            "models": "complex-rag-models",
            "cache": "complex-rag-cache",
            "logs": "complex-rag-logs",
            "temp": "complex-rag-temp",
            "backups": "complex-rag-backups",
        },
        env="STORAGE_BUCKETS"
    )

    # File Upload Settings
    upload_max_size: int = Field(default=100 * 1024 * 1024, env="UPLOAD_MAX_SIZE")  # 100MB
    upload_chunk_size: int = Field(default=8 * 1024 * 1024, env="UPLOAD_CHUNK_SIZE")  # 8MB
    upload_timeout: int = Field(default=300, env="UPLOAD_TIMEOUT")
    upload_concurrent_limit: int = Field(default=5, env="UPLOAD_CONCURRENT_LIMIT")

    # Allowed File Types
    allowed_extensions: list[str] = Field(
        default=[
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            ".txt", ".md", ".rtf", ".html", ".htm", ".xml", ".json",
            ".csv", ".tsv", ".yaml", ".yml", ".ini", ".toml", ".cfg",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
            ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a",
            ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mkv",
            ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"
        ],
        env="ALLOWED_EXTENSIONS"
    )

    # MIME Type Settings
    allowed_mime_types: list[str] = Field(
        default=[
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "text/plain",
            "text/markdown",
            "text/html",
            "text/xml",
            "application/json",
            "text/csv",
            "application/xml",
            "application/yaml",
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/bmp",
            "image/tiff",
            "image/webp",
            "audio/mpeg",
            "audio/wav",
            "audio/flac",
            "audio/aac",
            "audio/ogg",
            "video/mp4",
            "video/avi",
            "video/mov",
            "video/x-msvideo",
            "video/x-flv",
            "video/webm",
            "video/x-matroska",
            "application/zip",
            "application/x-rar-compressed",
            "application/x-7z-compressed",
            "application/x-tar",
            "application/gzip",
            "application/x-bzip2",
            "application/x-xz"
        ],
        env="ALLOWED_MIME_TYPES"
    )

    # File Naming Settings
    file_naming_strategy: str = Field(default="uuid", env="FILE_NAMING_STRATEGY")  # uuid, timestamp, original
    preserve_original_name: bool = Field(default=True, env="PRESERVE_ORIGINAL_NAME")
    file_prefix: str = Field(default="", env="FILE_PREFIX")
    file_suffix: str = Field(default="", env="FILE_SUFFIX")

    # Path Organization
    organize_by_date: bool = Field(default=True, env="ORGANIZE_BY_DATE")
    organize_by_type: bool = Field(default=True, env="ORGANIZE_BY_TYPE")
    organize_by_user: bool = Field(default=False, env="ORGANIZE_BY_USER")
    date_format: str = Field(default="%Y/%m/%d", env="DATE_FORMAT")

    # File Versioning
    versioning_enabled: bool = Field(default=False, env="VERSIONING_ENABLED")
    max_versions: int = Field(default=10, env="MAX_VERSIONS")
    versioning_strategy: str = Field(default="timestamp", env="VERSIONING_STRATEGY")  # timestamp, sequential

    # File Compression
    compression_enabled: bool = Field(default=False, env="COMPRESSION_ENABLED")
    compression_level: int = Field(default=6, env="COMPRESSION_LEVEL")
    compressible_types: list[str] = Field(
        default=["text/plain", "text/html", "text/xml", "application/json", "text/csv"],
        env="COMPRESSIBLE_TYPES"
    )

    # File Encryption
    encryption_enabled: bool = Field(default=False, env="ENCRYPTION_ENABLED")
    encryption_algorithm: str = Field(default="AES-256-GCM", env="ENCRYPTION_ALGORITHM")
    encryptible_types: list[str] = Field(
        default=[],
        env="ENCRYPTIBLE_TYPES"
    )

    # Cache Settings
    cache_enabled: bool = Field(default=True, env="STORAGE_CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, env="STORAGE_CACHE_TTL")  # 1 hour
    cache_size_limit: int = Field(default=1024 * 1024 * 1024, env="STORAGE_CACHE_SIZE_LIMIT")  # 1GB

    # Cleanup Settings
    cleanup_enabled: bool = Field(default=True, env="CLEANUP_ENABLED")
    cleanup_interval: int = Field(default=86400, env="CLEANUP_INTERVAL")  # 24 hours
    temp_file_ttl: int = Field(default=3600, env="TEMP_FILE_TTL")  # 1 hour
    log_file_ttl: int = Field(default=604800, env="LOG_FILE_TTL")  # 7 days
    backup_retention_days: int = Field(default=30, env="BACKUP_RETENTION_DAYS")

    # Monitoring Settings
    metrics_enabled: bool = Field(default=True, env="STORAGE_METRICS_ENABLED")
    access_log_enabled: bool = Field(default=True, env="STORAGE_ACCESS_LOG_ENABLED")
    storage_usage_alert_threshold: float = Field(default=0.8, env="STORAGE_USAGE_ALERT_THRESHOLD")

    def get_bucket_name(self, category: str) -> str:
        """Get bucket name for category."""
        return self.buckets.get(category, self.default_bucket)

    def get_object_path(self, category: str, filename: str, user_id: str = None, timestamp: str = None) -> str:
        """Get object path with organization."""
        path_parts = []

        # Add category
        path_parts.append(category)

        # Add date organization
        if self.organize_by_date and timestamp:
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            date_path = dt.strftime(self.date_format)
            path_parts.append(date_path)

        # Add user organization
        if self.organize_by_user and user_id:
            path_parts.append(f"user_{user_id}")

        # Add file type organization
        if self.organize_by_type:
            import os
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.pdf', '.doc', '.docx']:
                path_parts.append('documents')
            elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                path_parts.append('images')
            elif ext in ['.mp3', '.wav', '.flac', '.aac']:
                path_parts.append('audio')
            elif ext in ['.mp4', '.avi', '.mov', '.wmv']:
                path_parts.append('video')
            elif ext in ['.zip', '.rar', '.7z', '.tar']:
                path_parts.append('archives')
            else:
                path_parts.append('others')

        # Add filename
        if self.file_prefix:
            filename = f"{self.file_prefix}{filename}"
        if self.file_suffix:
            name, ext = os.path.splitext(filename)
            filename = f"{name}{self.file_suffix}{ext}"

        path_parts.append(filename)

        return "/".join(path_parts)

    def get_provider_config(self, provider: str) -> dict:
        """Get configuration for specific storage provider."""
        configs = {
            "minio": {
                "endpoint": self.minio_endpoint,
                "access_key": self.minio_access_key,
                "secret_key": self.minio_secret_key,
                "secure": self.minio_secure,
                "region": self.minio_region,
                "http_client": self.minio_http_client,
                "timeout": self.minio_timeout,
            },
            "s3": {
                "endpoint_url": self.s3_endpoint,
                "aws_access_key_id": self.s3_access_key,
                "aws_secret_access_key": self.s3_secret_key,
                "region_name": self.s3_region,
                "use_ssl": self.s3_secure,
            },
            "local": {
                "storage_path": self.local_storage_path,
                "create_dirs": self.local_storage_create_dirs,
            },
        }
        return configs.get(provider, {})


# Global storage configuration instance
storage_config = StorageConfig()