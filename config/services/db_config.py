"""
Database Configuration

This module contains configuration for all database services.
"""

from pydantic import Field

from ..settings import BaseConfig


class DatabaseConfig(BaseConfig):
    """Database configuration."""

    # MySQL Configuration
    mysql_host: str = Field(default="localhost", env="MYSQL_HOST")
    mysql_port: int = Field(default=3306, env="MYSQL_PORT")
    mysql_user: str = Field(default="root", env="MYSQL_USER")
    mysql_password: str = Field(default="", env="MYSQL_PASSWORD")
    mysql_database: str = Field(default="complex_rag", env="MYSQL_DATABASE")
    mysql_charset: str = Field(default="utf8mb4", env="MYSQL_CHARSET")

    # SQLAlchemy Settings
    sqlalchemy_echo: bool = Field(default=False, env="SQLALCHEMY_ECHO")
    sqlalchemy_pool_size: int = Field(default=10, env="SQLALCHEMY_POOL_SIZE")
    sqlalchemy_max_overflow: int = Field(default=20, env="SQLALCHEMY_MAX_OVERFLOW")
    sqlalchemy_pool_timeout: int = Field(default=30, env="SQLALCHEMY_POOL_TIMEOUT")
    sqlalchemy_pool_recycle: int = Field(default=3600, env="SQLALCHEMY_POOL_RECYCLE")
    sqlalchemy_pool_pre_ping: bool = Field(default=True, env="SQLALCHEMY_POOL_PRE_PING")

    # Milvus Configuration
    milvus_host: str = Field(default="localhost", env="MILVUS_HOST")
    milvus_port: int = Field(default=19530, env="MILVUS_PORT")
    milvus_user: str = Field(default="", env="MILVUS_USER")
    milvus_password: str = Field(default="", env="MILVUS_PASSWORD")
    milvus_database: str = Field(default="default", env="MILVUS_DATABASE")
    milvus_secure: bool = Field(default=False, env="MILVUS_SECURE")
    milvus_timeout: int = Field(default=10, env="MILVUS_TIMEOUT")

    # Elasticsearch Configuration
    elasticsearch_host: str = Field(default="localhost", env="ELASTICSEARCH_HOST")
    elasticsearch_port: int = Field(default=9200, env="ELASTICSEARCH_PORT")
    elasticsearch_username: str = Field(default="", env="ELASTICSEARCH_USERNAME")
    elasticsearch_password: str = Field(default="", env="ELASTICSEARCH_PASSWORD")
    elasticsearch_scheme: str = Field(default="http", env="ELASTICSEARCH_SCHEME")
    elasticsearch_index_prefix: str = Field(default="complex_rag", env="ELASTICSEARCH_INDEX_PREFIX")
    elasticsearch_timeout: int = Field(default=30, env="ELASTICSEARCH_TIMEOUT")
    elasticsearch_max_retries: int = Field(default=3, env="ELASTICSEARCH_MAX_RETRIES")
    elasticsearch_retry_on_timeout: bool = Field(default=True, env="ELASTICSEARCH_RETRY_ON_TIMEOUT")

    # Database Connection Settings
    connection_max_attempts: int = Field(default=3, env="CONNECTION_MAX_ATTEMPTS")
    connection_retry_delay: int = Field(default=5, env="CONNECTION_RETRY_DELAY")  # seconds
    connection_health_check: bool = Field(default=True, env="CONNECTION_HEALTH_CHECK")
    connection_health_check_interval: int = Field(default=30, env="CONNECTION_HEALTH_CHECK_INTERVAL")

    # Database Migration Settings
    auto_migrate: bool = Field(default=False, env="AUTO_MIGRATE")
    migration_timeout: int = Field(default=300, env="MIGRATION_TIMEOUT")
    backup_before_migrate: bool = Field(default=True, env="BACKUP_BEFORE_MIGRATE")

    # Vector Database Settings
    vector_index_type: str = Field(default="HNSW", env="VECTOR_INDEX_TYPE")
    vector_metric_type: str = Field(default="IP", env="VECTOR_METRIC_TYPE")  # IP (Inner Product), L2, COSINE
    vector_index_params: dict = Field(
        default={"M": 16, "efConstruction": 256},
        env="VECTOR_INDEX_PARAMS"
    )
    vector_search_params: dict = Field(
        default={"ef": 64},
        env="VECTOR_SEARCH_PARAMS"
    )

    # Search Index Settings
    search_index_shards: int = Field(default=1, env="SEARCH_INDEX_SHARDS")
    search_index_replicas: int = Field(default=0, env="SEARCH_INDEX_REPLICAS")
    search_refresh_interval: str = Field(default="1s", env="SEARCH_REFRESH_INTERVAL")
    search_max_result_window: int = Field(default=10000, env="SEARCH_MAX_RESULT_WINDOW")

    # Database Pool Settings
    pool_recycle: int = Field(default=3600, env="POOL_RECYCLE")
    pool_pre_ping: bool = Field(default=True, env="POOL_PRE_PING")
    pool_reset_on_return: str = Field(default="commit", env="POOL_RESET_ON_RETURN")

    def get_mysql_url(self) -> str:
        """Get MySQL database URL."""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset={self.mysql_charset}"
        )

    def get_mysql_url_async(self) -> str:
        """Get MySQL async database URL."""
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset={self.mysql_charset}"
        )

    def get_milvus_uri(self) -> str:
        """Get Milvus URI."""
        if self.milvus_secure:
            if self.milvus_password:
                return f"https://{self.milvus_user}:{self.milvus_password}@{self.milvus_host}:{self.milvus_port}"
            return f"https://{self.milvus_host}:{self.milvus_port}"
        else:
            if self.milvus_password:
                return f"http://{self.milvus_user}:{self.milvus_password}@{self.milvus_host}:{self.milvus_port}"
            return f"http://{self.milvus_host}:{self.milvus_port}"

    def get_elasticsearch_hosts(self) -> list[dict]:
        """Get Elasticsearch hosts configuration."""
        host_config = {
            "host": self.elasticsearch_host,
            "port": self.elasticsearch_port,
            "scheme": self.elasticsearch_scheme,
            "timeout": self.elasticsearch_timeout,
            "max_retries": self.elasticsearch_max_retries,
            "retry_on_timeout": self.elasticsearch_retry_on_timeout,
        }

        if self.elasticsearch_username and self.elasticsearch_password:
            host_config["http_auth"] = (self.elasticsearch_username, self.elasticsearch_password)

        return [host_config]

    def get_engine_kwargs(self) -> dict:
        """Get SQLAlchemy engine keyword arguments."""
        return {
            "echo": self.sqlalchemy_echo,
            "pool_size": self.sqlalchemy_pool_size,
            "max_overflow": self.sqlalchemy_max_overflow,
            "pool_timeout": self.sqlalchemy_pool_timeout,
            "pool_recycle": self.sqlalchemy_pool_recycle,
            "pool_pre_ping": self.sqlalchemy_pool_pre_ping,
        }


# Global database configuration instance
db_config = DatabaseConfig()