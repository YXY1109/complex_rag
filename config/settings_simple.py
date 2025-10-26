"""
简化的应用设置
为统一FastAPI服务提供配置
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置设置"""

    # 应用基本配置
    app_name: str = "Complex RAG API"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True

    # 服务器配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # 日志配置
    log_level: str = "INFO"
    log_dir: str = "logs"

    # 数据库配置
    database_url: Optional[str] = None

    # Redis配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None

    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: Optional[str] = None
    milvus_password: Optional[str] = None

    # Elasticsearch配置
    elasticsearch_host: str = "localhost"
    elasticsearch_port: int = 9200
    elasticsearch_username: Optional[str] = None
    elasticsearch_password: Optional[str] = None

    # MinIO配置
    minio_endpoint: str = "localhost:9000"
    minio_access_key: Optional[str] = None
    minio_secret_key: Optional[str] = None
    minio_bucket_name: str = "rag-storage"

    # AI模型配置
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"

    # 向量模型配置
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimensions: int = 1536

    # 重排序模型配置
    rerank_model: str = "bge-reranker-base"

    # LLM配置
    default_llm_model: str = "gpt-3.5-turbo"
    max_tokens: int = 2048
    temperature: float = 0.7

    # 安全配置
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30

    # CORS配置
    cors_origins: list = ["*"]
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]

    # 性能配置
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 30  # 30秒

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # 忽略额外的环境变量


@lru_cache()
def get_settings() -> Settings:
    """获取应用设置（带缓存）"""
    return Settings()


# 全局设置实例
settings = get_settings()