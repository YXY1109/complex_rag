"""
Testing Environment Configuration

This module contains testing-specific configuration settings.
"""

from ..settings import BaseConfig


class TestingConfig(BaseConfig):
    """Testing environment configuration."""

    # Environment
    environment: str = "testing"
    debug: bool = True

    # Database (Use in-memory or temporary databases)
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "test_user"
    mysql_password: str = "test_password"
    mysql_database: str = "complex_rag_test"

    # Milvus (Test instance)
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_database: str = "test"

    # Elasticsearch (Test instance)
    elasticsearch_host: str = "localhost"
    elasticsearch_port: int = 9200
    elasticsearch_index_prefix: str = "complex_rag_test"

    # Redis (Test database)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_database: int = 1  # Use database 1 for tests

    # MinIO (Test bucket)
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "testuser"
    minio_secret_key: str = "testpassword"
    minio_secure: bool = False
    default_bucket: str = "complex-rag-test"

    # API Settings
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_workers: int = 1
    api_reload: bool = False
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    rate_limit_enabled: bool = False

    # RAG Service Settings
    rag_host: str = "127.0.0.1"
    rag_port: int = 8001
    rag_workers: int = 1
    rag_debug: bool = True

    # AI Models (Use mock or local models for testing)
    default_llm_provider: str = "mock"
    default_embedding_provider: str = "mock"
    openai_api_key: str = "test-key"
    anthropic_api_key: str = "test-key"
    qwen_api_key: str = "test-key"
    bce_api_key: str = "test-key"

    # Ollama (Test instance)
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 30  # Shorter timeout for tests
    ollama_keep_alive: str = "5m"

    # Logging
    log_level: str = "DEBUG"
    access_log: bool = False  # Disable access logs for cleaner test output
    rag_access_log: bool = False
    api_access_log: bool = False

    # Security
    secret_key: str = "test-secret-key-for-testing-only"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60  # Shorter for tests

    # Performance
    sqlalchemy_echo: bool = False  # Disable for cleaner test output
    cache_enabled: bool = False  # Disable cache for tests
    memory_cache_size: int = 100

    # File Upload
    upload_max_size: int = 10 * 1024 * 1024  # 10MB for tests
    upload_dir: str = "test_uploads"
    allowed_file_types: list[str] = [
        "text/plain",
        "application/json",
        "text/csv",
        "text/markdown",
    ]

    # Testing Settings
    testing_enabled: bool = True
    test_database: str = "complex_rag_test"
    test_redis_database: int = 1
    test_milvus_database: str = "test"
    test_elasticsearch_prefix: str = "test_"

    # Mock Settings
    mock_external_services: bool = True
    mock_ai_responses: bool = True
    mock_database: bool = False  # Use real database for integration tests
    mock_storage: bool = True

    # Test Data
    create_test_data: bool = True
    cleanup_test_data: bool = True
    test_data_seed: int = 42

    # Test Timeouts
    test_timeout: int = 300  # 5 minutes
    short_test_timeout: int = 30  # 30 seconds
    integration_test_timeout: int = 600  # 10 minutes

    # Test Isolation
    use_test_database: bool = True
    use_test_redis: bool = True
    use_test_storage: bool = True
    isolated_test_environment: bool = True

    # Coverage
    coverage_enabled: bool = True
    coverage_threshold: float = 80.0
    coverage_report_format: str = "html"

    # Performance Testing
    performance_tests_enabled: bool = True
    load_test_enabled: bool = False  # Disabled by default
    performance_threshold_ms: float = 1000.0

    # Database Testing
    test_database_reset: bool = True
    use_database_transactions: bool = True
    rollback_after_test: bool = True

    # Test Parallelization
    parallel_tests: bool = True
    max_test_workers: int = 4
    test_parallelism_mode: str = "thread"

    # Test Fixtures
    auto_use_fixtures: list[str] = [
        "test_database",
        "test_redis",
        "test_client",
        "test_storage",
    ]

    # Test Scopes
    default_test_scope: str = "function"
    session_scope_fixtures: list[str] = [
        "test_database",
        "test_redis",
        "test_storage",
    ]

    # Test Data Management
    test_data_factories: bool = True
    use_factory_boy: bool = True
    use_faker: bool = True
    test_data_locale: str = "en_US"

    # API Testing
    test_api_client: bool = True
    test_authentication: bool = True
    test_authorization: bool = True
    test_rate_limiting: bool = False

    # Integration Testing
    integration_tests_enabled: bool = True
    external_service_tests: bool = False  # Requires real services
    database_integration_tests: bool = True
    cache_integration_tests: bool = True

    # Error Testing
    test_error_handling: bool = True
    test_timeouts: bool = True
    test_failures: bool = True
    test_edge_cases: bool = True

    # Security Testing
    security_tests_enabled: bool = True
    input_validation_tests: bool = True
    authentication_tests: bool = True
    authorization_tests: bool = True

    # Monitoring and Debugging
    debug_test_failures: bool = True
    capture_test_output: bool = True
    test_screenshots: bool = False  # For UI tests
    test_network_logs: bool = True

    # CI/CD Integration
    ci_mode: bool = False
    ci_test_timeout: int = 1800  # 30 minutes
    ci_parallel_workers: int = 2

    # Test Reporting
    generate_test_report: bool = True
    test_report_format: str = "html"
    include_test_coverage: bool = True
    include_test_performance: bool = True

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return True

    def get_database_url(self) -> str:
        """Get database URL for testing."""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset=utf8mb4"
        )

    def get_test_storage_path(self) -> str:
        """Get test storage path."""
        import tempfile
        import os

        test_dir = os.path.join(tempfile.gettempdir(), "complex_rag_tests")
        os.makedirs(test_dir, exist_ok=True)
        return test_dir

    def get_test_upload_dir(self) -> str:
        """Get test upload directory."""
        import tempfile
        import os

        upload_dir = os.path.join(tempfile.gettempdir(), "complex_rag_test_uploads")
        os.makedirs(upload_dir, exist_ok=True)
        return upload_dir


# Global testing configuration instance
testing_config = TestingConfig()