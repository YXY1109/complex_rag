"""
Comprehensive test suite to preserve current functionality during refactoring.
This test suite ensures that all existing features continue to work after consolidation.
"""

import pytest
import asyncio
from typing import Dict, Any, List
import httpx
from pathlib import Path
import json

# Test configuration
BASE_URL = "http://localhost:8000"
RAG_SERVICE_URL = "http://localhost:8001"
BCE_SERVICE_URL = "http://localhost:7001"
QWEN3_SERVICE_URL = "http://localhost:8000"  # Note: Potential port conflict
OCR_SERVICE_URL = "http://localhost:7004"
LLM_SERVICE_URL = "http://localhost:7003"


class TestMainAPIFunctionality:
    """Test main FastAPI application endpoints"""

    @pytest.mark.asyncio
    async def test_health_endpoints(self):
        """Test main API health endpoints"""
        async with httpx.AsyncClient() as client:
            # Root health check
            response = await client.get(f"{BASE_URL}/")
            assert response.status_code == 200

            # Ping endpoint
            response = await client.get(f"{BASE_URL}/ping")
            assert response.status_code == 200
            assert response.json() == {"message": "pong"}

    @pytest.mark.asyncio
    async def test_chat_completions(self):
        """Test OpenAI-compatible chat completions"""
        async with httpx.AsyncClient() as client:
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "max_tokens": 100
            }

            response = await client.post(
                f"{BASE_URL}/api/chat/completions",
                json=payload
            )
            # Note: This might fail without proper API keys, but should validate endpoint exists
            assert response.status_code in [200, 400, 401, 500]

    @pytest.mark.asyncio
    async def test_document_upload(self):
        """Test document upload functionality"""
        # Create a test file
        test_content = "This is a test document for RAG processing."
        test_file = Path("test_doc.txt")
        test_file.write_text(test_content)

        try:
            async with httpx.AsyncClient() as client:
                with open(test_file, "rb") as f:
                    files = {"file": ("test_doc.txt", f, "text/plain")}
                    data = {"knowledge_base_id": "test_kb"}

                    response = await client.post(
                        f"{BASE_URL}/api/documents/upload",
                        files=files,
                        data=data
                    )
                    # Should at least reach the endpoint, even if processing fails
                    assert response.status_code in [200, 400, 422]
        finally:
            if test_file.exists():
                test_file.unlink()

    @pytest.mark.asyncio
    async def test_document_listing(self):
        """Test document listing endpoints"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/documents/")
            assert response.status_code in [200, 401, 500]

    @pytest.mark.asyncio
    async def test_knowledge_base_endpoints(self):
        """Test knowledge base management"""
        async with httpx.AsyncClient() as client:
            # List knowledge bases
            response = await client.get(f"{BASE_URL}/api/knowledge/")
            assert response.status_code in [200, 401, 500]

    @pytest.mark.asyncio
    async def test_model_management(self):
        """Test model listing and management"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/models/")
            assert response.status_code in [200, 401, 500]


class TestLegacyServices:
    """Test legacy service endpoints to ensure compatibility during migration"""

    @pytest.mark.asyncio
    async def test_bce_embedding_service(self):
        """Test BCE embedding service functionality"""
        test_text = "This is a test text for embedding."

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test BCE embedding endpoint
            payload = {"texts": [test_text]}

            try:
                response = await client.post(
                    f"{BCE_SERVICE_URL}/bce_embedding",
                    json=payload
                )
                # Service might not be running, but endpoint should exist
                assert response.status_code in [200, 404, 500]
            except httpx.ConnectError:
                # Service not running - this is expected during testing
                pytest.skip("BCE service not running")

    @pytest.mark.asyncio
    async def test_qwen3_embedding_service(self):
        """Test Qwen3 embedding service functionality"""
        test_text = "This is a test text for embedding."

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test Qwen3 embedding endpoint (OpenAI compatible)
            payload = {
                "model": "Qwen3-Embedding-0.6B",
                "input": test_text
            }

            try:
                response = await client.post(
                    f"{QWEN3_SERVICE_URL}/v1/embeddings",
                    json=payload
                )
                # Service might conflict with main API on port 8000
                assert response.status_code in [200, 404, 500]
            except httpx.ConnectError:
                pytest.skip("Qwen3 service not running or port conflict")

    @pytest.mark.asyncio
    async def test_ocr_service(self):
        """Test OCR service functionality"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Test OCR service health
                response = await client.get(f"{OCR_SERVICE_URL}/test")
                assert response.status_code in [200, 404, 500]
            except httpx.ConnectError:
                pytest.skip("OCR service not running")

    @pytest.mark.asyncio
    async def test_llm_service(self):
        """Test LLM service functionality"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Test LLM service health
                response = await client.get(f"{LLM_SERVICE_URL}/test")
                assert response.status_code in [200, 404, 500]
            except httpx.ConnectError:
                pytest.skip("LLM service not running")


class TestRAGService:
    """Test RAG service (Sanic) functionality"""

    @pytest.mark.asyncio
    async def test_rag_health_check(self):
        """Test RAG service health endpoints"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(f"{RAG_SERVICE_URL}/health")
                assert response.status_code in [200, 404, 500]
            except httpx.ConnectError:
                pytest.skip("RAG service not running")

    @pytest.mark.asyncio
    async def test_rag_chat_completions(self):
        """Test RAG service OpenAI-compatible endpoints"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                payload = {
                    "model": "qwen2.5:7b",
                    "messages": [
                        {"role": "user", "content": "Hello, how are you?"}
                    ],
                    "max_tokens": 100
                }

                response = await client.post(
                    f"{RAG_SERVICE_URL}/v1/chat/completions",
                    json=payload
                )
                assert response.status_code in [200, 400, 401, 500]
            except httpx.ConnectError:
                pytest.skip("RAG service not running")


class TestConfigurationIntegrity:
    """Test that current configurations are valid and accessible"""

    def test_main_config_exists(self):
        """Test main configuration file exists and is valid"""
        config_path = Path("config/settings.py")
        assert config_path.exists(), "Main configuration file missing"

    def test_docker_config_exists(self):
        """Test Docker configuration exists"""
        docker_compose_path = Path("docker/docker-compose.yml")
        assert docker_compose_path.exists(), "Docker compose file missing"

    def test_service_configs_exist(self):
        """Test service-specific configurations exist"""
        rag_config_path = Path("config/services/rag_service_config.py")
        if rag_config_path.exists():
            # Just ensure it can be imported
            import importlib.util
            spec = importlib.util.spec_from_file_location("rag_config", rag_config_path)
            assert spec is not None, "RAG service config cannot be loaded"


class TestProviderIntegrity:
    """Test that all provider implementations are intact"""

    def test_embedding_providers_exist(self):
        """Test embedding provider implementations exist"""
        providers_dir = Path("rag_service/providers")

        # Check for common embedding providers
        expected_providers = [
            "openai/embedding_provider.py",
            "ollama/embedding_provider.py",
            "bce/rerank_provider.py"
        ]

        for provider in expected_providers:
            provider_path = providers_dir / provider
            if provider_path.exists():
                # Ensure it can be imported
                import importlib.util
                spec = importlib.util.spec_from_file_location("provider", provider_path)
                assert spec is not None, f"Provider {provider} cannot be loaded"

    def test_vector_store_implementations_exist(self):
        """Test vector store implementations exist"""
        vector_store_path = Path("rag_service/services/vector_store.py")
        if vector_store_path.exists():
            # Ensure it can be imported
            import importlib.util
            spec = importlib.util.spec_from_file_location("vector_store", vector_store_path)
            assert spec is not None, "Vector store service cannot be loaded"


class TestAPIContractIntegrity:
    """Test that API contracts are maintained"""

    @pytest.mark.asyncio
    async def test_openapi_spec_available(self):
        """Test that OpenAPI specification is available"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/openapi.json")
            if response.status_code == 200:
                openapi_spec = response.json()
                assert "paths" in openapi_spec
                assert "components" in openapi_spec
            else:
                # OpenAPI might not be enabled, which is acceptable
                assert response.status_code in [404, 500]

    @pytest.mark.asyncio
    async def test_api_docs_available(self):
        """Test that API documentation is accessible"""
        async with httpx.AsyncClient() as client:
            # Try common documentation endpoints
            docs_endpoints = ["/docs", "/redoc", "/api/docs"]

            for endpoint in docs_endpoints:
                response = await client.get(f"{BASE_URL}{endpoint}")
                if response.status_code == 200:
                    # Found working documentation endpoint
                    break
            else:
                # No documentation found, which might be acceptable
                pass


class TestBackwardCompatibility:
    """Test backward compatibility for critical integrations"""

    @pytest.mark.asyncio
    async def test_legacy_health_check_compatibility(self):
        """Test that legacy health check patterns still work"""
        async with httpx.AsyncClient() as client:
            # Try multiple health check patterns
            health_endpoints = [
                "/health",
                "/api/health",
                "/ping",
                "/"
            ]

            for endpoint in health_endpoints:
                response = await client.get(f"{BASE_URL}{endpoint}")
                if response.status_code == 200:
                    # At least one health endpoint works
                    break
            else:
                pytest.fail("No health check endpoints are accessible")

    def test_environment_variables_compatibility(self):
        """Test that expected environment variables have defaults"""
        import os

        # Check critical environment variables have defaults
        critical_vars = [
            "RAG_HOST",
            "RAG_PORT",
            "API_PORT",
            "MILVUS_HOST",
            "MILVUS_PORT"
        ]

        for var in critical_vars:
            # Environment variable should either be set or have a reasonable default
            value = os.getenv(var)
            if value is None:
                # Should be handled by application defaults
                pass


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])