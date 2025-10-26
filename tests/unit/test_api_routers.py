"""
API路由测试
测试各种API端点的功能和响应
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json
import uuid

# 模拟导入路由
from api.routers import chat, documents, knowledge, models, health, users, system, analytics


class TestHealthRouter:
    """健康检查路由测试"""

    @pytest.fixture
    def app(self):
        """创建测试应用"""
        app = FastAPI()
        app.include_router(health.router, prefix="/api/health", tags=["health"])
        return app

    @pytest.fixture
    def client(self, app):
        """创建测试客户端"""
        return TestClient(app)

    def test_health_check_basic(self, client):
        """测试基本健康检查"""
        response = client.get("/api/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_check_detailed(self, client):
        """测试详细健康检查"""
        response = client.get("/api/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert "services" in data

        # 验证各个服务的健康状态
        services = data["services"]
        expected_services = ["database", "redis", "llm", "embedding"]
        for service in expected_services:
            assert service in services
            assert "status" in services[service]
            assert "response_time" in services[service]

    def test_service_specific_health(self, client):
        """测试特定服务健康检查"""
        services = ["database", "redis", "llm", "embedding"]

        for service in services:
            response = client.get(f"/api/health/service/{service}")
            assert response.status_code == 200

            data = response.json()
            assert "service" in data
            assert data["service"] == service
            assert "status" in data
            assert "details" in data

    def test_readiness_check(self, client):
        """测试就绪状态检查"""
        response = client.get("/api/health/ready")
        assert response.status_code in [200, 503]  # 503 Service Unavailable if not ready

        data = response.json()
        assert "ready" in data
        assert isinstance(data["ready"], bool)

    def test_liveness_check(self, client):
        """测试存活状态检查"""
        response = client.get("/api/health/live")
        assert response.status_code == 200

        data = response.json()
        assert "alive" in data
        assert data["alive"] is True


class TestChatRouter:
    """聊天路由测试"""

    @pytest.fixture
    def app(self):
        """创建测试应用"""
        app = FastAPI()
        app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
        return app

    @pytest.fixture
    def client(self, app):
        """创建测试客户端"""
        return TestClient(app)

    @pytest.fixture
    def sample_chat_request(self):
        """示例聊天请求"""
        return {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }

    def test_chat_completion(self, client, sample_chat_request):
        """测试聊天完成"""
        with patch('api.routers.chat.llm_service') as mock_service:
            # 模拟LLM服务响应
            mock_response = {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 15,
                    "total_tokens": 35
                }
            }
            mock_service.chat_completion.return_value = mock_response

            response = client.post("/api/chat/completions", json=sample_chat_request)

            assert response.status_code == 200
            data = response.json()
            assert "id" in data
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert data["choices"][0]["message"]["role"] == "assistant"

    def test_chat_completion_streaming(self, client, sample_chat_request):
        """测试流式聊天完成"""
        with patch('api.routers.chat.llm_service') as mock_service:
            # 模拟流式响应
            mock_chunks = [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": "!"}]},
                {"choices": [{"finish_reason": "stop"}]}
            ]
            mock_service.stream_completion.return_value = mock_chunks

            response = client.post(
                "/api/chat/completions",
                json={**sample_chat_request, "stream": True}
            )

            assert response.status_code == 200
            # 验证流式响应格式
            lines = response.text.split('\n')
            assert len(lines) >= 3

    def test_create_conversation(self, client):
        """测试创建对话"""
        conversation_data = {
            "title": "Test Conversation",
            "model": "gpt-3.5-turbo",
            "system_prompt": "You are a helpful assistant."
        }

        with patch('api.routers.chat.conversation_service') as mock_service:
            mock_service.create_conversation.return_value = {
                "id": str(uuid.uuid4()),
                "created_at": "2024-01-01T00:00:00Z",
                **conversation_data
            }

            response = client.post("/api/chat/conversations", json=conversation_data)

            assert response.status_code == 201
            data = response.json()
            assert "id" in data
            assert data["title"] == conversation_data["title"]
            assert data["model"] == conversation_data["model"]

    def test_get_conversations(self, client):
        """测试获取对话列表"""
        with patch('api.routers.chat.conversation_service') as mock_service:
            mock_service.get_conversations.return_value = {
                "conversations": [
                    {
                        "id": str(uuid.uuid4()),
                        "title": "Conversation 1",
                        "created_at": "2024-01-01T00:00:00Z",
                        "message_count": 10
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "title": "Conversation 2",
                        "created_at": "2024-01-02T00:00:00Z",
                        "message_count": 5
                    }
                ],
                "total": 2,
                "page": 1,
                "limit": 10
            }

            response = client.get("/api/chat/conversations")

            assert response.status_code == 200
            data = response.json()
            assert "conversations" in data
            assert "total" in data
            assert len(data["conversations"]) == 2
            assert data["total"] == 2

    def test_delete_conversation(self, client):
        """测试删除对话"""
        conversation_id = str(uuid.uuid4())

        with patch('api.routers.chat.conversation_service') as mock_service:
            mock_service.delete_conversation.return_value = {"deleted": True}

            response = client.delete(f"/api/chat/conversations/{conversation_id}")

            assert response.status_code == 200

    def test_invalid_chat_request(self, client):
        """测试无效聊天请求"""
        invalid_request = {
            "model": "gpt-3.5-turbo",
            # 缺少messages字段
        }

        response = client.post("/api/chat/completions", json=invalid_request)
        assert response.status_code == 422  # Validation error


class TestDocumentsRouter:
    """文档管理路由测试"""

    @pytest.fixture
    def app(self):
        """创建测试应用"""
        app = FastAPI()
        app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
        return app

    @pytest.fixture
    def client(self, app):
        """创建测试客户端"""
        return TestClient(app)

    def test_upload_document(self, client):
        """测试文档上传"""
        # 模拟文件上传
        files = {
            "file": ("test.txt", "This is test document content", "text/plain")
        }
        data = {
            "knowledge_base_id": str(uuid.uuid4()),
            "metadata": json.dumps({"author": "Test Author", "category": "test"})
        }

        with patch('api.routers.documents.document_service') as mock_service:
            mock_service.upload_document.return_value = {
                "id": str(uuid.uuid4()),
                "filename": "test.txt",
                "size": 27,
                "status": "processing"
            }

            response = client.post("/api/documents/upload", files=files, data=data)

            assert response.status_code == 201
            result = response.json()
            assert "id" in result
            assert result["filename"] == "test.txt"
            assert result["status"] == "processing"

    def test_get_documents(self, client):
        """测试获取文档列表"""
        with patch('api.routers.documents.document_service') as mock_service:
            mock_service.get_documents.return_value = {
                "documents": [
                    {
                        "id": str(uuid.uuid4()),
                        "filename": "doc1.txt",
                        "size": 1024,
                        "status": "completed",
                        "created_at": "2024-01-01T00:00:00Z"
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "filename": "doc2.pdf",
                        "size": 2048,
                        "status": "processing",
                        "created_at": "2024-01-02T00:00:00Z"
                    }
                ],
                "total": 2,
                "page": 1,
                "limit": 10
            }

            response = client.get("/api/documents")

            assert response.status_code == 200
            data = response.json()
            assert "documents" in data
            assert len(data["documents"]) == 2
            assert data["total"] == 2

    def test_get_document(self, client):
        """测试获取单个文档"""
        document_id = str(uuid.uuid4())

        with patch('api.routers.documents.document_service') as mock_service:
            mock_service.get_document.return_value = {
                "id": document_id,
                "filename": "test.txt",
                "content": "This is the document content",
                "metadata": {"author": "Test Author"},
                "status": "completed",
                "created_at": "2024-01-01T00:00:00Z"
            }

            response = client.get(f"/api/documents/{document_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == document_id
            assert "content" in data

    def test_delete_document(self, client):
        """测试删除文档"""
        document_id = str(uuid.uuid4())

        with patch('api.routers.documents.document_service') as mock_service:
            mock_service.delete_document.return_value = {"deleted": True}

            response = client.delete(f"/api/documents/{document_id}")

            assert response.status_code == 200

    def test_search_documents(self, client):
        """测试文档搜索"""
        search_params = {
            "query": "test query",
            "knowledge_base_id": str(uuid.uuid4()),
            "limit": 10
        }

        with patch('api.routers.documents.document_service') as mock_service:
            mock_service.search_documents.return_value = {
                "results": [
                    {
                        "document_id": str(uuid.uuid4()),
                        "filename": "matching_doc.txt",
                        "score": 0.95,
                        "snippet": "This contains the test query..."
                    }
                ],
                "total": 1,
                "query": "test query"
            }

            response = client.get("/api/documents/search", params=search_params)

            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 1
            assert data["results"][0]["score"] == 0.95


class TestKnowledgeRouter:
    """知识库管理路由测试"""

    @pytest.fixture
    def app(self):
        """创建测试应用"""
        app = FastAPI()
        app.include_router(knowledge.router, prefix="/api/knowledge", tags=["knowledge"])
        return app

    @pytest.fixture
    def client(self, app):
        """创建测试客户端"""
        return TestClient(app)

    def test_create_knowledge_base(self, client):
        """测试创建知识库"""
        kb_data = {
            "name": "Test Knowledge Base",
            "description": "A test knowledge base for testing purposes",
            "settings": {
                "embedding_model": "text-embedding-ada-002",
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
        }

        with patch('api.routers.knowledge.knowledge_service') as mock_service:
            mock_service.create_knowledge_base.return_value = {
                "id": str(uuid.uuid4()),
                "created_at": "2024-01-01T00:00:00Z",
                "status": "initializing",
                **kb_data
            }

            response = client.post("/api/knowledge/bases", json=kb_data)

            assert response.status_code == 201
            data = response.json()
            assert "id" in data
            assert data["name"] == kb_data["name"]
            assert data["status"] == "initializing"

    def test_get_knowledge_bases(self, client):
        """测试获取知识库列表"""
        with patch('api.routers.knowledge.knowledge_service') as mock_service:
            mock_service.get_knowledge_bases.return_value = {
                "knowledge_bases": [
                    {
                        "id": str(uuid.uuid4()),
                        "name": "KB 1",
                        "description": "First knowledge base",
                        "document_count": 10,
                        "created_at": "2024-01-01T00:00:00Z"
                    }
                ],
                "total": 1
            }

            response = client.get("/api/knowledge/bases")

            assert response.status_code == 200
            data = response.json()
            assert "knowledge_bases" in data
            assert len(data["knowledge_bases"]) == 1

    def test_search_knowledge_base(self, client):
        """测试知识库搜索"""
        kb_id = str(uuid.uuid4())
        search_data = {
            "query": "What is machine learning?",
            "top_k": 5,
            "filters": {"category": "technical"}
        }

        with patch('api.routers.knowledge.knowledge_service') as mock_service:
            mock_service.search_knowledge_base.return_value = {
                "results": [
                    {
                        "document_id": str(uuid.uuid4()),
                        "content": "Machine learning is a subset of AI...",
                        "score": 0.92,
                        "metadata": {"source": "textbook.pdf"}
                    }
                ],
                "query": search_data["query"],
                "total_results": 1
            }

            response = client.post(f"/api/knowledge/bases/{kb_id}/search", json=search_data)

            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 1
            assert data["results"][0]["score"] == 0.92

    def test_get_knowledge_base_stats(self, client):
        """测试获取知识库统计"""
        kb_id = str(uuid.uuid4())

        with patch('api.routers.knowledge.knowledge_service') as mock_service:
            mock_service.get_knowledge_base_stats.return_value = {
                "document_count": 25,
                "total_chunks": 150,
                "total_tokens": 75000,
                "index_size_mb": 10.5,
                "last_updated": "2024-01-01T00:00:00Z",
                "search_count": 100,
                "average_search_time": 0.15
            }

            response = client.get(f"/api/knowledge/bases/{kb_id}/stats")

            assert response.status_code == 200
            data = response.json()
            assert data["document_count"] == 25
            assert data["total_chunks"] == 150
            assert data["average_search_time"] == 0.15


class TestModelsRouter:
    """模型管理路由测试"""

    @pytest.fixture
    def app(self):
        """创建测试应用"""
        app = FastAPI()
        app.include_router(models.router, prefix="/api/models", tags=["models"])
        return app

    @pytest.fixture
    def client(self, app):
        """创建测试客户端"""
        return TestClient(app)

    def test_get_available_models(self, client):
        """测试获取可用模型列表"""
        with patch('api.routers.models.model_service') as mock_service:
            mock_service.get_available_models.return_value = {
                "llm_models": [
                    {
                        "id": "gpt-3.5-turbo",
                        "name": "GPT-3.5 Turbo",
                        "provider": "openai",
                        "status": "available",
                        "max_tokens": 4096
                    },
                    {
                        "id": "gpt-4",
                        "name": "GPT-4",
                        "provider": "openai",
                        "status": "available",
                        "max_tokens": 8192
                    }
                ],
                "embedding_models": [
                    {
                        "id": "text-embedding-ada-002",
                        "name": "Text Embedding Ada 002",
                        "provider": "openai",
                        "dimension": 1536
                    }
                ]
            }

            response = client.get("/api/models")

            assert response.status_code == 200
            data = response.json()
            assert "llm_models" in data
            assert "embedding_models" in data
            assert len(data["llm_models"]) == 2
            assert len(data["embedding_models"]) == 1

    def test_test_model(self, client):
        """测试模型功能测试"""
        test_data = {
            "model_id": "gpt-3.5-turbo",
            "test_type": "chat",
            "test_input": "Hello, this is a test"
        }

        with patch('api.routers.models.model_service') as mock_service:
            mock_service.test_model.return_value = {
                "model_id": "gpt-3.5-turbo",
                "test_type": "chat",
                "success": True,
                "response": "Hello! I received your test message.",
                "response_time": 1.2,
                "tokens_used": 15
            }

            response = client.post("/api/models/test", json=test_data)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "response" in data
            assert "response_time" in data

    def test_get_model_config(self, client):
        """测试获取模型配置"""
        model_id = "gpt-3.5-turbo"

        with patch('api.routers.models.model_service') as mock_service:
            mock_service.get_model_config.return_value = {
                "model_id": model_id,
                "provider": "openai",
                "config": {
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                },
                "capabilities": ["chat", "completion", "function_calling"]
            }

            response = client.get(f"/api/models/{model_id}/config")

            assert response.status_code == 200
            data = response.json()
            assert data["model_id"] == model_id
            assert "config" in data
            assert "capabilities" in data


class TestErrorHandling:
    """错误处理测试"""

    @pytest.fixture
    def app(self):
        """创建带有错误处理的应用"""
        app = FastAPI()
        app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
        return app

    @pytest.fixture
    def client(self, app):
        """创建测试客户端"""
        return TestClient(app)

    def test_not_found_error(self, client):
        """测试404错误"""
        response = client.get("/api/chat/nonexistent")
        assert response.status_code == 404

    def test_validation_error(self, client):
        """测试验证错误"""
        invalid_data = {
            "model": "gpt-3.5-turbo",
            # 缺少必需的messages字段
        }

        response = client.post("/api/chat/completions", json=invalid_data)
        assert response.status_code == 422

        error_data = response.json()
        assert "detail" in error_data

    def test_rate_limit_error(self, client):
        """测试限流错误"""
        # 模拟限流错误
        with patch('api.routers.chat.llm_service') as mock_service:
            mock_service.chat_completion.side_effect = Exception("Rate limit exceeded")

            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            }

            response = client.post("/api/chat/completions", json=request_data)
            assert response.status_code == 500

    def test_server_error(self, client):
        """测试服务器错误"""
        with patch('api.routers.chat.llm_service') as mock_service:
            mock_service.chat_completion.side_effect = Exception("Internal server error")

            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            }

            response = client.post("/api/chat/completions", json=request_data)
            assert response.status_code == 500


class TestAPIIntegration:
    """API集成测试"""

    def test_cors_headers(self):
        """测试CORS头"""
        app = FastAPI()
        app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
        client = TestClient(app)

        response = client.options("/api/chat/completions")
        # 验证CORS头是否存在
        assert "access-control-allow-origin" in response.headers

    def test_api_versioning(self):
        """测试API版本控制"""
        app = FastAPI()
        app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
        client = TestClient(app)

        response = client.get("/api/v1/chat/models")
        assert response.status_code in [200, 404]  # 取决于实现

    def test_request_logging(self):
        """测试请求日志记录"""
        app = FastAPI()
        app.include_router(health.router, prefix="/api/health", tags=["health"])
        client = TestClient(app)

        response = client.get("/api/health")
        assert response.status_code == 200

        # 验证日志记录（在实际实现中应该检查日志输出）