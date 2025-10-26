"""
测试配置文件
定义pytest的配置和共享fixtures
"""
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock
import tempfile
import os
from pathlib import Path

# 设置测试环境
os.environ["ENVIRONMENT"] = "test"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """创建事件循环用于异步测试"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """创建临时目录fixture"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_settings():
    """模拟应用配置"""
    from unittest.mock import Mock
    settings = Mock()
    settings.environment = "test"
    settings.debug = True
    settings.api_host = "127.0.0.1"
    settings.api_port = 8000
    settings.database_url = "sqlite:///:memory:"
    settings.redis_url = "redis://localhost:6379/1"
    return settings


@pytest.fixture
async def mock_async_client():
    """模拟异步HTTP客户端"""
    from unittest.mock import AsyncMock
    client = AsyncMock()
    return client


@pytest.fixture
def sample_text_data():
    """示例文本数据fixture"""
    return {
        "short_text": "这是一个测试文本。",
        "medium_text": "这是一个中等长度的测试文本，包含多个句子。它用于测试文本处理功能。",
        "long_text": """这是一个很长的测试文本，包含多个段落。

第一段包含了一些基本信息。

第二段包含了更详细的内容，用于测试长文本的处理能力。

第三段可能包含一些特殊字符和数字：12345, !@#$%, 中英文混合。

最后一段总结所有内容。""",
        "multilingual_text": "This is English text. 这是中文文本。これは日本語のテキストです。",
        "text_with_numbers": "产品价格：￥199.99，折扣：20%，库存：100件",
        "text_with_dates": "会议时间：2024-12-25 14:30，截止日期：2025/01/15"
    }


@pytest.fixture
def sample_json_data():
    """示例JSON数据fixture"""
    return {
        "user": {
            "id": 1,
            "name": "测试用户",
            "email": "test@example.com",
            "preferences": {
                "language": "zh-CN",
                "theme": "dark"
            }
        },
        "documents": [
            {
                "id": "doc1",
                "title": "测试文档1",
                "content": "这是第一个测试文档的内容",
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "tags": ["test", "document"]
                }
            },
            {
                "id": "doc2",
                "title": "测试文档2",
                "content": "这是第二个测试文档的内容",
                "metadata": {
                    "created_at": "2024-01-02T00:00:00Z",
                    "tags": ["test", "sample"]
                }
            }
        ]
    }


@pytest.fixture
def mock_llm_response():
    """模拟LLM响应fixture"""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "这是一个模拟的AI助手回复。"
                },
                "finish_reason": "stop",
                "index": 0
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        },
        "model": "gpt-3.5-turbo",
        "id": "chatcmpl-test123",
        "created": 1700000000
    }


@pytest.fixture
def mock_embedding_response():
    """模拟嵌入向量响应fixture"""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, -0.2, 0.3, -0.4, 0.5] * 204,  # 1024维向量
                "index": 0
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }


@pytest.fixture
def sample_document_metadata():
    """示例文档元数据fixture"""
    return {
        "filename": "test_document.pdf",
        "file_type": "application/pdf",
        "file_size": 1024000,
        "created_at": "2024-01-01T00:00:00Z",
        "modified_at": "2024-01-02T00:00:00Z",
        "author": "测试作者",
        "title": "测试文档标题",
        "subject": "测试主题",
        "keywords": ["测试", "文档", "示例"],
        "language": "zh-CN",
        "page_count": 10,
        "word_count": 5000,
        "checksum": "md5:abc123def456"
    }


# 异步测试工具函数
async def run_async_test(coro):
    """运行异步测试的辅助函数"""
    return await coro


def create_mock_async_response(data, status_code=200):
    """创建模拟异步HTTP响应"""
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.json = AsyncMock(return_value=data)
    mock_response.text = AsyncMock(return_value=str(data))
    mock_response.headers = {"content-type": "application/json"}
    return mock_response


def assert_dict_subset(subset, superset):
    """断言字典包含子集"""
    for key, value in subset.items():
        assert key in superset
        assert superset[key] == value


def assert_async_called_once(mock_async):
    """断言异步函数被调用一次"""
    assert mock_async.called
    assert mock_async.call_count == 1


# 测试标记
pytest_plugins = []

# 自定义标记
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line("markers", "unit: 单元测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "slow: 慢速测试")
    config.addinivalue_line("markers", "async_test: 异步测试")
    config.addinivalue_line("markers", "requires_redis: 需要Redis的测试")
    config.addinivalue_line("markers", "requires_db: 需要数据库的测试")


# 测试收集配置
def pytest_collection_modifyitems(config, items):
    """修改测试收集配置"""
    for item in items:
        # 为异步测试添加 asyncio 标记
        if "async_test" in item.keywords:
            item.add_marker(pytest.mark.asyncio)

        # 为慢速测试添加 timeout
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.timeout(300))