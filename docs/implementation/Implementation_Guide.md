# RAG项目重构实施指南

## 🚀 快速开始

### 环境准备

#### 1. 系统要求
- Python 3.9+
- Git
- Docker (可选，用于容器化部署)
- 足过8GB RAM（用于AI模型服务）

#### 2. 安装uv包管理器
```bash
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install.ps1"
.\install.ps1

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 3. 克隆项目
```bash
git clone <repository-url>
cd complex_rag
```

#### 4. 创建虚拟环境并安装依赖
```bash
uv venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

uv pip install -e .
```

## 📋 实施阶段

### 阶段1：基础架构搭建（2-3周）

#### 1.1 创建项目目录结构
```bash
# 项目基础目录已经存在，验证完整性
ls -la
```

#### 1.2 配置分离式服务配置
```bash
# 检查配置文件
ls config/services/
ls config/ragflow_configs/
```

#### 1.3 验证多租户数据模型
```python
# 检查SQLAlchemy模型是否创建
from infrastructure.database.models import BaseModel, TenantBaseModel

# 验证模型结构
from infrastructure.database.models.user_model import User
from infrastructure.database.models.tenant_model import Tenant
from infrastructure.database.models.knowledge_model import KnowledgeBase
```

#### 1.4 测试文件来源检测器
```python
# 创建测试文件
python -c "
from document_parser.strategies.source_detector import FileSourceDetector

# 测试来源检测
test_cases = [
    {'url': 'https://example.com', 'expected': FileSource.WEB_DOCUMENT},
    {'file_extension': '.pdf', 'expected': FileSource.OFFICE_DOCUMENT},
    {'file_extension': '.json', 'expected': FileSource.STRUCTURED_DATA}
]

for case in test_cases:
    detected = FileSourceDetector.detect_source(case)
    print(f"Input: {case}, Detected: {detected}, Expected: {case['expected']}")
"
```

### 阶段2：核心功能实现（3-4周）

#### 2.1 集成RAGFlow视觉识别模块

**步骤1：复制RAGFlow视觉识别代码**
```bash
# 创建视觉识别目录
mkdir -p document_parser/vision

# 从RAGFlow复制核心文件（需要手动操作）
# 复制以下文件到 document_parser/vision/：
# - ocr.py
# - recognizer.py
# - layout_recognizer.py
# - table_structure_recognizer.py
# - operators.py
# - postprocess.py
```

**步骤2：适配视觉识别模块**
```python
# document_parser/vision/__init__.py
from .ocr import OCR
from .recognizer import Recognizer
from .layout_recognizer import LayoutRecognizer
from .table_structure_recognizer import TableStructureRecognizer

# 适配异步架构
class AsyncOCR:
    async def extract_text(self, image_path: str) -> str:
        # 适配RAGFlow同步代码到异步
        # 实现异步调用逻辑
        pass
```

#### 2.2 实现文件来源专用处理器

**步骤1：创建来源处理器基础类**
```python
# document_parser/source_handlers/__init__.py
from abc import ABC, abstractmethod

class DocumentHandler(ABC):
    @abstractmethod
    async def handle(self, file_info: dict) -> dict:
        pass

    @abstractmethod
    def supports(self, file_source: FileSource) -> bool:
        pass
```

**步骤2：实现web_documents处理器**
```python
# document_parser/source_handlers/web_documents/__init__.py
from ..base import DocumentHandler
from ...parsers.html_parser import HTMLParser
from ...parsers.markdown_parser import MarkdownParser

class WebDocumentHandler(DocumentHandler):
    def supports(self, file_source: FileSource) -> bool:
        return file_source == FileSource.WEB_DOCUMENT

    async def handle(self, file_info: dict) -> dict:
        file_extension = file_info.get('file_extension', '').lower()

        if file_extension == '.html':
            parser = HTMLParser()
        elif file_extension == '.md':
            parser = MarkdownParser()
        else:
            raise ValueError(f"Unsupported web document type: {file_extension}")

        return await parser.parse(file_info['file_path'])
```

**步骤3：实现office_documents处理器**
```python
# document_parser/source_handlers/office_documents/pdf_handler.py
from ...parsers.ragflow_pdf_parser import RAGFlowPdfParser
from ...vision import OCR, LayoutRecognizer

class PDFHandler(DocumentHandler):
    def supports(self, file_source: FileSource) -> bool:
        return file_source == FileSource.OFFICE_DOCUMENT

    async def handle(self, file_info: dict) -> dict:
        # 使用RAGFlow PDF解析器
        parser = RAGFlowPdfParser()

        # 处理PDF文档
        result = parser(file_info['file_path'], binary=file_info.get('binary'))

        return {
            'content': result,
            'metadata': {
                'pages': len(result),
                'file_type': 'pdf',
                'source': 'office_document'
            }
        }
```

#### 2.3 实现处理策略选择器
```python
# document_parser/strategies/strategy_selector.py
from enum import Enum
from typing import Dict, Any

class FileSource(Enum):
    WEB_DOCUMENT = "web_document"
    OFFICE_DOCUMENT = "office_document"
    SCANNED_DOCUMENT = "scanned_document"
    STRUCTURED_DATA = "structured_data"
    CODE_REPOSITORY = "code_repository"

class ProcessingStrategy(BaseModel):
    file_source: FileSource
    parser_type: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "qwen3_embedding"
    rerank_enabled: bool = True

class StrategySelector:
    def __init__(self):
        self.strategies = {
            FileSource.WEB_DOCUMENT: ProcessingStrategy(
                file_source=FileSource.WEB_DOCUMENT,
                parser_type="html_parser",
                chunk_size=800,
                chunk_overlap=150,
                embedding_model="qwen3_embedding",
                rerank_enabled=True
            ),
            FileSource.OFFICE_DOCUMENT: ProcessingStrategy(
                file_source=FileSource.OFFICE_DOCUMENT,
                parser_type="pdf_parser",
                chunk_size=1000,
                chunk_overlap=200,
                embedding_model="qwen3_embedding",
                rerank_enabled=True
            ),
            # ... 其他策略配置
        }

    def get_strategy(self, file_source: FileSource) -> ProcessingStrategy:
        return self.strategies.get(file_source)
```

#### 2.4 实现数据库连接和基础模型

**步骤1：配置MySQL连接**
```python
# config/services/db_config.py
from pydantic import BaseSettings

class DatabaseConfig(BaseSettings):
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DB: str = "rag_system"
    MYSQL_USER: str = "rag_user"
    MYSQL_PASSWORD: str = "rag_password"
    MYSQL_CHARSET: str = "utf8mb4"

    class Config:
        env_file = ".env"

# infrastructure/database/implementations/mysql_client.py
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

class MySQLClient:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = create_async_engine(
            f"mysql+aiomysql://{config.MYSQL_USER}:{config.MYSQL_PASSWORD}@"
            f"{config.MYSQL_HOST}:{config.MYSQL_PORT}/{config.MYSQL_DB}",
            echo=False,
            charset=config.MYSQL_CHARSET
        )

    async def get_session(self) -> AsyncSession:
        async with self.engine.begin() as conn:
            return AsyncSession(conn)
```

**步骤2：实现基础数据模型**
```python
# infrastructure/database/models/base_model.py
from sqlalchemy import Column, String, BigInteger, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import datetime

Base = declarative_base()

class BaseModel(Base):
    __abstract__ = True

    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    create_time = Column(BigInteger, nullable=False, index=True, default=int(datetime.now().timestamp()))
    create_date = Column(DateTime, nullable=False, index=True, default=datetime.now)
    update_time = Column(BigInteger, nullable=True, index=True)
    update_date = Column(DateTime, nullable=True, index=True)
    status = Column(String(1), default="1")  # "1"=正常, "0"=删除

    def update_timestamp(self):
        self.update_time = int(datetime.now().timestamp())
        self.update_date = datetime.now()

# infrastructure/database/models/tenant_base_model.py
from .base_model import BaseModel

class TenantBaseModel(BaseModel):
    __abstract__ = True

    tenant_id = Column(String(32), nullable=False, index=True, comment="租户ID")
```

### 阶段3：高级功能实现（2-3周）

#### 3.1 集成RAGFlow GraphRAG模块

**步骤1：复制RAGFlow GraphRAG代码**
```bash
# 创建GraphRAG目录
mkdir -p core_rag/graph_rag/general core_rag/graph_rag/light

# 从RAGFlow复制核心文件（需要手动操作）
# core_rag/graph_rag/general/
# core_rag/graph_rag/light/
```

**步骤2：实现GraphRAG服务**
```python
# core_rag/graph_rag/services/graph_service.py
from .general.extraction import EntityExtraction
from .light.extraction import LightExtraction
from .search.kg_search import KGSearch

class GraphRAGService:
    def __init__(self):
        self.general_extractor = EntityExtraction()
        self.light_extractor = LightExtraction()
        self.kg_search = KGSearch()

    async def process_document(self, text: str, mode: str = "general") -> dict:
        if mode == "general":
            # 使用General模式（多轮抽取）
            return await self.general_extractor.extract(text)
        elif mode == "light":
            # 使用Light模式（单次抽取）
            return await self.light_extractor.extract(text)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
```

#### 3.2 实现Mem0上下文处理

**步骤1：安装和配置Mem0**
```bash
# 安装Mem0
uv add mem0ai
```

**步骤2：实现Mem0服务**
```python
# core_rag/memory/implementations/mem0_memory.py
import mem0
from ..interfaces.memory_interface import MemoryInterface

class Mem0Memory(MemoryInterface):
    def __init__(self, api_key: str):
        self.client = mem0.Memory()
        self.api_key = api_key

    async def store_memory(self, messages: list, user_id: str, metadata: dict = None) -> str:
        # 存储对话记忆
        result = self.client.add(
            messages=messages,
            user_id=user_id,
            metadata=metadata or {}
        )
        return result.get("id")

    async def search_memory(self, query: str, user_id: str, limit: int = 5) -> list:
        # 搜索相关记忆
        results = self.client.search(
            query=query,
            user_id=user_id,
            limit=limit
        )
        return results.get("results", [])
```

#### 3.3 实现LangGraph召回组件

**步骤1：安装LangGraph**
```bash
# 安装LangGraph
uv add langgraph
```

**步骤2：实现召回节点**
```python
# core_rag/langgraph/nodes/vector_node.py
from langgraph.graph import State
from ..interfaces.vector_retriever import VectorRetrieverInterface

class VectorNode:
    def __init__(self, retriever: VectorRetrieverInterface):
        self.retriever = retriever

    async def __call__(self, state: State) -> State:
        # 执行向量检索
        query = state["query"]
        results = await self.retriever.retrieve(query)
        state["vector_results"] = results
        return state

# core_rag/langgraph/nodes/keyword_node.py
# 类似实现关键词检索节点

# core_rag/langgraph/nodes/entity_node.py
# 类似实现实体检索节点
```

**步骤3：实现召回工作流**
```python
# core_rag/langgraph/workflows/retrieval_workflow.py
from langgraph.graph import StateGraph, START, END
from ..nodes.vector_node import VectorNode
from ..nodes.keyword_node import KeywordNode
from ..nodes.entity_node import EntityNode

class RetrievalWorkflow:
    def __init__(self):
        self.workflow = StateGraph(State)

        # 添加节点
        self.workflow.add_node("vector_retrieval", VectorNode(retriever))
        self.workflow.add_node("keyword_retrieval", KeywordNode(retriever))
        self.workflow.add_node("entity_retrieval", EntityNode(retriever))

        # 设置边
        self.workflow.add_edge(START, "vector_retrieval")
        self.workflow.add_edge("vector_retrieval", "keyword_retrieval")
        self.workflow.add_edge("keyword_retrieval", "entity_retrieval")
        self.workflow.add_edge("entity_retrieval", END)

    async def run(self, initial_state: dict) -> dict:
        runnable = self.workflow.compile()
        state = initial_state
        result = await runnable.ainvoke(state)
        return result
```

### 阶段4：测试和优化（1-2周）

#### 4.1 编写单元测试

**示例：测试文件来源检测器**
```python
# tests/unit/test_source_detector.py
import pytest
from document_parser.strategies.source_detector import FileSourceDetector

class TestFileSourceDetector:
    def test_detect_web_document(self):
        file_info = {"url": "https://example.com"}
        result = FileSourceDetector.detect_source(file_info)
        assert result == FileSource.WEB_DOCUMENT

    def test_detect_office_document(self):
        file_info = {"file_extension": ".pdf"}
        result = FileSourceDetector.detect_source(file_info)
        assert result == FileSource.OFFICE_DOCUMENT
```

**示例：测试文档处理器**
```python
# tests/unit/test_document_handlers.py
import pytest
from document_parser.source_handlers.office_documents.pdf_handler import PDFHandler
from document_parser.strategies.source_detector import FileSource

class TestPDFHandler:
    @pytest.fixture
    def pdf_handler(self):
        return PDFHandler()

    def test_supports_pdf(self, pdf_handler):
        assert pdf_handler.supports(FileSource.OFFICE_DOCUMENT)
        assert not pdf_handler.supports(FileSource.WEB_DOCUMENT)

    @pytest.mark.asyncio
    async def test_handle_pdf_file(self, pdf_handler):
        # 创建测试PDF文件
        # 测试PDF处理逻辑
        pass
```

#### 4.2 实现开发调试工具

**一键清理数据脚本**
```python
# scripts/clear_all_data.py
import asyncio
from infrastructure.database.mysql_client import MySQLClient
from infrastructure.database.milvus_client import MilvusClient
from infrastructure.database.es_client import ElasticsearchClient
from infrastructure.cache.redis_client import RedisClient

async def clear_all_data():
    print("开始清理所有数据...")

    # 清理MySQL数据
    mysql_client = MySQLClient(config.db_config)
    await mysql_client.clear_all_tables()
    print("✅ MySQL数据清理完成")

    # 清理Milvus数据
    milvus_client = MilvusClient(config.milvus_config)
    await milvus_client.clear_all_collections()
    print("✅ Milvus数据清理完成")

    # 清理Elasticsearch数据
    es_client = ElasticsearchClient(config.es_config)
    await es_client.clear_all_indices()
    print("✅ Elasticsearch数据清理完成")

    # 清理Redis缓存
    redis_client = RedisClient(config.redis_config)
    await redis_client.clear_all_data()
    print("✅ Redis缓存清理完成")

    print("🎉 所有数据清理完成！")

if __name__ == "__main__":
    asyncio.run(clear_all_data())
```

**服务启动脚本**
```python
# scripts/start_services.py
import asyncio
import uvicorn
from api.main import app as api_app
from rag_service.app import app as rag_app

async def start_api_service():
    config = uvicorn.Config(
        api_app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    await uvicorn.serve(config)

async def start_rag_service():
    config = uvicorn.Config(
        rag_app,
        host="0.0.0.0",
        port=8001,
        workers=1,  # 单进程模式
        log_level="info"
    )
    await uvicorn.serve(config)

async def main():
    print("🚀 启动服务...")

    # 并发启动两个服务
    await asyncio.gather(
        start_api_service(),
        start_rag_service()
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔧 配置管理

### 环境变量配置
创建 `.env` 文件：
```bash
# 数据库配置
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=rag_system
MYSQL_USER=rag_user
MYSQL_PASSWORD=your_password
MYSQL_CHARSET=utf8mb4

# 向量数据库配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=document_embeddings

# Elasticsearch配置
ES_HOST=localhost
ES_PORT=9200
ES_INDEX_NAME=document_search

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# AI模型配置
QWEN3_API_KEY=your_qwen3_api_key
OPENAI_API_KEY=your_openai_api_key

# Mem0配置
MEM0_API_KEY=your_mem0_api_key

# MinIO配置
MINIO_HOST=localhost
MINIO_PORT=9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
MINIO_BUCKET_NAME=rag-documents
```

### 服务配置示例

**API服务配置 (config/services/api_config.py)**
```python
from pydantic import BaseSettings

class APIConfig(BaseSettings):
    # 服务器配置
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    DEBUG: bool = False

    # CORS配置
    CORS_ORIGINS: list = ["*"]
    ALLOW_CREDENTIALS: bool = True

    # API配置
    API_V1_PREFIX: str = "/api/v1"

    # 响应配置
    DEFAULT_RESPONSE_CLASS: str = "json"
    MAX_REQUEST_SIZE: int = 16 * 1024 * 1024  # 16MB
```

**RAG服务配置 (config/services/rag_service_config.py)**
```python
from pydantic import BaseSettings

class RAGServiceConfig(BaseSettings):
    # 服务器配置
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8001
    WORKERS: int = 1  # 单进程模式

    # AI模型配置
    DEFAULT_LLM_MODEL: str = "qwen3:latest"
    DEFAULT_EMBEDDING_MODEL: str = "qwen3-embedding:latest"
    DEFAULT_RERANK_MODEL: str = "qwen3-reranker:latest"

    # OpenAI兼容配置
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    OPENAI_API_KEY: str = "your_openai_key"

    # 处理配置
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30
```

## 🐛 故障排除

### 常见问题

#### 1. 数据库连接失败
```bash
# 检查数据库服务状态
mysql -h localhost -u rag_user -p

# 检查防火墙设置
telnet localhost 3306

# 查看MySQL错误日志
tail -f /var/log/mysql/error.log
```

#### 2. 向量数据库连接失败
```bash
# 检查Milvus服务状态
python -c "
from pymilvus import connections
connections.connect('default', host='localhost', port='19530')
print('Milvus连接成功')
"

# 检查Milvus集合状态
# 使用Milvus客户端工具
```

#### 3. 内存不足
```bash
# 检查系统内存使用
free -h
htop

# 调整Python内存限制（临时方案）
export PYTHONMALLOC=max(100000000, $PYTHONMALLOC)
```

#### 4. 模型加载失败
```python
# 检查模型文件路径
import os
from pathlib import Path

model_path = Path("assets/models/llm_models")
if model_path.exists():
    print(f"模型目录存在: {model_path}")
    print(f"模型文件: {list(model_path.glob('*'))}")
else:
    print("模型目录不存在，请下载模型")
```

### 调试技巧

#### 1. 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或在配置文件中设置
LOG_LEVEL=DEBUG
```

#### 2. 使用调试工具
```python
# 添加断点
import pdb; pdb.set_trace()

# 使用IPython调试
import IPython; IPython.embed()
```

#### 3. 性能分析
```python
import cProfile
import pstats

# 分析函数性能
cProfile.run('your_function()')
```

## 📊 性能优化建议

### 1. 数据库优化
- 在tenant_id上建立复合索引
- 使用连接池管理连接
- 实现查询结果缓存

### 2. 向量检索优化
- 预计算并缓存常用查询的向量
- 批量处理向量计算
- 调整向量维度和索引参数

### 3. 缓存策略优化
- 实现多级缓存（内存+Redis）
- 设置合理的缓存过期时间
- 实现缓存预热机制

## 📚 部署指南

### Docker部署（推荐）

#### 1. 构建Docker镜像
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装uv
COPY pyproject.toml .
RUN uv venv
RUN uv pip install -e .

# 复制源代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app/.venv/bin
ENV PATH=/app/.venv/bin:$PATH

# 暴露端口
EXPOSE 8000 8001

# 启动命令
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Docker Compose配置
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MYSQL_HOST=mysql
      - REDIS_HOST=redis
    depends_on:
      - mysql
      - redis

  rag-service:
    build: .
    ports:
      - "8001:8001"
    environment:
      - MYSQL_HOST=mysql
      - REDIS_HOST=redis
      - MILVUS_HOST=milvus
    depends_on:
      - mysql
      - redis
      - milvus

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: rag_system
      MYSQL_USER: rag_user
      MYSQL_PASSWORD: ragpassword
    volumes:
      - mysql_data:/var/lib/mysql

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  milvus:
    image: milvusdb/milvus:latest
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"

volumes:
  mysql_data:
  redis_data:
  milvus_data:
```

### 生产环境部署

#### 1. Gunicorn部署
```bash
# 安装Gunicorn
pip install gunicorn

# 启动API服务
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
  --log-level info
```

#### 2. Nginx反向代理
```nginx
server {
    listen 80;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /rag/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## 📚 监控和日志

### 日志配置
```python
# config/loguru_config.py
import sys
from loguru import logger

# 移除默认处理器
logger.remove()

# 添加文件和控制台处理器
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
    rotation="1 day",
    retention="30 days"
)

# 按模块分类存储日志
logger.add(
    f"logs/api/{__name__}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
    rotation="1 day",
    retention="30 days",
    filter=lambda record: "api" in record["name"]
)
```

### 性能监控
```python
# infrastructure/monitoring/metrics.py
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
        end_time = time.time()

        duration = end_time - start_time
        logger.info(f"Function {func.__name__} executed in {duration:.2f}s, success: {success}")

        return result
    return wrapper
```

---

这个实施指南提供了详细的步骤和代码示例，帮助你从零开始构建这个RAG系统。建议按照阶段顺序逐步实施，每个阶段完成后进行充分测试，确保功能稳定后再进入下一阶段。

如果遇到任何问题，请参考故障排除部分或查阅相关技术文档。