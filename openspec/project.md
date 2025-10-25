# Project Context

## Purpose
这是一个基于 RAG（Retrieval-Augmented Generation）架构的智能问答系统，主要目的是实现企业级知识库检索和对话功能。系统支持用户上传文档、进行向量化处理，并提供基于检索结果的智能问答服务，特别针对中文知识库场景进行了优化。

## Tech Stack

### 核心技术框架
- **Python 3.9+** - 主要开发语言
- **FastAPI** - 现代化异步 Web 框架，提供 RESTful API
- **Sanic** - 高性能微服务框架，用于向量化和重排服务
- **Celery** - 分布式异步任务队列，处理文档解析等耗时任务
- **Uvicorn** - ASGI 服务器

### AI/ML 技术栈
- **LangChain** - 大语言模型应用框架
- **Transformers** - Hugging Face 模型库
- **PyTorch** - 深度学习框架
- **Qwen3** - 阿里通义千问模型，用于向量化和重排
- **Magic-PDF** - PDF 解析工具
- **Ollama** - 本地模型部署支持

### 数据存储
- **Milvus** - 向量数据库，用于向量检索
- **MySQL** - 关系型数据库，存储元数据
- **Redis** - 缓存和消息队列
- **Elasticsearch** - 全文检索引擎
- **MinIO** - 对象存储，用于文件存储

### 开发工具
- **Black** - 代码格式化
- **Flake8** - 代码风格检查
- **iSort** - 导入排序
- **MyPy** - 静态类型检查
- **Pytest** - 单元测试框架

## Project Conventions

### Code Style
- 使用 **Black** 进行代码格式化，行长度限制 88 字符
- 使用 **iSort** 进行导入语句排序
- 遵循 **PEP 8** Python 编码规范
- 使用 **Type Hints** 提高代码可读性和类型安全
- 函数和变量使用 **snake_case** 命名
- 类名使用 **PascalCase** 命名
- 常量使用 **UPPER_CASE** 命名

### Architecture Patterns

#### 微服务架构
- **FastAPI 主服务**：处理用户请求、文件上传、对话管理
- **Sanic 向量服务**：独立的向量化和重排服务
- **Celery 异步任务**：文档处理、向量计算等后台任务
- **消息队列**：Redis 作为任务队列和缓存

#### RAG 系统架构
- **文档处理流水线**：上传 → 解析 → 分块 → 向量化 → 存储
- **混合检索**：Milvus（向量检索）+ Elasticsearch（全文检索）
- **重排序机制**：使用 Qwen3-Reranker 优化检索结果
- **上下文构建**：基于检索结果构建 LLM 上下文

### Testing Strategy
- 使用 **Pytest** 进行单元测试和集成测试
- 测试覆盖率目标：**80%+**
- API 测试使用 **FastAPI TestClient**
- 数据库测试使用 **pytest fixtures** 进行隔离
- 异步任务测试使用 **pytest-asyncio**

### Git Workflow
- **主分支**：`master` - 生产环境代码
- **开发分支**：`develop` - 开发环境集成
- **功能分支**：`feature/功能名` - 新功能开发
- **提交格式**：`类型: 简短描述`（如：`feat: 添加文档上传功能`）
- **提交类型**：`feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Domain Context

### RAG 系统核心概念
- **向量检索 (Vector Search)**：将文档转换为向量，在向量空间中查找相似内容
- **重排序 (Reranking)**：对初步检索结果进行二次排序，提高相关性
- **混合检索 (Hybrid Search)**：结合向量检索和全文检索的优势
- **知识库管理**：文档的增删改查、版本控制、权限管理

### 中文优化特性
- **中文分词**：Elasticsearch 使用 ik 分词器
- **拼音转换**：支持拼音搜索功能
- **中文向量化**：Qwen3 模型针对中文场景优化
- **文档解析**：特别优化了 PDF 等中文文档的处理

## Important Constraints

### 性能约束
- **API 响应时间**：普通查询 < 2 秒，复杂查询 < 5 秒
- **文档处理**：单个文档处理时间 < 30 秒
- **并发支持**：至少支持 100 个并发用户
- **向量检索**：百万级向量库检索时间 < 100ms

### 技术约束
- **Python 版本**：3.9 及以上
- **内存要求**：向量服务至少需要 8GB RAM
- **存储要求**：支持本地文件系统和分布式存储
- **模型支持**：优先支持本地部署的开源模型

### 安全约束
- **文件安全**：上传文件类型限制，防止恶意文件
- **数据隐私**：敏感数据加密存储
- **访问控制**：基于用户的知识库访问权限
- **API 安全**：JWT token 认证，请求频率限制

## External Dependencies

### AI 模型服务
- **Qwen3-Embedding**：文本向量化模型
- **Qwen3-Reranker**：检索结果重排序模型
- **Ollama API**：本地大语言模型服务

### 基础设施服务
- **Milvus 集群**：向量数据库服务
- **MySQL 数据库**：元数据存储
- **Redis 集群**：缓存和消息队列
- **Elasticsearch 集群**：全文检索服务
- **MinIO 服务**：对象存储服务

### 外部 API（可选）
- **百度翻译 API**：多语言支持
- **百度 OCR API**：图像文字识别
- **天气 API**：天气查询功能
