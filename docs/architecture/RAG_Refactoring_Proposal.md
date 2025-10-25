# RAG项目重构提案

## 项目概述

### 背景
这是一个基于RAG（Retrieval-Augmented Generation）架构的智能问答系统完全重构项目。目标是从零开始构建一个现代化、高性能、可扩展的企业级RAG系统。

### 核心理念
- **渐进式开发**：先实现基础功能，确保核心流程跑通，再逐步添加高级功能
- **抽象优先设计**：所有功能模块都定义清晰的抽象接口，支持多种实现
- **RAGFlow最佳实践**：直接复用RAGFlow的成熟组件，加速开发进程
- **业务导向处理**：基于文件来源的精细化文档处理，更贴近实际需求

## 技术架构设计

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG系统架构                                    │
├─────────────────────────────────────────────────────────────────┤
│  API层 (FastAPI)                                                    │
│  ├── 对话接口        ├── 文档管理接口        ├── 知识库管理接口      │
│  ├── 模型管理接口      ├── 健康检查接口        └── 异常处理           │
├─────────────────────────────────────────────────────────────────┤
│  RAG服务层 (Sanic)                                                  │
│  ├── OpenAI兼容LLM服务 ├── 向量化服务          ├── 重排服务           │
│  ├── 对话生成服务      ├── 检索服务            └── 对话记忆服务       │
├─────────────────────────────────────────────────────────────────┤
│  文档解析层 (基于RAGFlow deepdoc + rag/app架构)                      │
│  ├── 文件来源检测器     ├── 来源专用处理器                         │
│  │   ├── web_documents/    ├── office_documents/                 │
│  │   ├── scanned_documents/ └── structured_data/                  │
│  ├── RAGFlow视觉识别    ├── RAGFlow解析器                         │
│  └── 处理流水线         └── 可扩展插件架构                         │
├─────────────────────────────────────────────────────────────────┤
│  核心RAG层                                                          │
│  ├── Mem0上下文处理    ├── LangGraph召回组件                      │
│  ├── 多策略检索器       ├── GraphRAG (General+Light模式)          │
│  ├── 检索结果融合        └── RAPTOR层次化检索                      │
├─────────────────────────────────────────────────────────────────┤
│  基础设施层                                                          │
│  ├── 多租户数据库(MySQL) ├── 向量数据库(Milvus)                   │
│  ├── 搜索引擎(Elasticsearch) ├── 对象存储(MinIO)                 │
│  ├── 缓存服务(Redis)    └── 异步任务队列(Trio)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 详细目录结构

```
complex_rag/
├── api/                          # API层 - FastAPI应用
├── rag_service/                   # RAG服务层 - Sanic高性能AI服务
├── document_parser/               # 文档解析层（基于RAGFlow）
│   ├── source_handlers/            # 文件来源处理器
│   │   ├── web_documents/         # 网页文档处理器
│   │   ├── office_documents/       # Office文档处理器
│   │   ├── scanned_documents/      # 扫描文档处理器
│   │   ├── structured_data/        # 结构化数据处理器
│   │   ├── code_repositories/      # 代码仓库处理器
│   │   └── custom_sources/         # 自定义来源处理器
│   ├── vision/                     # 视觉识别模块（RAGFlow）
│   ├── parsers/                    # 具体解析器实现
│   ├── strategies/                 # 处理策略配置
│   └── services/                   # 解析服务
├── core_rag/                      # 核心RAG层 - 业务逻辑
│   ├── context/                    # Mem0上下文处理和问题重写
│   ├── langgraph/                  # LangGraph召回组件架构
│   ├── graph_rag/                  # GraphRAG实现（RAGFlow）
│   ├── retrieval/                  # 检索引擎
│   ├── memory/                     # 对话记忆管理
│   └── pipeline/                   # RAG处理流水线
├── infrastructure/                # 基础设施层
│   ├── database/                   # 数据库抽象和实现
│   ├── storage/                    # 对象存储
│   ├── cache/                      # 缓存服务
│   ├── queue/                      # 消息队列
│   └── monitoring/                 # 监控和日志
├── assets/                        # 模型文件和资源存储
├── config/                        # 配置管理（分离式配置）
├── quality/                       # 代码质量检测配置
├── tests/                         # 单元测试
├── scripts/                       # 脚本工具
├── logs/                          # 日志文件存储
└── docs/                          # 文档
```

## 核心技术栈

### 主要框架
- **Python 3.9+** - 主要开发语言
- **FastAPI** - API层Web框架
- **Sanic** - RAG服务层高性能框架
- **Trio** - 异步任务处理框架

### AI/ML技术栈
- **LangChain** - 大语言模型应用框架
- **OpenAI API规范** - 统一的AI服务接口
- **Mem0** - 对话记忆管理
- **RAGFlow组件** - 文档解析和GraphRAG

### 数据存储
- **MySQL** - 业务数据库（多租户）
- **Milvus** - 向量数据库
- **Elasticsearch** - 搜索引擎
- **Redis** - 缓存和消息队列
- **MinIO** - 对象存储

### 开发工具
- **uv** - 包管理器
- **Black** - 代码格式化
- **iSort** - 导入排序
- **MyPy** - 静态类型检查
- **Loguru** - 结构化日志

## 多租户数据库架构

### 核心设计理念
基于RAGFlow的db_models.py设计符合三范式的多租户数据库架构：

```python
# 基础模型
class BaseModel(Base):
    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    create_time = Column(BigInteger, nullable=True, index=True)
    create_date = Column(DateTime, nullable=True, index=True)
    update_time = Column(BigInteger, nullable=True, index=True)
    update_date = Column(DateTime, nullable=True, index=True)
    status = Column(String(1), default="1")

# 多租户基础模型
class TenantBaseModel(BaseModel):
    tenant_id = Column(String(32), nullable=False, index=True)
```

### 核心实体模型
- **User** - 用户管理
- **Tenant** - 租户管理
- **UserTenant** - 用户租户关联
- **KnowledgeBase** - 知识库管理
- **Document** - 文档管理
- **Chat** - 对话管理
- **Graph** - 图数据模型

## 文档解析架构

### 基于RAGFlow deepdoc + rag/app的设计

#### 文件来源分类处理器
1. **web_documents/** - 网页文档处理器
   - HTML文档处理
   - Markdown文档处理
   - API文档处理

2. **office_documents/** - Office文档处理器
   - PDF文档处理（基于RAGFlow pdf_parser.py）
   - Word文档处理
   - Excel表格处理
   - PPT演示处理

3. **scanned_documents/** - 扫描文档处理器
   - OCR扫描文档处理
   - 图片文档处理
   - 多模态文档处理

4. **structured_data/** - 结构化数据处理器
   - JSON数据处理
   - CSV数据处理
   - XML数据处理
   - YAML数据处理

5. **code_repositories/** - 代码仓库处理器
   - GitHub仓库处理
   - 代码文件处理
   - 技术文档处理

#### 智能处理策略选择
```python
# 文件来源自动检测
class FileSourceDetector:
    @staticmethod
    def detect_source(file_info: dict) -> FileSource:
        if file_info.get("url"):
            return FileSource.WEB_DOCUMENT
        elif file_info.get("file_extension") in [".pdf", ".docx"]:
            return FileSource.OFFICE_DOCUMENT

# 处理策略选择器
class StrategySelector:
    def get_strategy(self, file_source: FileSource) -> ProcessingStrategy:
        return self.strategies.get(file_source)
```

## RAG服务架构

### Sanic高性能AI服务
- **单进程模式**：配置为单进程模式以获得最佳性能
- **OpenAI兼容接口**：完全兼容OpenAI API规范
- **异步处理**：支持流式和非流式响应模式

#### 核心接口
- `/v1/chat/completions` - LLM服务接口
- `/v1/embeddings` - 向量化服务接口
- `/v1/rerank` - 重排服务接口

## GraphRAG架构

### 基于RAGFlow graphrag的实现

#### 双模式支持
1. **General模式**（微软GraphRAG）
   - 多轮实体抽取
   - 实体解析
   - 社区发现

2. **Light模式**（LightRAG）
   - 单次实体抽取
   - 关键词总结

#### 核心组件
```python
# 图数据模型
class GraphModel:
    nodes = []  # 实体节点
    edges = []  # 关系边

# 图检索服务
class KGSearch:
    def retrieval(self, question, tenant_ids, kb_ids, emb_mdl, llm):
        # 多路检索：实体+关系+社区
```

## 核心RAG功能

### Mem0上下文处理
- **上下文检索**：基于对话历史的上下文信息提取
- **查询重写**：基于历史上下文重写查询语句
- **问题拆分**：复杂问题的多层次分解

### LangGraph召回组件
- **节点化架构**：每种召回策略定义为独立节点
- **工作流编排**：支持并行执行和条件跳转
- **数据流转**：节点间的标准化数据接口

### 多策略检索系统
- **向量检索**：基于语义相似度的检索
- **关键词检索**：基于全文检索的精确匹配
- **稀疏检索**：基于BM25等算法的稀疏检索
- **实体检索**：基于知识图谱的实体检索
- **RAPTOR检索**：层次化检索实现

## 配置管理架构

### 分离式服务配置
```
config/
├── services/              # 服务配置分离
│   ├── api_config.py       # API服务配置
│   ├── rag_service_config.py # RAG服务配置
│   ├── db_config.py        # 数据库配置
│   ├── cache_config.py     # 缓存配置
│   ├── storage_config.py   # 存储配置
│   └── ai_models_config.py # AI模型配置
├── ragflow_configs/        # RAGFlow配置参考
│   ├── deepdoc_config.py   # DeepDoc解析配置
│   └── graphrag_config.py  # GraphRAG配置
└── secrets/                # 敏感配置
```

## 开发工具和调试支持

### 一键数据清理脚本
```python
# clear_all_data.py
def clear_all_data():
    # 清理数据库中的所有表数据
    # 清理向量数据库中的所有向量数据
    # 清理缓存和临时文件
    # 保持数据库结构和索引
```

### 服务启动管理
```python
# start_services.py
def start_services():
    # 启动API服务和RAG服务
    # 启动文档解析服务
    # 服务健康检查
    # 优雅关闭支持
```

## 渐进式开发策略

### 第一阶段（2-3周）：基础架构
- 搭建项目目录结构
- 配置分离式服务配置
- 实现多租户SQLAlchemy数据模型
- 集成RAGFlow视觉识别模块
- 实现文件来源检测和处理策略

### 第二阶段（3-4周）：核心功能
- 实现五大来源专用处理器
- 集成RAGFlow GraphRAG模块
- 实现MySQL+Milvus+Elasticsearch基础功能
- 实现基本RAG流程

### 第三阶段（2-3周）：高级功能
- 实现Mem0上下文处理
- 完善LangGraph召回组件
- 集成第三方最佳实践
- 实现多策略检索基础功能

### 第四阶段（1-2周）：测试优化
- 单元测试覆盖
- 开发调试工具完善
- 系统稳定运行验证

## RAGFlow组件集成策略

### 直接复用组件
1. **DeepDoc视觉识别模块**
   - OCR文字识别（vision/ocr.py）
   - 基础识别器（vision/recognizer.py）
   - 布局识别（vision/layout_recognizer.py）
   - 表格结构识别（vision/table_structure_recognizer.py）

2. **PDF解析器**
   - PDF解析器（parser/pdf_parser.py）
   - Office文档解析器
   - Web文档解析器

3. **GraphRAG模块**
   - General模式实现（graphrag/general/）
   - Light模式实现（graphrag/light/）
   - 实体抽取和关系解析

4. **数据库模型**
   - 多租户数据模型设计
   - UUID主键和软删除机制
   - 三范式设计原则

## 质量保证

### 代码质量工具
- **Black** - 代码格式化
- **iSort** - 导入排序
- **MyPy** - 静态类型检查
- **Flake8** - 代码风格检查
- **Pre-commit hooks** - Git钩子配置

### 单元测试策略
- 每个抽象接口和具体实现都有对应测试
- 测试覆盖率达到80%以上
- 支持测试数据的自动生成和清理
- 环境隔离，测试不影响开发数据

## 性能优化

### 异步并发处理
- 使用Trio异步框架提供高性能并发处理
- 支持任务的优先级调度
- 实现任务队列的流量控制

### 多级缓存策略
- 内存缓存用于热点数据
- Redis缓存用于跨服务数据共享
- 支持缓存的自动过期和更新

### 数据库优化
- 在tenant_id上建立复合索引
- 使用连接池管理数据库连接
- 实现查询结果的缓存机制

## 安全和权限

### 多租户数据隔离
- 通过tenant_id字段实现租户级数据隔离
- 所有查询自动添加租户过滤条件
- 确保租户间数据完全隔离

### API安全
- JWT token认证（预留）
- 请求频率限制
- 输入验证和SQL注入防护

## 监控和日志

### Loguru日志系统
- 支持多级别日志记录
- 按模块分类存储日志文件
- 支持日志文件的自动轮转和压缩

### 性能监控
- 收集API响应时间和并发能力
- 监控数据库连接状态
- 提供性能指标的告警和通知

## 部署策略

### 容器化部署
- Docker容器化应用
- Docker Compose编排服务
- 支持开发和生产环境配置

### 环境配置
- 开发、测试、生产环境分离配置
- 敏感配置使用环境变量管理
- 支持配置的热更新

## 总结

这个RAG项目重构提案集成了RAGFlow的最佳实践，采用了现代化的技术栈和架构设计。主要特点包括：

1. **RAGFlow深度集成** - 直接复用成熟的文档解析和GraphRAG组件
2. **多租户企业架构** - 支持多用户、多租户的企业级应用
3. **业务导向文档处理** - 基于文件来源的精细化处理策略
4. **渐进式开发** - 分阶段实施，风险可控
5. **高可扩展性** - 抽象接口设计，支持多种实现

这个架构将构建一个生产就绪的企业级RAG系统，具备高性能、高可用性和高扩展性。