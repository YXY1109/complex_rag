## Context
这是一个RAG（Retrieval-Augmented Generation）系统的完全重构，目标是建立现代化、高性能、可扩展的智能问答系统。当前系统存在技术债务，需要按照Python异步最佳实践重新设计架构。

### 当前痛点
- 使用Celery处理异步任务，性能和资源利用率低
- 模型接口不统一，难以替换和扩展
- 项目结构混乱，功能模块耦合度高
- 文件解析功能单一，缺乏多模态支持
- 召回策略单一，检索效果有限
- 对话历史管理不完善
- 缺乏抽象层，扩展性差

### 约束条件
- 保持FastAPI作为主要Web框架
- 支持高并发异步处理
- 所有AI模型接口统一为OpenAI规范
- 必须支持前端动态切换不同实现
- 保持向后兼容的数据结构

## Goals / Non-Goals

### Goals
- **高并发性能**：使用Trio替代Celery，实现高效的并发处理
- **统一接口**：所有AI服务采用OpenAI兼容的API规范
- **模块化架构**：清晰的项目分层，每层都有明确的职责
- **可扩展性**：支持多种实现的插件式架构
- **多召回策略**：支持向量、关键词、重排、实体、RAPTOR等多种召回方式
- **多模态处理**：支持PDF→MD、图像、音频等多种文档解析
- **灵活配置**：支持前端动态切换不同的实现方案

### Non-Goals
- 重写已有数据库结构（保持兼容）
- 修改核心业务逻辑（重构架构而非功能）
- 引入复杂的微服务架构（保持单应用架构）

## Decisions

### Decision 1: 采用Trio作为主要异步框架
**理由**: Trio比Celery更适合I/O密集型任务，具有更好的错误处理和资源管理能力。
**替代方案**: asyncio（Trio提供更好的并发控制）、Celery（性能不如Trio）

### Decision 2: 统一OpenAI API规范
**理由**: OpenAI API已经成为事实上的标准，便于模型替换和生态兼容。
**替代方案**: 自定义API规范（增加复杂度，不利于生态集成）

### Decision 3: 抽象优先的设计原则
**理由**: 每个功能模块都定义抽象接口，支持多种实现，提高系统的灵活性。
**替代方案**: 直接实现具体功能（扩展性差，难以切换）

### Decision 4: 分层架构设计
**理由**: 清晰的职责分离，便于维护和测试。
**替代方案**: 单体架构（耦合度高，难以扩展）

### Decision 5: 插件式文件解析
**理由**: 支持多种文档格式和解析方式，适应不同场景需求。
**替代方案**: 固定的解析流程（灵活性差）

## 新项目目录结构

```
complex_rag/
├── api/                      # API层 - FastAPI应用
│   ├── __init__.py
│   ├── main.py              # FastAPI应用入口
│   ├── routers/             # API路由模块
│   │   ├── __init__.py
│   │   ├── chat.py          # 对话接口
│   │   ├── documents.py     # 文档管理接口
│   │   ├── knowledge.py     # 知识库管理接口
│   │   ├── models.py        # 模型管理接口
│   │   └── health.py        # 健康检查接口
│   ├── dependencies.py      # FastAPI依赖注入
│   ├── middleware.py        # 中间件
│   └── exceptions.py        # 异常处理
│
├── rag_service/             # RAG服务层 - Sanic高性能AI服务
│   ├── __init__.py
│   ├── app.py              # Sanic应用入口
│   ├── interfaces/         # AI服务抽象接口
│   │   ├── __init__.py
│   │   ├── llm_interface.py     # 大语言模型接口
│   │   ├── embedding_interface.py # 向量化接口
│   │   └── rerank_interface.py   # 重排接口
│   ├── providers/          # 具体的AI服务实现
│   │   ├── __init__.py
│   │   ├── openai/         # OpenAI接口实现
│   │   ├── ollama/         # Ollama本地模型
│   │   ├── qwen/           # 通义千问
│   │   └── bce/            # 百度文心
│   ├── routes/             # Sanic路由模块
│   │   ├── __init__.py
│   │   ├── llm.py          # LLM服务路由
│   │   ├── embeddings.py   # 向量化服务路由
│   │   └── rerank.py       # 重排服务路由
│   ├── services/           # RAG核心服务
│   │   ├── __init__.py
│   │   ├── chat_service.py      # 对话生成服务
│   │   ├── retrieval_service.py # 检索服务
│   │   └── memory_service.py    # 对话记忆服务
│   └── models/             # 数据传输对象
│       ├── __init__.py
│       ├── chat.py
│       └── retrieval.py
│
├── document_parser/         # 文档解析层（基于RAGFlow deepdoc + rag/app架构）
│   ├── __init__.py
│   ├── interfaces/          # 解析器抽象接口
│   │   ├── __init__.py
│   │   ├── parser_interface.py  # 文档解析接口
│   │   ├── converter_interface.py # 格式转换接口
│   │   └── source_interface.py    # 文件来源处理接口
│   ├── source_handlers/     # 文件来源处理器（参考RAGFlow rag/app）
│   │   ├── __init__.py
│   │   ├── web_documents/        # 网页文档处理器
│   │   │   ├── __init__.py
│   │   │   ├── html_handler.py    # HTML文档处理
│   │   │   ├── markdown_handler.py # Markdown文档处理
│   │   │   └── api_doc_handler.py  # API文档处理
│   │   ├── office_documents/      # Office文档处理器
│   │   │   ├── __init__.py
│   │   │   ├── pdf_handler.py     # PDF文档处理
│   │   │   ├── docx_handler.py    # Word文档处理
│   │   │   ├── excel_handler.py   # Excel表格处理
│   │   │   └── ppt_handler.py     # PPT演示处理
│   │   ├── scanned_documents/     # 扫描文档处理器
│   │   │   ├── __init__.py
│   │   │   ├── ocr_handler.py     # OCR扫描文档处理
│   │   │   ├── image_handler.py   # 图片文档处理
│   │   │   └── multi_modal_handler.py # 多模态文档处理
│   │   ├── structured_data/        # 结构化数据处理器
│   │   │   ├── __init__.py
│   │   │   ├── json_handler.py    # JSON数据处理
│   │   │   ├── csv_handler.py     # CSV数据处理
│   │   │   ├── xml_handler.py     # XML数据处理
│   │   │   └── yaml_handler.py    # YAML数据处理
│   │   ├── code_repositories/      # 代码仓库处理器
│   │   │   ├── __init__.py
│   │   │   ├── github_handler.py  # GitHub仓库处理
│   │   │   ├── code_file_handler.py # 代码文件处理
│   │   │   └── documentation_handler.py # 技术文档处理
│   │   └── custom_sources/         # 自定义来源处理器
│   │       ├── __init__.py
│   │       └── plugin_interface.py # 插件接口定义
│   ├── parsers/             # 具体解析器实现（参考RAGFlow deepdoc）
│   │   ├── __init__.py
│   │   ├── pdf/             # PDF解析器（参考RAGFlow pdf_parser.py）
│   │   │   ├── __init__.py
│   │   │   ├── ragflow_pdf_parser.py  # RAGFlow PDF解析器
│   │   │   └── mineru_parser.py        # Mineru解析实现（预留）
│   │   ├── office/          # Office文档解析（参考RAGFlow）
│   │   │   ├── __init__.py
│   │   │   ├── docx_parser.py          # DOCX解析器
│   │   │   ├── excel_parser.py         # Excel解析器
│   │   │   └── ppt_parser.py           # PPT解析器
│   │   ├── web/             # Web文档解析
│   │   │   ├── __init__.py
│   │   │   ├── html_parser.py          # HTML解析器
│   │   │   ├── json_parser.py          # JSON解析器
│   │   │   └── markdown_parser.py      # Markdown解析器
│   │   └── text/            # 纯文本解析
│   │       ├── __init__.py
│   │       └── txt_parser.py           # TXT解析器
│   ├── vision/              # 视觉识别模块（参考RAGFlow vision）
│   │   ├── __init__.py
│   │   ├── ocr.py               # OCR文字识别
│   │   ├── recognizer.py        # 基础识别器
│   │   ├── layout_recognizer.py # 布局识别
│   │   ├── table_structure_recognizer.py # 表格结构识别
│   │   ├── operators.py         # 图像预处理操作
│   │   └── postprocess.py       # 后处理算法
│   ├── preprocessors/       # 文档预处理器
│   │   ├── __init__.py
│   │   ├── chunker.py       # 文档分块
│   │   ├── cleaner.py       # 文档清理
│   │   └── extractor.py     # 信息提取
│   ├── strategies/          # 处理策略配置
│   │   ├── __init__.py
│   │   ├── source_detector.py    # 文件来源检测器
│   │   ├── strategy_selector.py  # 处理策略选择器
│   │   └── quality_monitor.py    # 处理质量监控
│   └── services/            # 解析服务
│       ├── __init__.py
│       ├── parser_service.py    # 解析调度服务
│       ├── source_service.py    # 来源处理服务
│       ├── pipeline_service.py  # 处理流水线服务
│       └── conversion_service.py # 格式转换服务
│
├── core_rag/               # 核心RAG层 - 业务逻辑
│   ├── __init__.py
│   ├── retrieval/          # 检索引擎
│   │   ├── __init__.py
│   │   ├── interfaces/         # 检索器抽象接口
│   │   │   ├── __init__.py
│   │   │   ├── vector_retriever.py   # 向量检索接口
│   │   │   ├── keyword_retriever.py  # 关键词检索接口
│   │   │   ├── entity_retriever.py   # 实体检索接口
│   │   │   ├── sparse_retriever.py   # 稀疏检索接口
│   │   │   └── raptor_retriever.py   # RAPTOR检索接口
│   │   ├── implementations/    # 具体检索器实现
│   │   │   ├── __init__.py
│   │   │   ├── vector/          # 向量检索实现
│   │   │   │   ├── milvus_retriever.py    # 首个实现
│   │   │   │   └── weaviate_retriever.py  # 预留
│   │   │   ├── keyword/         # 关键词检索实现
│   │   │   │   ├── es_retriever.py         # 首个实现
│   │   │   │   └── whoosh_retriever.py    # 预留
│   │   │   ├── entity/          # 实体检索实现（借鉴RAGFlow）
│   │   │   ├── sparse/          # 稀疏检索实现（预留）
│   │   │   └── raptor/          # RAPTOR层次化检索实现（预留）
│   │   ├── fusion/             # 检索结果融合
│   │   │   ├── __init__.py
│   │   │   ├── fusion_service.py
│   │   │   └── fusion_strategies.py
│   │   └── reranking/          # 重排序服务
│   │       ├── __init__.py
│   │       └── rerank_service.py
│   ├── memory/             # 对话记忆管理
│   │   ├── __init__.py
│   │   ├── interfaces/         # 记忆接口
│   │   │   ├── __init__.py
│   │   │   └── memory_interface.py
│   │   ├── implementations/    # 具体实现
│   │   │   ├── __init__.py
│   │   │   ├── mem0_memory.py   # Mem0实现
│   │   │   └── redis_memory.py  # Redis实现（预留）
│   │   └── services/
│   │       ├── __init__.py
│   │       └── memory_service.py
│   ├── context/            # 上下文处理和问题重写
│   │   ├── __init__.py
│   │   ├── interfaces/         # 上下文处理接口
│   │   │   ├── __init__.py
│   │   │   ├── context_interface.py     # 上下文处理接口
│   │   │   └── rewrite_interface.py     # 问题重写接口
│   │   ├── implementations/    # 具体实现
│   │   │   ├── __init__.py
│   │   │   ├── mem0_context.py      # Mem0上下文实现
│   │   │   ├── query_rewriter.py     # 查询重写实现
│   │   │   └── query_decomposer.py   # 问题拆分实现
│   │   └── services/
│   │       ├── __init__.py
│   │       └── context_service.py
│   ├── langgraph/          # LangGraph召回组件架构
│   │   ├── __init__.py
│   │   ├── nodes/              # LangGraph节点定义
│   │   │   ├── __init__.py
│   │   │   ├── vector_node.py          # 向量检索节点
│   │   │   ├── keyword_node.py         # 关键词检索节点
│   │   │   ├── entity_node.py          # 实体检索节点
│   │   │   ├── sparse_node.py          # 稀疏检索节点
│   │   │   └── raptor_node.py          # RAPTOR检索节点
│   │   ├── workflows/         # 工作流定义
│   │   │   ├── __init__.py
│   │   │   ├── retrieval_workflow.py   # 检索工作流
│   │   │   └── fusion_workflow.py      # 融合工作流
│   │   └── services/
│   │       ├── __init__.py
│   │       └── langgraph_service.py
│   ├── graph_rag/          # GraphRAG实现（参考RAGFlow graphrag）
│   │   ├── __init__.py
│   │   ├── interfaces/         # GraphRAG抽象接口
│   │   │   ├── __init__.py
│   │   │   ├── graph_interface.py    # 图数据库接口
│   │   │   ├── entity_interface.py   # 实体抽取接口
│   │   │   └── search_interface.py   # 图检索接口
│   │   ├── general/            # General模式（微软GraphRAG）
│   │   │   ├── __init__.py
│   │   │   ├── extraction.py      # 多轮实体抽取
│   │   │   ├── resolution.py       # 实体解析
│   │   │   └── community.py        # 社区发现
│   │   ├── light/              # Light模式（轻量化）
│   │   │   ├── __init__.py
│   │   │   ├── extraction.py      # 单次实体抽取
│   │   │   └── summarization.py   # 关键词总结
│   │   ├── graph/              # 图数据结构
│   │   │   ├── __init__.py
│   │   │   ├── graph_model.py     # 图数据模型
│   │   │   ├── node_model.py      # 节点模型
│   │   │   └── edge_model.py      # 边模型
│   │   ├── search/             # 图检索算法
│   │   │   ├── __init__.py
│   │   │   ├── kg_search.py       # 知识图谱检索
│   │   │   ├── entity_search.py   # 实体检索
│   │   │   └── path_search.py     # 路径检索
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── graph_service.py   # 图管理服务
│   │       └── search_service.py  # 检索服务
│   ├── light_rag/          # LightRAG实现（预留扩展）
│   │   ├── __init__.py
│   │   ├── interfaces/
│   │   ├── implementations/
│   │   └── services/
│   ├── pipeline/           # RAG处理流水线
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py      # 主RAG流水线
│   │   ├── document_pipeline.py # 文档处理流水线
│   │   └── query_pipeline.py    # 查询处理流水线
│   └── strategies/         # 策略模式实现
│       ├── __init__.py
│       ├── retrieval_strategies.py
│       └── generation_strategies.py
│
├── infrastructure/        # 基础设施层
│   ├── __init__.py
│   ├── database/           # 数据库抽象和实现
│   │   ├── __init__.py
│   │   ├── interfaces/         # 数据库接口
│   │   │   ├── __init__.py
│   │   │   ├── vector_db_interface.py  # 向量数据库接口
│   │   │   ├── relational_db_interface.py # 关系数据库接口
│   │   │   └── search_db_interface.py    # 搜索数据库接口
│   │   ├── implementations/    # 具体实现
│   │   │   ├── __init__.py
│   │   │   ├── vector/          # 向量数据库
│   │   │   │   ├── milvus_client.py    # 首个实现
│   │   │   │   └── weaviate_client.py  # 预留
│   │   │   ├── relational/      # 关系数据库
│   │   │   │   ├── mysql_client.py     # 首个实现
│   │   │   │   └── postgresql_client.py # 预留
│   │   │   └── search/          # 搜索数据库
│   │   │       ├── elasticsearch_client.py # 首个实现
│   │   │       └── opensearch_client.py   # 预留
│   │   └── models/             # 数据模型（符合三范式+多租户）
│   │       ├── __init__.py
│   │       ├── base_model.py         # 基础数据模型
│   │       ├── tenant_base_model.py  # 多租户基础模型
│   │       ├── user_model.py         # 用户模型
│   │       ├── tenant_model.py       # 租户模型
│   │       ├── user_tenant_model.py  # 用户租户关联模型
│   │       ├── knowledge_model.py    # 知识库模型
│   │       ├── document_model.py     # 文档实体模型
│   │       ├── chat_model.py         # 对话实体模型
│   │       ├── graph_model.py        # 图数据模型
│   │       └── config_model.py       # 配置模型
│   ├── storage/            # 对象存储
│   │   ├── __init__.py
│   │   ├── interfaces/         # 存储接口
│   │   │   ├── __init__.py
│   │   │   └── storage_interface.py
│   │   └── implementations/    # 具体实现
│   │       ├── __init__.py
│   │       ├── minio_storage.py
│   │       ├── s3_storage.py
│   │       └── local_storage.py
│   ├── cache/              # 缓存服务
│   │   ├── __init__.py
│   │   ├── interfaces/         # 缓存接口
│   │   │   ├── __init__.py
│   │   │   └── cache_interface.py
│   │   └── implementations/    # 具体实现
│   │       ├── __init__.py
│   │       ├── redis_cache.py
│   │       └── memory_cache.py
│   ├── queue/              # 消息队列
│   │   ├── __init__.py
│   │   ├── trio_queue.py       # Trio异步队列实现
│   │   └── services/           # 队列服务
│   │       ├── __init__.py
│   │       └── async_task_service.py
│   └── monitoring/         # 监控和日志
│       ├── __init__.py
│       ├── metrics.py          # 指标收集
│       ├── tracing.py          # 链路追踪
│       └── loguru_logger.py    # Loguru日志配置
│
├── assets/                # 模型文件和资源存储
│   ├── models/             # AI模型文件
│   │   ├── __init__.py
│   │   ├── llm_models/         # 大语言模型
│   │   ├── embedding_models/   # 向量化模型
│   │   ├── rerank_models/      # 重排模型
│   │   └── ocr_models/         # OCR模型
│   ├── weights/             # 模型权重文件
│   │   ├── __init__.py
│   │   └── model_weights/
│   ├── nltk_data/          # NLTK数据文件
│   │   ├── __init__.py
│   │   ├── tokenizers/
│   │   ├── corpora/
│   │   └── taggers/
│   └── static_resources/    # 静态资源文件
│       ├── __init__.py
│       ├── templates/
│       └── dictionaries/
│
├── config/                # 配置管理（分离式配置）
│   ├── __init__.py
│   ├── settings.py         # 主配置文件
│   ├── loguru_config.py    # Loguru日志配置
│   ├── environments/       # 环境配置
│   │   ├── __init__.py
│   │   ├── development.py
│   │   ├── production.py
│   │   └── testing.py
│   ├── services/           # 服务配置分离
│   │   ├── __init__.py
│   │   ├── api_config.py       # API服务配置
│   │   ├── rag_service_config.py # RAG服务配置
│   │   ├── db_config.py        # 数据库配置
│   │   ├── cache_config.py     # 缓存配置
│   │   ├── storage_config.py   # 存储配置
│   │   └── ai_models_config.py # AI模型配置
│   ├── ragflow_configs/    # RAGFlow配置参考
│   │   ├── __init__.py
│   │   ├── deepdoc_config.py   # DeepDoc解析配置
│   │   └── graphrag_config.py  # GraphRAG配置
│   └── secrets/            # 敏感配置（生产环境）
│       ├── __init__.py
│       ├── api_keys.py
│       ├── db_credentials.py
│       └── model_keys.py
│
├── quality/               # 代码质量检测配置
│   ├── pyproject.toml      # Black、isort、mypy配置
│   ├── .flake8             # Flake8代码风格配置
│   ├── .pre-commit-config.yaml  # Git hooks配置
│   ├── pytest.ini         # Pytest测试配置
│   └── .coverage.toml      # 代码覆盖率配置
│
├── tests/                 # 单元测试
│   ├── __init__.py
│   ├── unit/               # 单元测试
│   │   ├── test_api/
│   │   ├── test_rag_service/
│   │   ├── test_document_parser/
│   │   ├── test_core_rag/
│   │   └── test_infrastructure/
│   └── fixtures/           # 测试数据和模拟对象
│       ├── __init__.py
│       ├── mock_data.py
│       └── test_documents/
│
├── scripts/               # 脚本工具
│   ├── __init__.py
│   ├── setup.py           # 环境设置脚本
│   ├── start_services.py  # 服务启动脚本
│   ├── clear_all_data.py  # 一键清理所有数据脚本
│   ├── validate_env.py    # 环境验证脚本
│   └── dev_tools.py       # 开发调试工具
│
├── logs/                  # 日志文件存储
│   ├── api/               # API服务日志
│   ├── rag_service/       # RAG服务日志
│   ├── document_parser/   # 文档解析日志
│   ├── core_rag/          # 核心RAG日志
│   └── infrastructure/    # 基础设施日志
│
├── docs/                  # 文档
│   ├── api/               # API文档
│   ├── architecture/      # 架构文档
│   └── deployment/        # 部署文档
│
├── pyproject.toml         # 项目配置（使用uv）
├── requirements.txt       # 依赖列表
├── README.md             # 项目说明
├── .env.example          # 环境变量示例
├── docker-compose.yml    # Docker编排
├── Dockerfile            # Docker镜像
└── .gitignore           # Git忽略文件
```

## Risks / Trade-offs

### 风险
1. **重构复杂度高** - 完全重写可能引入新的bug
2. **兼容性风险** - 数据结构和API接口的兼容性
3. **开发时间成本** - 重构需要较长的开发周期
4. **学习成本** - 团队需要学习新的架构和工具

### 缓解措施
1. **渐进式迁移** - 分阶段重构，保持核心功能稳定
2. **充分的测试** - 建立完整的测试体系
3. **详细的文档** - 提供清晰的架构文档和使用指南
4. **并行开发** - 在不影响现有系统的情况下开发新架构

### 权衡
- **性能 vs 复杂度**: 增加的抽象层可能带来轻微的性能开销，但换取了更好的可扩展性
- **灵活性 vs 简单性**: 支持多种实现增加了系统的灵活性，但也增加了配置和管理的复杂度
- **标准化 vs 定制化**: 统一接口规范便于生态集成，但可能限制某些定制化需求

## 实施计划

### 阶段1: 基础架构搭建（2-3周）
- [ ] 搭建新的项目目录结构
- [ ] 配置uv包管理和Trio异步框架
- [ ] 实现基础的抽象接口层
- [ ] 配置代码质量检测工具
- [ ] 配置开发环境和CI/CD

### 阶段2: 核心功能实现（3-4周）
- [ ] 实现RAG核心逻辑
- [ ] 实现统一的OpenAI API接口
- [ ] 实现Sanic高性能AI服务
- [ ] 实现文档解析抽象层

### 阶段3: 高级功能实现（2-3周）
- [ ] 实现多召回策略
- [ ] 集成Mem0对话记忆
- [ ] 实现多模态文档解析
- [ ] 完善监控和日志系统

### 阶段4: 测试和优化（1-2周）
- [ ] 完整的单元测试
- [ ] 性能测试和优化
- [ ] 文档完善
- [ ] 生产环境部署

## Open Questions

1. **性能基准**: 新架构的性能目标应该如何设定？
2. **前端集成**: 如何为后续的Gradio前端集成设计合适的接口？
3. **配置管理**: 多实现方案的管理和切换机制如何设计？
4. **部署策略**: 新架构的部署流程和监控如何建立？
5. **模型管理**: 不同AI模型服务的资源调度和负载均衡如何设计？