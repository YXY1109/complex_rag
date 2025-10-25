# RAG重构项目快速参考

## 🎯 项目核心目标
从零开始构建现代化、高性能、可扩展的企业级RAG系统，集成RAGFlow最佳实践。

## 🏗️ 核心架构分层

### 1. API层 (FastAPI)
- **职责**: RESTful API接口，用户交互入口
- **关键组件**: 对话、文档、知识库、模型管理接口
- **特点**: 无用户认证、匿名访问、OpenAPI文档

### 2. RAG服务层 (Sanic)
- **职责**: AI模型服务，OpenAI兼容接口
- **关键组件**: LLM服务、向量化服务、重排服务
- **配置**: 单进程模式，性能优化

### 3. 文档解析层 (RAGFlow)
- **职责**: 基于文件来源的精细化文档处理
- **关键组件**: 来源检测、专用处理器、视觉识别
- **特色**: 5大来源分类、智能策略选择

### 4. 核心RAG层
- **职责**: RAG核心业务逻辑
- **关键组件**: Mem0上下文、LangGraph召回、GraphRAG
- **特色**: 多策略检索、实体召回、RAPTOR

### 5. 基础设施层
- **职责**: 底层服务和存储
- **关键组件**: 多租户数据库、向量存储、缓存、队列
- **特色**: 三范式设计、数据隔离

## 📁 目录结构速览

```
complex_rag/
├── api/                     # FastAPI接口
├── rag_service/             # Sanic AI服务
├── document_parser/         # 文档解析(RAGFlow)
│   ├── source_handlers/     # 来源处理器 ⭐
│   ├── vision/             # 视觉识别(RAGFlow)
│   ├── parsers/            # 解析器
│   └── strategies/         # 处理策略 ⭐
├── core_rag/               # 核心RAG逻辑
│   ├── context/            # Mem0上下文 ⭐
│   ├── langgraph/          # LangGraph召回 ⭐
│   ├── graph_rag/          # GraphRAG(RAGFlow) ⭐
│   └── retrieval/          # 检索引擎
├── infrastructure/         # 基础设施
│   ├── database/           # 数据库(多租户) ⭐
│   └── storage/            # 对象存储
├── config/                 # 配置管理
├── assets/                 # 模型文件
├── scripts/                # 工具脚本
└── tests/                  # 单元测试
```

## 🔧 技术栈

### 框架
- **FastAPI** - API层
- **Sanic** - AI服务层
- **Trio** - 异步任务

### AI/ML
- **LangChain** - LLM框架
- **OpenAI API** - 统一接口规范
- **Mem0** - 对话记忆
- **RAGFlow** - 文档解析+GraphRAG

### 数据存储
- **MySQL** - 业务数据库(多租户)
- **Milvus** - 向量数据库
- **Elasticsearch** - 搜索引擎
- **Redis** - 缓存/队列
- **MinIO** - 对象存储

### 开发工具
- **uv** - 包管理器
- **Black/iSort/MyPy** - 代码质量
- **Loguru** - 结构化日志

## 🎯 RAGFlow集成策略

### 直接复用组件
```bash
# DeepDoc视觉识别模块
document_parser/vision/
├── ocr.py                 # OCR文字识别
├── recognizer.py          # 基础识别器
├── layout_recognizer.py   # 布局识别
└── table_structure_recognizer.py # 表格识别

# PDF解析器
document_parser/parsers/pdf/ragflow_pdf_parser.py

# GraphRAG模块
core_rag/graph_rag/
├── general/               # 微软GraphRAG
└── light/                 # LightRAG

# 数据库模型
infrastructure/database/models/
├── user_model.py
├── tenant_model.py
├── knowledge_model.py
└── document_model.py
```

### 文件来源处理器设计
```python
# 5大来源分类
source_handlers/
├── web_documents/         # 网页文档 (HTML, Markdown, API)
├── office_documents/       # Office文档 (PDF, DOCX, Excel, PPT)
├── scanned_documents/      # 扫描文档 (OCR, 图片, 多模态)
├── structured_data/        # 结构化数据 (JSON, CSV, XML, YAML)
└── code_repositories/      # 代码仓库 (GitHub, 代码, 技术文档)
```

## 🚀 渐进式开发计划

### 阶段1 (2-3周): 基础架构 ✅
- [x] 项目目录结构
- [x] uv包管理器配置
- [x] 分离式服务配置
- [x] 多租户数据模型
- [x] 文件来源检测器

### 阶段2 (3-4周): 核心功能 🚧
- [ ] RAGFlow视觉识别集成
- [ ] 5大来源专用处理器
- [ ] MySQL+Milvus+Elasticsearch
- [ ] 基本RAG流程

### 阶段3 (2-3周): 高级功能 📋
- [ ] RAGFlow GraphRAG集成
- [ ] Mem0上下文处理
- [ ] LangGraph召回组件
- [ ] 多策略检索

### 阶段4 (1-2周): 测试优化 📋
- [ ] 单元测试覆盖
- [ ] 开发调试工具
- [ ] 性能优化
- [ ] 系统稳定性验证

## 💡 核心设计理念

### 1. 抽象优先，具体实现后置
- 每个功能都定义清晰的抽象接口
- 首个实现选择最成熟稳定的方案
- 其他实现作为备注预留，便于扩展

### 2. 业务导向的文档处理
- 超越通用解析，针对不同来源专门优化
- 智能检测文件来源，自动选择处理策略
- 内置质量监控和优化机制

### 3. RAGFlow最佳实践
- 直接复用生产级成熟组件
- 减少重复开发，加速进程
- 保证系统的稳定性和可靠性

### 4. 渐进式开发
- 先基础实现，确保功能可用
- 后高级功能，逐步完善
- 充分测试，保证质量

## 🛠️ 开发工具

### 一键清理数据
```bash
python scripts/clear_all_data.py
```
- 清理数据库所有数据
- 清理向量数据库
- 清理缓存和临时文件

### 服务启动管理
```bash
python scripts/start_services.py
```
- 启动API服务
- 启动RAG服务
- 健康检查

### 环境验证
```bash
python scripts/validate_env.py
```
- 检查依赖安装
- 验证数据库连接
- 验证模型完整性

## 📊 配置管理

### 分离式配置架构
```
config/
├── services/              # 服务配置 ⭐
├── ragflow_configs/        # RAGFlow参考配置
└── secrets/                # 敏感配置
```

### 关键配置文件
- `config/services/api_config.py` - API服务配置
- `config/services/rag_service_config.py` - RAG服务配置
- `config/services/db_config.py` - 数据库配置
- `config/services/ai_models_config.py` - AI模型配置

## 🔍 质量保证

### 代码质量工具
- **Black** - 代码格式化
- **iSort** - 导入排序
- **MyPy** - 类型检查
- **Flake8** - 风格检查

### 测试策略
- 单元测试覆盖率目标：80%+
- 环境隔离测试
- 自动化测试脚本

## 🎯 核心价值

✅ **企业级架构** - 多租户、数据隔离、权限管理
✅ **RAGFlow集成** - 生产级组件直接复用
✅ **业务导向处理** - 基于来源的精细化处理策略
✅ **高可扩展性** - 抽象接口设计，易于扩展
✅ **渐进式开发** - 分阶段实施，风险可控
✅ **开发友好** - 完整工具链，调试便利

## 📋 快速开始检查清单

### 开发环境准备
- [ ] Python 3.9+ 环境
- [ ] uv包管理器安装
- [ ] 项目依赖配置完成

### 核心功能验证
- [ ] 多租户数据库设计实现
- [ ] 文件来源检测器工作正常
- [ ] RAGFlow视觉识别模块集成
- [ ] 基础RAG流程可以运行

### 高级功能扩展
- [ ] GraphRAG功能集成
- [ ] Mem0上下文处理实现
- [ ] 多策略检索优化
- [ ] 自定义处理器插件开发

---

**注**: 这是一个完整的重构提案，建议仔细阅读文档后根据实际需求进行调整。所有设计都基于RAGFlow的最佳实践，确保系统的稳定性和可靠性。