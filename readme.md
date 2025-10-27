# 复杂RAG服务系统

基于多模态文档解析和智能检索的高级RAG（Retrieval-Augmented Generation）服务系统。

## 🚀 特性

### 核心功能
- **多模态文档解析**: 支持PDF、Word、Excel、PowerPoint、图片、网页等多种格式
- **智能检索引擎**: 向量检索、全文检索、语义检索、混合检索
- **高级生成服务**: 支持多种LLM模型，链式思考、文档摘要、内容比较
- **对话系统**: 多轮对话、会话管理、上下文保持
- **知识库管理**: 多租户支持、权限控制、版本管理

### 技术特性
- **现代架构**: 基于FastAPI + SQLAlchemy + Pydantic的异步架构
- **高性能**: 支持并发处理、流式响应、批量操作
- **可扩展**: 插件化架构、微服务设计、容器化部署
- **高可用**: 多级缓存、故障转移、健康检查
- **可观测**: 结构化日志、性能监控、分布式追踪

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │   FastAPI App   │    │   RAG Engine    │
│   (Nginx)       │────│   (RESTful)     │────│   (Core Logic)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌───────────────────────────────┼───────────────────────────────┐
                       │                               │                               │
        ┌─────────────────┐                ┌─────────────────┐                ┌─────────────────┐
        │  Vector Store   │                │   Generation    │                │  Knowledge Mgr  │
        │  (Milvus)       │                │  (OpenAI/...)   │                │  (Multi-tenant) │
        └─────────────────┘                └─────────────────┘                └─────────────────┘
                       │                               │                               │
        ┌─────────────────┐                ┌─────────────────┐                ┌─────────────────┐
        │    Search DB    │                │     Cache       │                │  Object Storage │
        │ (Elasticsearch) │                │   (Redis)       │                │   (MinIO/S3)    │
        └─────────────────┘                └─────────────────┘                └─────────────────┘
                       │                               │                               │
        ┌─────────────────┐                ┌─────────────────┐                ┌─────────────────┐
        │   Database      │                │   Monitoring    │                │ Document Parser │
        │   (MySQL)       │                │(Prometheus/Graf)│                │   (RAGFlow)      │
        └─────────────────┘                └─────────────────┘                └─────────────────┘
```

## 📋 快速开始

### 前置要求

- Docker & Docker Compose
- Python 3.9+ (开发环境)
- 8GB+ RAM
- 20GB+ 磁盘空间

### 1. 克隆项目

```bash
git clone <repository-url>
cd complex-rag
```

### 2. 配置环境

```bash
# 复制环境配置文件
cp .env.example .env

# 编辑配置文件，设置API密钥等
vim .env
```

### 3. 启动服务

#### 选项1：统一FastAPI服务（推荐）

```bash
# Linux/macOS
./deploy-unified.sh deploy

# Windows
deploy-unified.bat
```

#### 选项2：传统部署方式

```bash
# 开发环境
./deployment/scripts/deploy.sh dev --build --seed

# 生产环境
./deployment/scripts/deploy.sh prod --build --migrate
```

### 4. 验证部署

#### 统一服务

```bash
# 检查服务状态
curl http://localhost:8000/health/ping

# 查看详细健康状态
curl http://localhost:8000/health/detailed

# 查看API文档
open http://localhost:8000/docs

# 查看OpenAPI规范
open http://localhost:8000/openapi.json
```

#### 管理命令

```bash
# 查看服务日志
./deploy-unified.sh logs

# 查看服务状态
./deploy-unified.sh status

# 重启服务
./deploy-unified.sh update

# 停止服务
./deploy-unified.sh stop

# 清理部署
./deploy-unified.sh cleanup
```

#### 独立服务端点

统一服务包含以下主要端点：

- **聊天服务**: `POST /v1/chat/completions`
- **嵌入服务**: `POST /v1/embeddings/`
- **重排序服务**: `POST /v1/rerank/`
- **记忆管理**: `/v1/memory/*`
- **健康检查**: `/health/*`

## 🔧 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API密钥 | - |
| `MYSQL_HOST` | MySQL主机地址 | localhost |
| `MILVUS_HOST` | Milvus主机地址 | localhost |
| `ELASTICSEARCH_HOSTS` | Elasticsearch地址 | localhost:9200 |
| `REDIS_HOST` | Redis主机地址 | localhost |
| `MINIO_ENDPOINT` | MinIO端点 | localhost:9000 |

### 主要配置文件

- `config.json` - 主配置文件
- `docker-compose.yml` - 生产环境容器编排
- `docker-compose.dev.yml` - 开发环境容器编排
- `deployment/nginx/` - Nginx配置
- `deployment/prometheus/` - 监控配置

## 📚 使用指南

### API接口

#### RAG查询
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是人工智能？",
    "retrieval_mode": "hybrid",
    "top_k": 5
  }'
```

#### 创建知识库
```bash
curl -X POST "http://localhost:8000/api/v1/knowledge" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "技术文档知识库",
    "description": "包含各种技术文档的知识库"
  }'
```

#### 上传文档
```bash
curl -X POST "http://localhost:8000/api/v1/documents/{kb_id}/file" \
  -F "file=@document.pdf" \
  -F "title=技术文档"
```

#### 聊天对话
```bash
curl -X POST "http://localhost:8000/api/v1/chat/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "技术学习对话",
    "knowledge_bases": ["kb_id"]
  }'

curl -X POST "http://localhost:8000/api/v1/chat/sessions/{session_id}/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Python是什么？"
  }'
```

### Python SDK

```python
import asyncio
from rag_service.api.main import create_app
from rag_service.examples.basic_usage import basic_qa_example

async def main():
    # 基础问答
    await basic_qa_example()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🛠️ 开发指南

### 开发环境设置

```bash
# 安装依赖
pip install uv
uv sync

# 启动开发服务
uv run python -m rag_service.api.main
```

### 代码结构

```
rag_service/
├── api/                    # API层
│   ├── routes/            # 路由定义
│   ├── middleware/        # 中间件
│   ├── dependencies/      # 依赖注入
│   └── exceptions/        # 异常处理
├── core/                  # 核心层
│   └── rag_engine.py      # RAG引擎
├── services/              # 服务层
│   ├── vector_store.py    # 向量存储
│   ├── embedding_service.py # 嵌入服务
│   ├── knowledge_manager.py # 知识管理
│   └── ...
├── infrastructure/        # 基础设施层
│   ├── database/         # 数据库
│   ├── cache/           # 缓存
│   ├── storage/         # 存储
│   └── monitoring/      # 监控
├── document_parsing/     # 文档解析层
│   ├── processors/      # 处理器
│   ├── pipeline/        # 管道
│   └── strategy/        # 策略
├── interfaces/           # 接口层
└── examples/            # 示例代码
```

### 添加新功能

1. 在`interfaces/`中定义抽象接口
2. 在`services/`中实现服务逻辑
3. 在`api/routes/`中添加API端点
4. 在`examples/`中提供使用示例
5. 更新文档和测试

## 📊 监控和运维

### 健康检查

```bash
# 基础健康检查
curl http://localhost:8000/health

# 详细健康检查
curl http://localhost:8000/health/detailed

# 就绪检查
curl http://localhost:8000/health/ready
```

### 监控服务

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MinIO**: http://localhost:9001 (minioadmin/minioadmin)

### 运维脚本

```bash
# 部署服务
./deployment/scripts/deploy.sh prod --build --migrate

# 备份数据
./deployment/scripts/backup.sh prod all

# 监控检查
./deployment/scripts/monitor.sh prod all

# 查看日志
docker-compose logs -f rag-api
```

### 性能调优

1. **数据库优化**
   - 调整MySQL缓冲池大小
   - 优化查询索引
   - 启用查询缓存

2. **向量搜索优化**
   - 调整Milvus索引参数
   - 优化向量维度
   - 使用批量操作

3. **缓存策略**
   - 配置Redis缓存策略
   - 使用多级缓存
   - 设置合理的TTL

## 🔒 安全配置

### 基础安全

- 更改默认密码
- 启用HTTPS/TLS
- 配置防火墙
- 限制API访问频率

### 数据加密

- 数据库连接加密
- 对象存储加密
- 传输过程加密
- 敏感信息脱敏

### 访问控制

- JWT认证
- RBAC权限控制
- 多租户隔离
- API密钥管理

## 📈 性能指标

### 基准性能

- **QPS**: 100+ (单实例)
- **响应时间**: < 500ms (P95)
- **并发用户**: 1000+
- **文档处理**: 10MB/s

### 扩展性

- **水平扩展**: 支持多实例部署
- **存储扩展**: 支持分布式存储
- **计算扩展**: 支持GPU加速
- **网络扩展**: 支持负载均衡

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

### 开发规范

- 遵循PEP 8代码风格
- 添加类型注解
- 编写单元测试
- 更新文档

## 📄 许可证

本项目采用MIT许可证，详见[LICENSE](LICENSE)文件。

## 🆘 支持

- **文档**: [项目文档](docs/)
- **问题反馈**: [GitHub Issues](issues)
- **讨论**: [GitHub Discussions](discussions)
- **邮件**: support@example.com

## 🗺️ 路线图

### v1.1 (计划中)
- [ ] 支持更多LLM模型
- [ ] 增强文档解析能力
- [ ] 优化向量搜索算法
- [ ] 增加Web管理界面

### v1.2 (计划中)
- [ ] 支持插件系统
- [ ] 增加数据版本管理
- [ ] 优化多语言支持
- [ ] 增强安全功能

### v2.0 (远期规划)
- [ ] 支持分布式部署
- [ ] 增加AI辅助功能
- [ ] 支持实时协作
- [ ] 增加移动端支持

---

**注意**: 这是一个复杂的企业级RAG系统，建议在生产环境部署前进行充分的测试和安全评估。

## 学习记录

- k8s教程： https://www.bilibili.com/video/BV1MT411x7GH/?p=29
- transformers教程：https://www.bilibili.com/video/BV1KX4y1a7Jk
- python高级进阶：开始看第七章