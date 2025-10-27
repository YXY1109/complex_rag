# 统一嵌入服务

## 概述

统一嵌入服务是对原有BCE、Qwen3、Generic三个独立嵌入服务的整合，提供了统一、高效、可扩展的文本嵌入解决方案。

## 特性

### 🚀 核心功能
- **多模型支持**: 支持BCE、Qwen3、OpenAI等多种嵌入模型
- **统一接口**: 提供OpenAI兼容的REST API接口
- **智能缓存**: 内存缓存支持，提升重复查询性能
- **批量处理**: 高效的批量文本嵌入处理
- **生命周期管理**: 智能的模型加载和卸载管理

### 🛠️ 技术特性
- **可插拔后端**: 模块化的模型后端架构
- **异步处理**: 基于asyncio的异步处理
- **设备自适应**: 自动检测和适配CPU/GPU设备
- **容错设计**: 完善的错误处理和恢复机制
- **监控友好**: 内置健康检查和性能监控

## 支持的模型

### BCE嵌入模型
- **模型名称**: `bce-base`
- **类型**: 本地模型
- **维度**: 768
- **特点**: 中文优化，性能优秀

### Qwen3嵌入模型
- **模型名称**: `qwen3-embedding`
- **类型**: 本地模型
- **维度**: 1536
- **特点**: 通用性强，多语言支持

### OpenAI嵌入模型
- **模型名称**: `openai-text-embedding-3-small/large`
- **类型**: 云端API
- **维度**: 1536/3072
- **特点**: 高质量，付费服务

## API接口

### 1. 生成文本嵌入（OpenAI兼容）

```http
POST /v1/embeddings/
```

**请求体**:
```json
{
    "input": "要嵌入的文本或文本列表",
    "model": "bce-base",  // 可选，为空时使用默认模型
    "normalize": true,      // 可选，是否归一化
    "use_cache": true,     // 可选，是否使用缓存
    "batch_size": 32       // 可选，批量大小
}
```

**响应**:
```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [0.1, 0.2, ...],
            "index": 0
        }
    ],
    "model": "bce-base",
    "usage": {
        "prompt_tokens": 10,
        "total_tokens": 10
    }
}
```

### 2. 计算文本相似度

```http
POST /v1/embeddings/similarity
```

**请求体**:
```json
{
    "text1": "第一个文本",
    "text2": "第二个文本",
    "model": "bce-base"  // 可选
}
```

**响应**:
```json
{
    "similarity_score": 0.8542,
    "model": "bce-base",
    "processing_time": 0.045
}
```

### 3. 批量处理

```http
POST /v1/embeddings/batch
```

**查询参数**:
- `texts`: 文本列表
- `model`: 模型名称（可选）
- `batch_size`: 批量大小（可选）
- `use_cache`: 是否使用缓存（可选）

### 4. 列出可用模型

```http
GET /v1/embeddings/models
```

**响应**:
```json
{
    "object": "list",
    "data": [
        {
            "name": "bce-base",
            "type": "bce",
            "dimension": 768,
            "loaded": true,
            "is_default": true,
            "priority": 1
        }
    ]
}
```

### 5. 健康检查

```http
GET /v1/embeddings/health
```

## 配置

### 基本配置

```python
from config.unified_embedding_config import get_unified_embedding_config

config = get_unified_embedding_config()
```

### 模型配置示例

```python
{
    "models": {
        "my-custom-model": {
            "model_type": "sentence_transformer",
            "model_path": "/path/to/model",
            "device": "cuda",
            "use_gpu": True,
            "dimension": 768,
            "max_length": 512,
            "batch_size": 32,
            "cache_enabled": True,
            "priority": 1
        }
    },
    "default_model": "my-custom-model"
}
```

### 缓存配置

```python
"cache": {
    "enabled": True,
    "ttl": 3600,        // 缓存时间（秒）
    "max_size": 100000   // 最大缓存条目数
}
```

## 性能优化

### 1. 模型预加载
服务启动时自动预加载默认模型，减少首次请求延迟。

### 2. 智能缓存
- 基于文本内容的哈希缓存
- 可配置的缓存过期时间
- 自动缓存清理机制

### 3. 批量处理优化
- 自动批量请求合并
- 可配置的批量大小
- 内存使用优化

### 4. 设备优化
- 自动检测GPU可用性
- 智能内存管理
- 模型卸载/重载机制

## 监控和日志

### 健康检查
```bash
curl http://localhost:8000/v1/embeddings/health
```

### 性能指标
- 总请求数
- 缓存命中率
- 平均处理时间
- 模型加载统计

### 结构化日志
```python
import logging

logger = logging.getLogger("api.unified_embeddings")
logger.info("嵌入请求", extra={
    "model": "bce-base",
    "text_count": 5,
    "cache_hit": True
})
```

## 使用示例

### Python客户端

```python
import httpx
import asyncio

async def generate_embeddings():
    async with httpx.AsyncClient() as client:
        # 生成嵌入
        response = await client.post(
            "http://localhost:8000/v1/embeddings/",
            json={
                "input": "这是一个测试文本",
                "model": "bce-base"
            }
        )
        result = response.json()
        embedding = result['data'][0]['embedding']
        return embedding

# 运行
embedding = asyncio.run(generate_embeddings())
print(f"嵌入向量维度: {len(embedding)}")
```

### 批量处理

```python
async def batch_embeddings():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/embeddings/batch",
            params={
                "model": "bce-base",
                "batch_size": 16
            },
            json=["文本1", "文本2", "文本3"]
        )
        return response.json()

results = asyncio.run(batch_embeddings())
print(f"生成了 {len(results['embeddings'])} 个嵌入向量")
```

### 相似度计算

```python
async def compute_similarity():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/embeddings/similarity",
            json={
                "text1": "苹果",
                "text2": "橙子",
                "model": "bce-base"
            }
        )
        result = response.json()
        return result['similarity_score']

similarity = asyncio.run(compute_similarity())
print(f"相似度分数: {similarity}")
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径
   - 确认GPU驱动和CUDA版本
   - 查看服务日志获取详细错误

2. **内存不足**
   - 减小批量大小
   - 启用模型缓存管理
   - 考虑使用更小的模型

3. **性能问题**
   - 启用缓存
   - 调整批量大小
   - 使用GPU加速

### 日志查看

```bash
# 查看应用日志
docker logs -f complex-rag-unified

# 查看嵌入服务特定日志
docker logs complex-rag-unified | grep "unified_embeddings"
```

## 迁移指南

### 从BCE服务迁移

**旧接口**:
```http
POST /bce_embedding
```

**新接口**:
```http
POST /v1/embeddings/
```

### 从Qwen3服务迁移

**旧接口**:
```http
POST /embeddings
```

**新接口**:
```http
POST /v1/embeddings/
```

### 兼容性说明
- 请求格式完全兼容OpenAI接口
- 响应格式保持一致
- 无需修改现有客户端代码

## 开发和扩展

### 添加新的模型后端

1. 继承`EmbeddingBackend`基类
2. 实现必要的抽象方法
3. 在统一服务中注册新后端

```python
class CustomEmbeddingBackend(EmbeddingBackend):
    async def load_model(self):
        # 实现模型加载
        pass

    async def embed(self, texts):
        # 实现嵌入生成
        pass

    # ... 其他必要方法
```

### 自定义配置

可以通过环境变量或配置文件自定义服务行为：

```bash
# 环境变量
export EMBEDDING_DEFAULT_MODEL="qwen3-embedding"
export EMBEDDING_CACHE_TTL="7200"
export EMBEDDING_MAX_CONCURRENT="20"
```

## 版本历史

- **v2.0.0**: 统一嵌入服务发布，整合BCE、Qwen3、OpenAI
- **v1.x.x**: 独立的嵌入服务

## 许可证

本项目采用MIT许可证，详见LICENSE文件。