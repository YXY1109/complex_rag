# BCE服务 OpenAI兼容接口文档

本文档描述了BCE服务的OpenAI兼容接口实现。

## 概述

BCE服务提供了完全兼容OpenAI API规范的向量和重排序接口。当前版本为简化版本，专注于OpenAI兼容的核心功能。

## 服务启动

```bash
python service.py --host 0.0.0.0 --port 7001 --workers 1 --use_gpu True
```

## OpenAI兼容接口

### 1. 向量化接口 `/v1/embeddings`

**请求方式**: POST
**Content-Type**: application/json

#### 请求参数

| 参数              | 类型              | 必填 | 说明                                 |
|-----------------|-----------------|----|------------------------------------|
| input           | string/string[] | 是  | 输入文本，支持单个字符串或字符串数组                 |
| model           | string          | 否  | 模型名称，默认 "bce-embedding-base_v1"    |
| encoding_format | string          | 否  | 编码格式，"float" 或 "base64"，默认 "float" |
| dimensions      | integer         | 否  | 输出向量维度，默认为模型原始维度                   |

#### 请求示例

```bash
curl -X POST http://localhost:7001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["这是一个测试文本", "这是另一个测试文本"],
    "model": "bce-embedding-base_v1",
    "encoding_format": "float"
  }'
```

#### 响应格式

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        0.1234,
        -0.5678,
        ...
      ],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [
        0.2345,
        -0.6789,
        ...
      ],
      "index": 1
    }
  ],
  "model": "bce-embedding-base_v1",
  "usage": {
    "prompt_tokens": 20,
    "total_tokens": 20
  }
}
```

### 2. 重排序接口 `/v1/rerank`

**请求方式**: POST
**Content-Type**: application/json

#### 请求参数

| 参数        | 类型       | 必填 | 说明                             |
|-----------|----------|----|--------------------------------|
| query     | string   | 是  | 查询文本                           |
| documents | string[] | 是  | 待排序的文档列表                       |
| model     | string   | 否  | 模型名称，默认 "bce-reranker-base_v1" |
| top_n     | integer  | 否  | 返回前N个结果，默认返回全部                 |

#### 请求示例

```bash
curl -X POST http://localhost:7001/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是人工智能",
    "documents": [
      "人工智能是指由人制造出来的机器所表现出来的智能",
      "机器学习是人工智能的一个子领域",
      "今天天气晴朗适合外出游玩"
    ],
    "model": "bce-reranker-base_v1",
    "top_n": 3
  }'
```

#### 响应格式

```json
{
  "id": "rerank-1638360000000",
  "model": "bce-reranker-base_v1",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.9542,
      "document": {
        "text": "人工智能是指由人制造出来的机器所表现出来的智能"
      }
    },
    {
      "index": 1,
      "relevance_score": 0.8723,
      "document": {
        "text": "机器学习是人工智能的一个子领域"
      }
    }
  ],
  "usage": {
    "total_tokens": 45
  }
}
```

### 3. 健康检查接口 `/health`

**请求方式**: GET

#### 响应格式

```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T10:30:00.000Z",
  "models": {
    "embedding": {
      "loaded": true,
      "model": "bce-embedding-base_v1"
    },
    "rerank": {
      "loaded": true,
      "model": "bce-reranker-base_v1"
    }
  },
  "device": {
    "type": "cuda",
    "gpu_available": true,
    "gpu_memory_used_gb": 2.34
  },
  "version": "1.0.0"
}
```

### 4. 测试接口 `/test`

**请求方式**: GET

#### 响应格式

```json
{
  "test": "我是bce测试接口：2023-12-01 10:30:00"
}
```

## 错误处理

所有接口都遵循统一的错误响应格式：

```json
{
  "error": {
    "message": "错误描述信息",
    "type": "error_type"
  }
}
```

常见错误类型：

- `invalid_request_error`: 请求参数错误
- `internal_server_error`: 服务器内部错误

## 测试

使用提供的测试脚本验证接口功能：

```bash
# 运行所有测试
python test_openai_api.py --url http://localhost:7001

# 单独测试向量化
python test_openai_api.py --test embeddings

# 单独测试重排序
python test_openai_api.py --test rerank

# 测试健康检查
python test_openai_api.py --test health

```

## 性能优化

1. **异步处理**: 使用 `asyncio` 和 `run_in_executor` 实现异步模型推理
2. **批处理**: 向量化支持批量输入，提高处理效率
3. **GPU加速**: 支持GPU计算，自动检测GPU可用性
4. **内存优化**: 合理的内存管理，避免内存泄漏

## 使用建议

1. **批量处理**: 尽量使用批量向量化，提高效率
2. **文本长度**: BCE模型对中文文本长度有一定限制，建议单段文本不超过512个字符
3. **并发控制**: 根据服务器配置合理控制并发请求数
4. **监控**: 定期检查 `/health` 接口监控服务状态

## 技术细节

- **框架**: Sanic (异步Web框架)
- **模型**: SentenceTransformer + CrossEncoder
- **向量化**: BCE-Embedding-base_v1 (768维)
- **重排序**: BCE-Reranker-base_v1
- **支持格式**: Float数组、Base64编码
- **并发处理**: 异步IO + 线程池
