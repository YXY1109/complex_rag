# Qwen3 Embedding & Rerank Service API Documentation

## 概述

基于 Qwen3 模型的向量嵌入和重排服务，提供完全兼容 OpenAI API 规范的接口。

- **向量模型**: [Qwen3-Embedding-0.6B](https://modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B/summary)
- **重排模型**: [Qwen3-Reranker-0.6B](https://modelscope.cn/models/Qwen/Qwen3-Reranker-0.6B/summary)
- **API规范**: OpenAI Compatible
- **服务地址**: `http://127.0.0.1:8000`
- **API文档**: `http://127.0.0.1:8000/docs/swagger`

## 接口列表

| 接口 | 方法 | 路径 | 描述 |
|------|------|------|------|
| 文本嵌入 | POST | `/v1/embeddings` | 生成文本的向量嵌入 |
| 文档重排 | POST | `/v1/rerank` | 对文档进行相关性重排序 |
| 健康检查 | GET | `/health` | 服务状态检查 |

---

## 1. 文本嵌入接口

### 接口信息
- **URL**: `/v1/embeddings`
- **方法**: `POST`
- **内容类型**: `application/json`

### 请求参数

| 参数名 | 类型 | 必填 | 描述 | 示例 |
|--------|------|------|------|------|
| input | string \| array | 是 | 输入文本，支持单个字符串或字符串数组 | `"Hello world"` 或 `["Hello", "World"]` |
| model | string | 否 | 模型名称，默认为 `qwen3-embedding` | `"qwen3-embedding"` |
| encoding_format | string | 否 | 编码格式，默认为 `float` | `"float"` |
| dimensions | integer | 否 | 嵌入向量维度，默认使用模型默认维度 | `1536` |
| user | string | 否 | 用户标识，用于请求追踪 | `"user123"` |

### 请求示例

#### 单文本嵌入
```bash
curl -X POST "http://127.0.0.1:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好，世界！",
    "model": "qwen3-embedding",
    "user": "test_user"
  }'
```

#### 批量文本嵌入
```bash
curl -X POST "http://127.0.0.1:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "人工智能是未来",
      "机器学习很强大",
      "深度学习改变了世界"
    ],
    "model": "qwen3-embedding",
    "dimensions": 768
  }'
```

### 响应格式

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, 0.3, ...],
      "index": 0
    }
  ],
  "model": "qwen3-embedding",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

### 响应字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| object | string | 对象类型，固定为 "list" |
| data | array | 嵌入数据数组 |
| data[0].object | string | 数据对象类型，固定为 "embedding" |
| data[0].embedding | array | 嵌入向量（浮点数数组） |
| data[0].index | integer | 索引位置 |
| model | string | 使用的模型名称 |
| usage.prompt_tokens | integer | 提示词token数量 |
| usage.total_tokens | integer | 总token数量 |

### 错误响应

```json
{
  "error": {
    "message": "Missing 'input' field in request",
    "type": "invalid_request_error",
    "code": "missing_input"
  }
}
```

---

## 2. 文档重排接口

### 接口信息
- **URL**: `/v1/rerank`
- **方法**: `POST`
- **内容类型**: `application/json`

### 请求参数

| 参数名 | 类型 | 必填 | 描述 | 示例 |
|--------|------|------|------|------|
| model | string | 否 | 模型名称，默认为 `qwen3-reranker` | `"qwen3-reranker"` |
| query | string | 是 | 查询文本 | `"什么是人工智能？"` |
| documents | array | 是 | 待重排的文档列表 | `["文档1", "文档2"]` |
| top_n | integer | 否 | 返回前N个结果，默认返回所有 | `5` |
| user | string | 否 | 用户标识，用于请求追踪 | `"user123"` |

### 请求示例

```bash
curl -X POST "http://127.0.0.1:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习？",
    "documents": [
      "机器学习是人工智能的一个分支",
      "深度学习基于神经网络",
      "今天天气很好",
      "机器学习算法可以从数据中学习模式"
    ],
    "model": "qwen3-reranker",
    "top_n": 3,
    "user": "test_user"
  }'
```

### 响应格式

```json
{
  "object": "list",
  "data": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": "机器学习是人工智能的一个分支"
    },
    {
      "index": 3,
      "relevance_score": 0.87,
      "document": "机器学习算法可以从数据中学习模式"
    },
    {
      "index": 1,
      "relevance_score": 0.65,
      "document": "深度学习基于神经网络"
    }
  ],
  "model": "qwen3-reranker",
  "usage": {
    "prompt_tokens": 25,
    "total_tokens": 25
  }
}
```

### 响应字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| object | string | 对象类型，固定为 "list" |
| data | array | 重排结果数组 |
| data[0].index | integer | 原始文档索引 |
| data[0].relevance_score | float | 相关性分数（0-1） |
| data[0].document | string | 文档内容 |
| model | string | 使用的模型名称 |
| usage.prompt_tokens | integer | 提示词token数量 |
| usage.total_tokens | integer | 总token数量 |

### 错误响应

```json
{
  "error": {
    "message": "Missing 'query' field in request",
    "type": "invalid_request_error",
    "code": "missing_query"
  }
}
```

---

## 3. 健康检查接口

### 接口信息
- **URL**: `/health`
- **方法**: `GET`

### 请求示例

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

### 响应格式

```json
{
  "status": "healthy",
  "models": {
    "embedding": {
      "name": "Qwen3-Embedding-0___6B",
      "available": true
    },
    "rerank": {
      "name": "Qwen3-Reranker-0___6B",
      "available": true
    }
  },
  "system": {
    "device": "cuda",
    "gpu_memory_allocated": "2.50 GB",
    "gpu_memory_reserved": "3.00 GB",
    "gpu_name": "NVIDIA RTX 4090"
  },
  "version": "1.0.0",
  "endpoints": [
    "/v1/embeddings",
    "/v1/rerank",
    "/health"
  ]
}
```

### 响应字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| status | string | 服务状态：`healthy` 或 `unhealthy` |
| models | object | 模型状态信息 |
| models.embedding.name | string | 嵌入模型名称 |
| models.embedding.available | boolean | 嵌入模型是否可用 |
| models.rerank.name | string | 重排模型名称 |
| models.rerank.available | boolean | 重排模型是否可用 |
| system | object | 系统信息 |
| system.device | string | 设备类型：`cuda` 或 `cpu` |
| system.gpu_memory_allocated | string | GPU已分配内存（仅GPU环境） |
| system.gpu_memory_reserved | string | GPU已保留内存（仅GPU环境） |
| system.gpu_name | string | GPU名称（仅GPU环境） |
| version | string | 服务版本号 |
| endpoints | array | 可用端点列表 |

---

## 错误码说明

### HTTP状态码

| 状态码 | 描述 |
|--------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 |

### 错误类型

| 错误类型 | 描述 |
|----------|------|
| `invalid_request_error` | 请求参数无效 |
| `internal_server_error` | 服务器内部错误 |

### 具体错误码

| 错误码 | 描述 | 出现场景 |
|--------|------|----------|
| `missing_input` | 缺少input参数 | 嵌入接口缺少输入文本 |
| `missing_query` | 缺少query参数 | 重排接口缺少查询文本 |
| `invalid_documents` | documents参数无效 | 重排接口文档列表格式错误 |
| `invalid_json` | JSON格式错误 | 请求体不是有效JSON |
| `embedding_generation_failed` | 嵌入生成失败 | 嵌入模型推理异常 |
| `reranking_failed` | 重排失败 | 重排模型推理异常 |

---

## 使用示例

### Python 示例

```python
import requests
import json

# 服务地址
BASE_URL = "http://127.0.0.1:8000"

# 1. 文本嵌入示例
def test_embedding():
    url = f"{BASE_URL}/v1/embeddings"
    data = {
        "input": ["人工智能", "机器学习", "深度学习"],
        "model": "qwen3-embedding",
        "user": "python_test"
    }

    response = requests.post(url, json=data)
    result = response.json()

    print("嵌入结果:")
    for item in result["data"]:
        print(f"文本 {item['index']}: 向量维度 {len(item['embedding'])}")

# 2. 文档重排示例
def test_rerank():
    url = f"{BASE_URL}/v1/rerank"
    data = {
        "query": "什么是深度学习？",
        "documents": [
            "深度学习是机器学习的一个子领域",
            "今天是个好天气",
            "神经网络是深度学习的基础",
            "深度学习在图像识别中表现出色"
        ],
        "top_n": 3,
        "user": "python_test"
    }

    response = requests.post(url, json=data)
    result = response.json()

    print("\n重排结果:")
    for item in result["data"]:
        print(f"分数: {item['relevance_score']:.3f} - {item['document']}")

# 3. 健康检查示例
def test_health():
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    result = response.json()

    print(f"\n服务状态: {result['status']}")
    print(f"嵌入模型: {result['models']['embedding']['available']}")
    print(f"重排模型: {result['models']['rerank']['available']}")

if __name__ == "__main__":
    test_embedding()
    test_rerank()
    test_health()
```

### JavaScript 示例

```javascript
const BASE_URL = "http://127.0.0.1:8000";

// 1. 文本嵌入
async function testEmbedding() {
    const response = await fetch(`${BASE_URL}/v1/embeddings`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            input: "你好，世界",
            model: "qwen3-embedding"
        })
    });

    const result = await response.json();
    console.log("嵌入结果:", result);
}

// 2. 文档重排
async function testRerank() {
    const response = await fetch(`${BASE_URL}/v1/rerank`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: "什么是人工智能？",
            documents: [
                "AI是人工智能的缩写",
                "今天天气很好",
                "人工智能正在改变世界"
            ],
            top_n: 2
        })
    });

    const result = await response.json();
    console.log("重排结果:", result);
}

// 执行测试
testEmbedding();
testRerank();
```

---

## 性能建议

### 嵌入服务
- **批量处理**: 尽量使用批量文本输入以提高效率
- **文本长度**: 单个文本建议不超过8192个token
- **维度选择**: 根据下游任务需求选择合适的嵌入维度

### 重排服务
- **文档数量**: 建议单次重排的文档数量不超过100个
- **查询长度**: 查询文本建议简洁明确
- **结果数量**: 使用 `top_n` 参数控制返回结果数量

### 系统优化
- **GPU加速**: 服务自动检测并使用GPU加速
- **并发控制**: 服务内置异步处理机制
- **内存管理**: 长时间运行建议定期监控GPU内存使用

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0.0 | 2024-01-01 | 初始版本，支持嵌入和重排服务，OpenAI API规范兼容 |

---

## 技术支持

如有问题或建议，请联系技术支持团队。

- **服务端口**: 8000
- **健康检查**: `/health`
- **API文档**: `/docs/swagger`
- **OpenAPI规范**: `/docs/openapi.json`