# Complex RAG API Documentation

**版本:** 1.0.0
**基础URL:** http://localhost:8000

高性能RAG系统API服务，提供智能问答和文档检索功能

## 目录

- [健康检查](#健康检查)
  - [系统健康检查](#系统健康检查)
  - [详细健康检查](#详细健康检查)
  - [Ping检查](#ping检查)
- [对话服务](#对话服务)
  - [对话完成](#对话完成)
  - [流式对话完成](#流式对话完成)
- [文档管理](#文档管理)
  - [获取文档列表](#获取文档列表)
  - [上传文档](#上传文档)
- [知识库管理](#知识库管理)
  - [创建知识库](#创建知识库)
  - [获取知识库列表](#获取知识库列表)
  - [搜索知识库](#搜索知识库)
- [模型管理](#模型管理)
  - [获取模型列表](#获取模型列表)
  - [测试模型](#测试模型)
- [用户管理](#用户管理)
  - [获取当前用户信息](#获取当前用户信息)
  - [获取用户会话列表](#获取用户会话列表)
  - [获取用户统计信息](#获取用户统计信息)
- [系统管理](#系统管理)
  - [获取系统信息](#获取系统信息)
  - [获取系统配置](#获取系统配置)
  - [获取系统指标](#获取系统指标)
- [统计分析](#统计分析)
  - [获取仪表板数据](#获取仪表板数据)
  - [获取使用情况概览](#获取使用情况概览)

---

## 健康检查

### 系统健康检查

**路径:** `GET /api/health/`

**描述:** 检查系统整体健康状态，包括API服务、数据库、缓存等关键组件

**使用示例:**

```bash
curl -X GET http://localhost:8000/api/health/
```

```python
import requests; response = requests.get('http://localhost:8000/api/health/')
```

---

### 详细健康检查

**路径:** `GET /api/health/detailed`

**描述:** 获取详细的系统健康检查信息，包括性能指标

---

### Ping检查

**路径:** `GET /api/health/ping`

**描述:** 简单的ping检查，用于快速验证服务可用性

---

## 对话服务

### 对话完成

**路径:** `POST /api/chat/completions`

**描述:** 兼容OpenAI的对话接口，生成智能回复

**参数:**

- `messages` (array) - ✓ 对话消息列表 - 示例: `[{'role': 'user', 'content': '你好'}]`
- `model` (string) - ✗ 使用的模型名称 - 示例: `gpt-3.5-turbo`
- `stream` (boolean) - ✗ 是否流式返回

**使用示例:**

```bash

curl -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "model": "gpt-3.5-turbo"
  }'
                    
```

```python

import requests

response = requests.post(
    "http://localhost:8000/api/chat/completions",
    json={
        "messages": [{"role": "user", "content": "你好"}],
        "model": "gpt-3.5-turbo"
    }
)
                    
```

---

### 流式对话完成

**路径:** `POST /api/chat/completions/stream`

**描述:** 流式生成对话回复，实时返回生成内容

**参数:**

- `messages` (array) - ✓ 对话消息列表

---

## 文档管理

### 获取文档列表

**路径:** `GET /api/documents/`

**描述:** 分页获取文档列表，支持过滤

**参数:**

- `page` (integer) - ✗ 页码 - 示例: `1`
- `page_size` (integer) - ✗ 每页数量 - 示例: `20`

---

### 上传文档

**路径:** `POST /api/documents/upload`

**描述:** 上传文档到指定知识库

**参数:**

- `file` (file) - ✓ 上传的文件
- `knowledge_base_id` (string) - ✓ 知识库ID

---

## 知识库管理

### 创建知识库

**路径:** `POST /api/knowledge/`

**描述:** 创建新的知识库

**参数:**

- `name` (string) - ✓ 知识库名称
- `description` (string) - ✗ 知识库描述

---

### 获取知识库列表

**路径:** `GET /api/knowledge/`

**描述:** 获取知识库列表，支持分页

---

### 搜索知识库

**路径:** `POST /api/knowledge/search`

**描述:** 在知识库中搜索相关内容

**参数:**

- `query` (string) - ✓ 搜索查询
- `top_k` (integer) - ✗ 返回结果数量 - 示例: `10`

---

## 模型管理

### 获取模型列表

**路径:** `GET /api/models/`

**描述:** 获取可用的AI模型列表

**参数:**

- `type` (string) - ✗ 模型类型过滤 - 示例: `llm`

---

### 测试模型

**路径:** `POST /api/models/{model_id}/test`

**描述:** 测试指定模型的功能

**参数:**

- `model_id` (string) - ✓ 模型ID
- `input` (string) - ✓ 测试输入

---

## 用户管理

### 获取当前用户信息

**路径:** `GET /api/users/me`

**描述:** 获取当前用户的基本信息

---

### 获取用户会话列表

**路径:** `GET /api/users/me/sessions`

**描述:** 获取当前用户的会话列表

---

### 获取用户统计信息

**路径:** `GET /api/users/me/stats`

**描述:** 获取当前用户的统计信息

---

## 系统管理

### 获取系统信息

**路径:** `GET /api/system/info`

**描述:** 获取系统基本信息和状态

---

### 获取系统配置

**路径:** `GET /api/system/config`

**描述:** 获取当前系统配置（敏感信息已隐藏）

---

### 获取系统指标

**路径:** `GET /api/system/metrics`

**描述:** 获取系统性能指标

**参数:**

- `time_range` (string) - ✗ 时间范围 - 示例: `1h`

---

## 统计分析

### 获取仪表板数据

**路径:** `GET /api/analytics/dashboard`

**描述:** 获取仪表板概览数据

**参数:**

- `time_range` (string) - ✗ 时间范围 - 示例: `7d`

---

### 获取使用情况概览

**路径:** `GET /api/analytics/usage/overview`

**描述:** 获取系统使用情况概览

---

