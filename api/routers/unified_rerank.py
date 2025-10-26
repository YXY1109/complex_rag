"""
统一重新排序路由
迁移Sanic RAG服务的重排序功能到FastAPI
"""

import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from config.loguru_config import get_logger

# 创建路由器
router = APIRouter()
structured_logger = get_logger("api.unified_rerank")


# Pydantic模型定义
class RerankDocument(BaseModel):
    text: str = Field(..., description="文档文本")
    id: Optional[str] = Field(None, description="文档ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="文档元数据")


class RerankRequest(BaseModel):
    model: str = Field(..., description="使用的重排序模型")
    query: str = Field(..., description="查询文本")
    documents: List[RerankDocument] = Field(..., description="文档列表")
    top_n: Optional[int] = Field(None, description="返回的文档数量")


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: RerankDocument


class RerankResponse(BaseModel):
    model: str
    results: List[RerankResult]
    usage: Optional[Dict[str, int]] = None


@router.post("/", response_model=RerankResponse)
async def rerank_documents(
    request: Request,
    rerank_request: RerankRequest
) -> RerankResponse:
    """
    文档重排序接口

    Args:
        request: FastAPI请求对象
        rerank_request: 重排序请求

    Returns:
        RerankResponse: 重排序响应
    """
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time.time()

    try:
        structured_logger.info(
            "文档重排序请求",
            extra={
                "request_id": request_id,
                "model": rerank_request.model,
                "query_length": len(rerank_request.query),
                "documents_count": len(rerank_request.documents),
                "top_n": rerank_request.top_n,
            }
        )

        # TODO: 集成实际的重排序服务
        # 这里暂时返回模拟重排序结果
        response = await _create_mock_rerank(rerank_request, request_id, start_time)

        return response

    except Exception as e:
        structured_logger.error(
            f"文档重排序失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to rerank documents",
                "type": "internal_server_error",
                "code": "rerank_error",
                "request_id": request_id,
            }
        )


@router.get("/models")
async def list_rerank_models(request: Request) -> Dict[str, Any]:
    """
    列出可用的重排序模型

    Args:
        request: FastAPI请求对象

    Returns:
        Dict: 重排序模型列表响应
    """
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        # TODO: 从实际重排序服务获取模型列表
        # 这里返回模拟模型列表
        mock_models = [
            {
                "id": "bge-reranker-base",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "unified-rag-service",
                "rerank": {
                    "max_documents": 1000,
                    "max_query_length": 512,
                    "max_document_length": 2048,
                    "supports_batch": True,
                    "supports_custom_threshold": True,
                }
            },
            {
                "id": "Qwen3-Reranker-0.6B",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "unified-rag-service",
                "rerank": {
                    "max_documents": 500,
                    "max_query_length": 1024,
                    "max_document_length": 4096,
                    "supports_batch": True,
                    "supports_custom_threshold": True,
                }
            },
            {
                "id": "bce-reranker-base_v1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "unified-rag-service",
                "rerank": {
                    "max_documents": 200,
                    "max_query_length": 256,
                    "max_document_length": 1024,
                    "supports_batch": False,
                    "supports_custom_threshold": False,
                }
            }
        ]

        structured_logger.info(
            "重排序模型列表请求",
            extra={
                "request_id": request_id,
                "models_count": len(mock_models),
            }
        )

        return {
            "object": "list",
            "data": mock_models,
        }

    except Exception as e:
        structured_logger.error(
            f"获取重排序模型列表失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to retrieve rerank models",
                "type": "internal_server_error",
                "code": "models_error",
                "request_id": request_id,
            }
        )


async def _create_mock_rerank(
    rerank_request: RerankRequest,
    request_id: str,
    start_time: float
) -> RerankResponse:
    """
    创建模拟重排序响应

    Args:
        rerank_request: 重排序请求
        request_id: 请求ID
        start_time: 开始时间

    Returns:
        RerankResponse: 模拟响应
    """
    processing_time = time.time() - start_time

    # 生成模拟相关性分数并排序
    documents_with_scores = []
    for i, doc in enumerate(rerank_request.documents):
        # 简单的模拟算法：基于文档长度和查询匹配度生成分数
        base_score = 0.5 + (i * 0.1) % 0.5  # 0.5-1.0之间的分数

        # 如果查询出现在文档中，提高分数
        if rerank_request.query.lower() in doc.text.lower():
            base_score += 0.2

        # 添加一些随机性
        import random
        base_score += random.uniform(-0.1, 0.1)
        base_score = max(0.0, min(1.0, base_score))  # 限制在0-1之间

        documents_with_scores.append((i, base_score, doc))

    # 按分数排序
    documents_with_scores.sort(key=lambda x: x[1], reverse=True)

    # 应用top_n限制
    if rerank_request.top_n:
        documents_with_scores = documents_with_scores[:rerank_request.top_n]

    # 构建结果
    results = []
    for original_index, score, doc in documents_with_scores:
        result = RerankResult(
            index=original_index,
            relevance_score=score,
            document=doc
        )
        results.append(result)

    # 估算使用量
    total_chars = len(rerank_request.query) + sum(len(doc.text) for doc in rerank_request.documents)
    estimated_tokens = total_chars // 4  # 简单估算

    response = RerankResponse(
        model=rerank_request.model,
        results=results,
        usage={
            "prompt_tokens": estimated_tokens,
            "total_tokens": estimated_tokens,
        }
    )

    structured_logger.info(
        "文档重排序完成",
        extra={
            "request_id": request_id,
            "model": response.model,
            "documents_processed": len(rerank_request.documents),
            "results_returned": len(response.results),
            "processing_time_seconds": round(processing_time, 3),
            "throughput_docs_per_second": round(len(rerank_request.documents) / processing_time, 2),
        }
    )

    return response


structured_logger.info("统一重排序路由加载完成")