"""
统一嵌入路由
迁移Sanic RAG服务的嵌入功能到FastAPI
"""

import time
from typing import Dict, Any, List, Union, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from config.loguru_config import get_logger

# 创建路由器
router = APIRouter()
structured_logger = get_logger("api.unified_embeddings")


# Pydantic模型定义
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="输入文本或文本列表")
    model: str = Field(..., description="使用的模型")
    encoding_format: Optional[str] = Field("float", description="编码格式")
    dimensions: Optional[int] = Field(None, description="嵌入维度")
    user: Optional[str] = Field(None, description="用户标识")
    input_type: Optional[str] = Field(None, description="输入类型")


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class SimilarityRequest(BaseModel):
    text1: str = Field(..., description="第一个文本")
    text2: str = Field(..., description="第二个文本")
    model: Optional[str] = Field(None, description="使用的模型")
    metric: Optional[str] = Field("cosine", description="相似度计算方法")


class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="文本列表")
    model: Optional[str] = Field(None, description="使用的模型")
    input_type: Optional[str] = Field(None, description="输入类型")
    batch_size: Optional[int] = Field(32, description="批处理大小")


class ModelEmbeddingInfo(BaseModel):
    dimensions: int
    max_input_length: int
    max_batch_size: int
    supports_batch: bool
    supports_different_input_types: bool
    supports_custom_dimensions: bool


class EmbeddingModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    embedding: ModelEmbeddingInfo


@router.post("/", response_model=EmbeddingResponse)
async def create_embeddings(
    request: Request,
    embedding_request: EmbeddingRequest
) -> EmbeddingResponse:
    """
    OpenAI兼容的文本向量化接口

    Args:
        request: FastAPI请求对象
        embedding_request: 向量化请求

    Returns:
        EmbeddingResponse: 向量化响应
    """
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time.time()

    try:
        # 处理输入文本
        input_texts = embedding_request.input
        if isinstance(input_texts, str):
            input_texts = [input_texts]
            input_count = 1
            input_length = len(input_texts[0])
        else:
            input_count = len(input_texts)
            input_length = sum(len(text) for text in input_texts)

        structured_logger.info(
            "向量化请求",
            extra={
                "request_id": request_id,
                "model": embedding_request.model,
                "input_count": input_count,
                "input_length": input_length,
                "encoding_format": embedding_request.encoding_format,
                "dimensions": embedding_request.dimensions,
                "input_type": embedding_request.input_type,
            }
        )

        # TODO: 集成实际的嵌入服务
        # 这里暂时返回模拟嵌入向量
        response = await _create_mock_embeddings(embedding_request, request_id, start_time)

        return response

    except Exception as e:
        structured_logger.error(
            f"向量化请求处理失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "An unexpected error occurred",
                "type": "internal_server_error",
                "code": "internal_error",
                "request_id": request_id,
            }
        )


@router.get("/models")
async def list_embedding_models(request: Request) -> Dict[str, Any]:
    """
    列出可用的向量化模型

    Args:
        request: FastAPI请求对象

    Returns:
        Dict: 向量化模型列表响应
    """
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        # TODO: 从实际嵌入服务获取模型列表
        # 这里返回模拟模型列表
        mock_models = [
            {
                "id": "text-embedding-ada-002",
                "embedding": {
                    "dimensions": 1536,
                    "max_input_length": 8192,
                    "max_batch_size": 100,
                    "supports_batch": True,
                    "supports_different_input_types": True,
                    "supports_custom_dimensions": False,
                }
            },
            {
                "id": "Qwen3-Embedding-0.6B",
                "embedding": {
                    "dimensions": 768,
                    "max_input_length": 4096,
                    "max_batch_size": 64,
                    "supports_batch": True,
                    "supports_different_input_types": True,
                    "supports_custom_dimensions": False,
                }
            },
            {
                "id": "bge-large-zh-v1.5",
                "embedding": {
                    "dimensions": 1024,
                    "max_input_length": 512,
                    "max_batch_size": 32,
                    "supports_batch": True,
                    "supports_different_input_types": False,
                    "supports_custom_dimensions": False,
                }
            }
        ]

        models = []
        for model_data in mock_models:
            model_info = EmbeddingModelInfo(
                id=model_data["id"],
                created=int(time.time()),
                owned_by="unified-rag-service",
                embedding=ModelEmbeddingInfo(**model_data["embedding"])
            )
            models.append(model_info.dict())

        structured_logger.info(
            "向量化模型列表请求",
            extra={
                "request_id": request_id,
                "models_count": len(models),
            }
        )

        return {
            "object": "list",
            "data": models,
        }

    except Exception as e:
        structured_logger.error(
            f"获取向量化模型列表失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to retrieve embedding models",
                "type": "internal_server_error",
                "code": "models_error",
                "request_id": request_id,
            }
        )


@router.post("/similarity")
async def compute_similarity(
    request: Request,
    similarity_request: SimilarityRequest
) -> Dict[str, Any]:
    """
    计算两个文本的相似度

    Args:
        request: FastAPI请求对象
        similarity_request: 相似度计算请求

    Returns:
        Dict: 相似度计算结果
    """
    request_id = getattr(request.state, "request_id", "unknown")

    try:
        structured_logger.info(
            "相似度计算请求",
            extra={
                "request_id": request_id,
                "model": similarity_request.model,
                "metric": similarity_request.metric,
                "text1_length": len(similarity_request.text1),
                "text2_length": len(similarity_request.text2),
            }
        )

        # TODO: 集成实际的相似度计算
        # 这里返回模拟相似度分数
        mock_similarity_score = 0.85  # 模拟相似度分数

        structured_logger.info(
            "相似度计算完成",
            extra={
                "request_id": request_id,
                "model": similarity_request.model or "default",
                "metric": similarity_request.metric,
                "similarity_score": mock_similarity_score,
            }
        )

        return {
            "similarity_score": mock_similarity_score,
            "metric": similarity_request.metric,
            "model": similarity_request.model or "default",
            "text1_length": len(similarity_request.text1),
            "text2_length": len(similarity_request.text2),
            "embedding_dimensions": 768,  # 模拟嵌入维度
        }

    except Exception as e:
        structured_logger.error(
            f"相似度计算失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to compute similarity",
                "type": "internal_server_error",
                "code": "similarity_error",
                "request_id": request_id,
            }
        )


@router.post("/batch")
async def batch_embeddings(
    request: Request,
    batch_request: BatchEmbeddingRequest
) -> Dict[str, Any]:
    """
    批量向量化接口

    Args:
        request: FastAPI请求对象
        batch_request: 批量向量化请求

    Returns:
        Dict: 批量向量化响应
    """
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time.time()

    try:
        if len(batch_request.texts) == 0:
            return {
                "object": "list",
                "data": [],
                "model": batch_request.model or "default",
                "usage": {
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                }
            }

        structured_logger.info(
            "批量向量化请求",
            extra={
                "request_id": request_id,
                "model": batch_request.model,
                "texts_count": len(batch_request.texts),
                "batch_size": batch_request.batch_size,
                "input_type": batch_request.input_type,
                "total_chars": sum(len(text) for text in batch_request.texts),
            }
        )

        # TODO: 集成实际的批量向量化服务
        # 这里返回模拟嵌入向量
        response = await _create_mock_batch_embeddings(batch_request, request_id, start_time)

        return response

    except Exception as e:
        structured_logger.error(
            f"批量向量化失败: {e}",
            extra={
                "request_id": request_id,
                "error": str(e),
            }
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to process batch embeddings",
                "type": "internal_server_error",
                "code": "batch_error",
                "request_id": request_id,
            }
        )


async def _create_mock_embeddings(
    embedding_request: EmbeddingRequest,
    request_id: str,
    start_time: float
) -> EmbeddingResponse:
    """
    创建模拟嵌入向量响应

    Args:
        embedding_request: 嵌入请求
        request_id: 请求ID
        start_time: 开始时间

    Returns:
        EmbeddingResponse: 模拟响应
    """
    processing_time = time.time() - start_time

    # 处理输入文本
    input_texts = embedding_request.input
    if isinstance(input_texts, str):
        input_texts = [input_texts]

    # 创建模拟嵌入向量
    data = []
    for i, text in enumerate(input_texts):
        # 生成固定长度的模拟嵌入向量
        embedding_dimensions = embedding_request.dimensions or 1536
        mock_embedding = [0.1 + (i * 0.01) % 1.0] * embedding_dimensions

        embedding_data = EmbeddingData(
            embedding=mock_embedding,
            index=i
        )
        data.append(embedding_data)

    # 估算token使用量
    total_chars = sum(len(text) for text in input_texts)
    estimated_tokens = total_chars // 4  # 简单估算

    response = EmbeddingResponse(
        data=data,
        model=embedding_request.model,
        usage=EmbeddingUsage(
            prompt_tokens=estimated_tokens,
            total_tokens=estimated_tokens
        )
    )

    structured_logger.info(
        "向量化完成",
        extra={
            "request_id": request_id,
            "model": response.model,
            "embeddings_count": len(response.data),
            "embedding_dimensions": len(response.data[0].embedding) if response.data else 0,
            "prompt_tokens": response.usage.prompt_tokens,
            "processing_time_seconds": round(processing_time, 3),
        }
    )

    return response


async def _create_mock_batch_embeddings(
    batch_request: BatchEmbeddingRequest,
    request_id: str,
    start_time: float
) -> Dict[str, Any]:
    """
    创建模拟批量嵌入向量响应

    Args:
        batch_request: 批量嵌入请求
        request_id: 请求ID
        start_time: 开始时间

    Returns:
        Dict: 批量嵌入响应
    """
    processing_time = time.time() - start_time

    # 创建模拟嵌入向量
    data = []
    for i, text in enumerate(batch_request.texts):
        # 生成固定长度的模拟嵌入向量
        mock_embedding = [0.2 + (i * 0.02) % 1.0] * 768

        embedding_data = {
            "object": "embedding",
            "embedding": mock_embedding,
            "index": i,
        }
        data.append(embedding_data)

    # 估算token使用量
    total_chars = sum(len(text) for text in batch_request.texts)
    estimated_tokens = total_chars // 4  # 简单估算

    structured_logger.info(
        "批量向量化完成",
        extra={
            "request_id": request_id,
            "model": batch_request.model or "default",
            "embeddings_count": len(data),
            "embedding_dimensions": len(data[0]["embedding"]) if data else 0,
            "estimated_tokens": estimated_tokens,
            "processing_time_seconds": round(processing_time, 3),
            "throughput_texts_per_second": round(len(batch_request.texts) / processing_time, 2),
        }
    )

    return {
        "object": "list",
        "data": data,
        "model": batch_request.model or "default",
        "usage": {
            "prompt_tokens": estimated_tokens,
            "total_tokens": estimated_tokens,
        },
        "processing_info": {
            "batch_size": batch_request.batch_size,
            "processing_time_seconds": round(processing_time, 3),
            "throughput_texts_per_second": round(len(batch_request.texts) / processing_time, 2),
        }
    }


structured_logger.info("统一嵌入路由加载完成")