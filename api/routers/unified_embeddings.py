"""
统一嵌入路由
迁移Sanic RAG服务的嵌入功能到FastAPI
"""

import time
import asyncio
from typing import Dict, Any, List, Union, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from config.loguru_config import get_logger
from services.unified_embedding_service import UnifiedEmbeddingService
from config.settings_simple import get_settings

# 创建路由器
router = APIRouter()
structured_logger = get_logger("api.unified_embeddings")

# 全局服务实例
_embedding_service: Optional[UnifiedEmbeddingService] = None
_settings = get_settings()

async def get_embedding_service() -> UnifiedEmbeddingService:
    """获取统一嵌入服务实例"""
    global _embedding_service
    if _embedding_service is None:
        try:
            _embedding_service = UnifiedEmbeddingService()
            await _embedding_service.initialize()
            structured_logger.info("统一嵌入服务初始化成功")
        except Exception as e:
            structured_logger.error(f"统一嵌入服务初始化失败: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Embedding service initialization failed",
                    "type": "service_initialization_error",
                    "code": "embedding_service_error"
                }
            )
    return _embedding_service


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

        # 集成实际的嵌入服务
        embedding_service = await get_embedding_service()

        # 处理输入文本
        input_texts = embedding_request.input
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        # 确定使用的模型/提供商
        model_name = embedding_request.model
        provider_name = _settings.embedding_default_provider

        # 如果指定了特定的模型，尝试映射到提供商
        if model_name:
            if "qwen" in model_name.lower():
                provider_name = "qwen3"
            elif "bce" in model_name.lower() or "bge" in model_name.lower():
                provider_name = "bce"
            else:
                provider_name = "generic"

        # 生成嵌入向量
        embeddings = await embedding_service.embed_texts(
            texts=input_texts,
            provider_name=provider_name,
            model_name=model_name,
            input_type=embedding_request.input_type
        )

        # 构造响应
        data = []
        for i, embedding in enumerate(embeddings):
            embedding_data = EmbeddingData(
                embedding=embedding,
                index=i
            )
            data.append(embedding_data)

        # 估算token使用量
        total_chars = sum(len(text) for text in input_texts)
        estimated_tokens = total_chars // 4  # 简单估算

        response = EmbeddingResponse(
            data=data,
            model=model_name,
            usage=EmbeddingUsage(
                prompt_tokens=estimated_tokens,
                total_tokens=estimated_tokens
            )
        )

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
        # 从实际嵌入服务获取模型列表
        embedding_service = await get_embedding_service()
        available_providers = embedding_service.list_providers()

        models = []
        for provider_name, provider in available_providers.items():
            model_info = EmbeddingModelInfo(
                id=f"{provider_name}-{provider.name}",
                created=int(time.time()),
                owned_by="unified-rag-service",
                embedding=ModelEmbeddingInfo(
                    dimensions=provider.dimension,
                    max_input_length=provider.max_seq_length,
                    max_batch_size=provider.batch_size,
                    supports_batch=True,
                    supports_different_input_types=True,
                    supports_custom_dimensions=False,
                )
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

        # 集成实际的相似度计算
        embedding_service = await get_embedding_service()

        # 确定使用的提供商
        provider_name = _settings.embedding_default_provider
        if similarity_request.model:
            if "qwen" in similarity_request.model.lower():
                provider_name = "qwen3"
            elif "bce" in similarity_request.model.lower() or "bge" in similarity_request.model.lower():
                provider_name = "bce"
            else:
                provider_name = "generic"

        # 计算相似度
        similarity_score = await embedding_service.compute_similarity(
            text1=similarity_request.text1,
            text2=similarity_request.text2,
            provider_name=provider_name,
            metric=similarity_request.metric
        )

        structured_logger.info(
            "相似度计算完成",
            extra={
                "request_id": request_id,
                "model": similarity_request.model or provider_name,
                "provider": provider_name,
                "metric": similarity_request.metric,
                "similarity_score": similarity_score,
                "embedding_dimensions": embedding_dimensions,
            }
        )

        # 获取提供商信息以提供准确的维度信息
        providers = embedding_service.list_providers()
        provider = providers.get(provider_name) or providers.get("generic")
        embedding_dimensions = getattr(provider, 'dimension', 768) if provider else 768

        return {
            "similarity_score": similarity_score,
            "metric": similarity_request.metric,
            "model": similarity_request.model or provider_name,
            "provider": provider_name,
            "text1_length": len(similarity_request.text1),
            "text2_length": len(similarity_request.text2),
            "embedding_dimensions": embedding_dimensions,
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

        # 集成实际的批量向量化服务
        embedding_service = await get_embedding_service()

        # 确定使用的提供商
        provider_name = _settings.embedding_default_provider
        if batch_request.model:
            if "qwen" in batch_request.model.lower():
                provider_name = "qwen3"
            elif "bce" in batch_request.model.lower() or "bge" in batch_request.model.lower():
                provider_name = "bce"
            else:
                provider_name = "generic"

        # 生成批量嵌入向量
        embeddings = await embedding_service.embed_texts(
            texts=batch_request.texts,
            provider_name=provider_name,
            model_name=batch_request.model,
            input_type=batch_request.input_type,
            batch_size=batch_request.batch_size
        )

        # 构造响应
        data = []
        for i, embedding in enumerate(embeddings):
            embedding_data = {
                "object": "embedding",
                "embedding": embedding,
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
                "model": batch_request.model or provider_name,
                "provider": provider_name,
                "embeddings_count": len(data),
                "embedding_dimensions": len(data[0]["embedding"]) if data else 0,
                "estimated_tokens": estimated_tokens,
                "processing_time_seconds": round(time.time() - start_time, 3),
                "throughput_texts_per_second": round(len(batch_request.texts) / (time.time() - start_time), 2),
            }
        )

        response = {
            "object": "list",
            "data": data,
            "model": batch_request.model or provider_name,
            "usage": {
                "prompt_tokens": estimated_tokens,
                "total_tokens": estimated_tokens,
            },
            "processing_info": {
                "batch_size": batch_request.batch_size,
                "processing_time_seconds": round(time.time() - start_time, 3),
                "throughput_texts_per_second": round(len(batch_request.texts) / (time.time() - start_time), 2),
                "provider": provider_name,
            }
        }

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




structured_logger.info("统一嵌入路由加载完成")